import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from Embedding_dijkandrandom import Node2Vec


class Env:
    """
    Modernized MARVEL Multi-Agent Canadian Traveller Problem (MACTP) Environment.
    Integrates dynamic GPU graph embeddings, information entropy reduction (Delta H),
    and expert-augmented policy gradient updates on CUDA.
    """
    def __init__(self, data_name: str, model_path: str, emb_size: int, robots: list, device: torch.device):
        self.device = device
        self.robots = robots
        for robot in self.robots:
            if isinstance(robot['time_spent'], torch.Tensor):
                robot['time_spent'] = robot['time_spent'].detach().clone().to( dtype=torch.float32, device=self.device)
            else:
                robot['time_spent'] = torch.tensor(float(robot['time_spent']), dtype=torch.float32, device=self.device)

        self.goal_nodes = [robot['goal_node'] for robot in robots]
        self.node2vec = Node2Vec(data_name, emb_size=emb_size, device=self.device)
        self.num_nodes = self.node2vec.G.number_of_nodes()
        self.node_mapping = {node: idx for idx, node in enumerate(self.node2vec.G.nodes())}
        self.idx_to_node = {idx: node for node, idx in self.node_mapping.items()}

        self.initial_adj_matrix = self.generate_adj_matrix().to(self.device)
        self.adj_matrix = self.initial_adj_matrix.clone()

        self.original_node_features = self.generate_node_features(model_path, emb_size).to(self.device)
        self.node_features = self.original_node_features.clone()
        self.new_node_features = self.node_features.clone()

        self.global_time_spent = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.visited_nodes = set()
        self.mean_edge_weights = self.initialize_edge_weights()

    def generate_adj_matrix(self) -> torch.Tensor:
        adj_matrix = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32, device=self.device)
        for u, v, data in self.node2vec.G.edges(data=True):
            adj_matrix[self.node_mapping[u], self.node_mapping[v]] = float(data['probability'])
        return adj_matrix

    def generate_node_features(self, model_path: str, emb_size: int) -> torch.Tensor:
        w2v = self.node2vec.train(
            self.adj_matrix,
            self.node_mapping,
            self.goal_nodes,
            is_loadmodel=False,
            is_loaddata=False,
        )
        node_features = torch.zeros((self.num_nodes, emb_size + 40), dtype=torch.float32, device=self.device)
        node_features[:, :emb_size] = w2v.embeddings.to(self.device)
        return node_features

    def update_node_features(self, robot_num: int, current_node: int, goal_node: int):
        """
        Encodes active and peer robot positions/goals into the 40-dim agent state suffix
        on `self.device` without in-place mutation of autograd leaf tensors.
        """
        updated = self.new_node_features.detach().clone().to(self.device)
        current_node_idx = self.node_mapping[current_node]
        goal_node_idx = self.node_mapping[goal_node]

        updated[current_node_idx, -40:-30] = robot_num / 10.0
        updated[goal_node_idx, -30:-20] = robot_num / 10.0

        for robot in self.robots:
            if robot['robot_num'] != robot_num:
                other_goal_node_idx = self.node_mapping[robot['goal_node']]
                other_current_node_idx = self.node_mapping[robot['current_node']]
                updated[other_current_node_idx, -20:-10] = robot['robot_num'] / 10.0
                updated[other_goal_node_idx, -10:] = robot['robot_num'] / 10.0

        self.node_features = updated

    def initialize_edge_weights(self) -> dict:
        mean_edge_weights = {}
        for u, v, data in self.node2vec.G.edges(data=True):
            mean_cost = torch.tensor(float(data['weight']), dtype=torch.float32, device=self.device)
            sigma = float(np.random.uniform(0.2, 0.4)) * mean_cost
            mean_edge_weights[(u, v)] = (mean_cost, sigma)
        return mean_edge_weights

    def sample_edge_cost(self, u: int, v: int) -> torch.Tensor:
        mean_cost, sigma = self.mean_edge_weights[(u, v)]
        sampled = torch.normal(mean=mean_cost, std=sigma)
        return torch.clamp(sampled, min=mean_cost * 0.7)

    def calculate_total_success_probability(self, T, means: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
        mu_total = torch.sum(means)
        sigma_total = torch.sqrt(torch.sum(torch.square(sigmas)) + 1e-8)
        T_tensor = T if isinstance(T, torch.Tensor) else torch.tensor(float(T), dtype=torch.float32, device=self.device)
        z = (T_tensor.to(self.device) - mu_total) / sigma_total
        return torch.sigmoid(z)

    def calculate_prob(self, adj_matrix: torch.Tensor, robots: list, mean_cost: torch.Tensor, rob_num: int) -> torch.Tensor:
        probabilities = []
        adj_cpu = adj_matrix.detach().cpu().numpy()
        temp_graph = self.node2vec.G.copy()

        for u, v, data in temp_graph.edges(data=True):
            if adj_cpu[self.node_mapping[u], self.node_mapping[v]] == 1.0:
                temp_graph[u][v]['weight'] = data['weight']
            else:
                temp_graph[u][v]['weight'] = float('inf')

        for robot in robots:
            if robot['robot_num'] == rob_num:
                new_time_spent = robot['time_spent'] + mean_cost
            else:
                new_time_spent = robot['time_spent']

            path_cost = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            path_sigmas = []
            current_node = robot['current_node']
            goal_node = robot['goal_node']

            try:
                path = nx.dijkstra_path(temp_graph, source=current_node, target=goal_node, weight='weight')
            except nx.NetworkXNoPath:
                probabilities.append(torch.tensor(0.0, dtype=torch.float32, device=self.device))
                continue

            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = self.node2vec.G.get_edge_data(u, v)
                if edge_data and adj_cpu[self.node_mapping[u], self.node_mapping[v]] == 1.0:
                    m_cost, sigma = self.mean_edge_weights[(u, v)]
                    path_cost = path_cost + m_cost
                    path_sigmas.append(sigma)

            if len(path_sigmas) > 0:
                sigmas_tensor = torch.stack(path_sigmas).to(self.device)
                prob = robot['importance'] * self.calculate_total_success_probability(
                    robot['max_time'] - new_time_spent, path_cost, sigmas_tensor
                )
                probabilities.append(prob)

        if not probabilities:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        return torch.sum(torch.stack(probabilities))

    def calculate_information_entropy(self, p: float) -> torch.Tensor:
        if p > 0.0:
            p_tensor = torch.tensor(float(p), dtype=torch.float32, device=self.device)
            return p_tensor * torch.log(p_tensor)
        return torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def update_adj_matrix_and_entropy(self, current_node_idx: int) -> torch.Tensor:
        neighbors = torch.where(self.adj_matrix[current_node_idx] > 0)[0]
        entropies = []
        for neighbor in neighbors:
            prob = self.adj_matrix[current_node_idx, neighbor].item()
            if 0.0 < prob < 1.0:
                new_prob = 1.0 if torch.rand(1, device=self.device).item() < prob else 0.0
                adj_matrix_clone = self.adj_matrix.clone()
                adj_matrix_clone[current_node_idx, neighbor] = new_prob
                entropies.append(self.calculate_information_entropy(prob))
                self.adj_matrix = adj_matrix_clone

        if not entropies:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        return torch.sum(torch.stack(entropies))

    def get_action(self, model: nn.Module, robot: dict, device: torch.device):
        """
        Computes the expert action that maximizes expected on-time arrival probability
        plus information entropy gain (-Delta H).
        """
        current_node = robot['current_node']
        goal_node = robot['goal_node']
        rob_num = robot['robot_num']

        valid_edges = self.adj_matrix[self.node_mapping[current_node]] == 1.0
        next_nodes = torch.where(valid_edges)[0]

        if len(next_nodes) == 0:
            curr_idx = torch.tensor(self.node_mapping[current_node], dtype=torch.long, device=self.device)
            return curr_idx, curr_idx.unsqueeze(0)

        with torch.no_grad():
            probs_entropies = []
            original_adj_matrix = self.adj_matrix.clone()

            for next_node in next_nodes:
                trans_next_node = self.idx_to_node[next_node.item()]
                adj_matrix_clone = self.adj_matrix.clone()

                if trans_next_node not in self.visited_nodes:
                    entropy = self.update_adj_matrix_and_entropy(next_node.item())
                else:
                    entropy = torch.tensor(0.0, dtype=torch.float32, device=self.device)

                edge_data = self.node2vec.G.get_edge_data(current_node, trans_next_node)
                if edge_data:
                    mean_cost, _ = self.mean_edge_weights[(current_node, trans_next_node)]
                    robot['current_node'] = trans_next_node
                    if trans_next_node == goal_node:
                        prob_next = torch.tensor(1.0, dtype=torch.float32, device=self.device)
                    else:
                        prob_next = self.calculate_prob(self.adj_matrix, self.robots, mean_cost, rob_num)

                    self.adj_matrix = adj_matrix_clone
                    prob_entropy = prob_next - entropy
                    probs_entropies.append(prob_entropy)

                self.adj_matrix = original_adj_matrix.clone()

            robot['current_node'] = current_node

            if not probs_entropies:
                probs_entropies.append(torch.tensor(0.0, dtype=torch.float32, device=self.device))

            scores_tensor = torch.stack(probs_entropies).to(self.device)
            max_index = torch.argmax(scores_tensor)
            max_next_node = next_nodes[max_index]

        return max_next_node, next_nodes

    def custom_cross_entropy_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-12
        return -torch.sum(
            targets * torch.log(predictions + epsilon) + (1.0 - targets) * torch.log(1.0 - predictions + epsilon)
        )

    def step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        model_path: str,
        emb_size: int,
    ):
        total_reward = 0.0
        done = False
        success = False
        step_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        for robot in self.robots:
            if robot['current_node'] == robot['goal_node']:
                continue

            optimal_next_node, next_nodes = self.get_action(model, robot, device)
            from_node_idx = self.node_mapping[robot['current_node']]
            self.update_node_features(robot['robot_num'], robot['current_node'], robot['goal_node'])

            if len(next_nodes) == 1:
                next_node_idx = next_nodes[0].item()
                next_node = self.idx_to_node[next_node_idx]
            else:
                optimizer.zero_grad()
                edge_index = torch.stack(torch.where(self.adj_matrix == 1.0)).to(device)
                current_node_indices = torch.tensor([from_node_idx], dtype=torch.long, device=device)

                output = model(self.node_features.to(device), edge_index, current_node_indices)
                selected_values = output[next_nodes]
                q_values_flattened = selected_values.view(1, -1)

                optimal_next_node_index = (next_nodes == optimal_next_node).nonzero(as_tuple=True)[0]
                optimal_next_node_one_hot = torch.zeros(len(next_nodes), dtype=torch.float32, device=device)
                optimal_next_node_one_hot[optimal_next_node_index] = 1.0
                optimal_next_node_one_hot = optimal_next_node_one_hot.view(1, -1)

                loss = criterion(q_values_flattened, optimal_next_node_one_hot)
                loss.backward()
                optimizer.step()
                step_loss = step_loss + loss.detach()

                with torch.no_grad():
                    softmax_probs = F.softmax(selected_values.view(-1), dim=0)
                    sampled_idx = torch.multinomial(softmax_probs, num_samples=1).item()
                    next_node_idx = next_nodes[sampled_idx].item()
                    next_node = self.idx_to_node[next_node_idx]

            to_node = next_node
            from_node = self.idx_to_node[from_node_idx]
            edge_data = self.node2vec.G.get_edge_data(from_node, to_node)
            if edge_data is not None:
                time_cost = self.sample_edge_cost(from_node, to_node)
            else:
                time_cost = torch.tensor(100.0, dtype=torch.float32, device=self.device)

            robot['time_spent'] = robot['time_spent'] + time_cost

            if robot['time_spent'].item() > robot['max_time']:
                done = True
                break

            robot['current_node'] = next_node
            if robot['current_node'] not in self.visited_nodes:
                entropy = self.update_adj_matrix_and_entropy(self.node_mapping[robot['current_node']])
                if entropy.item() != 0.0:
                    self.node_features = self.generate_node_features(model_path, emb_size).to(device)
                    self.new_node_features = self.node_features.clone()
            self.visited_nodes.add(next_node)

        if all(robot['current_node'] == robot['goal_node'] for robot in self.robots):
            success = True
            done = True
            total_reward = 1.0
        elif any(robot['time_spent'].item() > robot['max_time'] for robot in self.robots):
            done = True

        return total_reward, done, step_loss, success

    def reset(self):
        self.adj_matrix = self.initial_adj_matrix.detach().clone().to(self.device)
        self.node_features = self.original_node_features.detach().clone().to(self.device)
        self.new_node_features = self.node_features.detach().clone().to(self.device)
        self.visited_nodes = set()
        for robot in self.robots:
            robot['current_node'] = robot['start_node']
            robot['time_spent'] = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        return self.node_features
