import os
import torch
from Embedding_dijkandrandom import Node2Vec
from GAT_zzt_725 import GAT


class Simulation:
    """
    Modernized MARVEL Multi-Agent Evaluation & Rollout Simulator on CUDA.
    """
    def __init__(self, data_name: str, emb_size: int, robots: list, current_agent: int, device: torch.device):
        self.data_name = data_name
        self.robots = robots
        self.goal_nodes = [robot['goal_node'] for robot in robots]
        self.device = device
        self.emb_size = emb_size
        self.current_agent = current_agent

        self.node2vec = Node2Vec(data_name, emb_size=emb_size, device=device)
        self.num_nodes = self.node2vec.G.number_of_nodes()
        self.node_mapping = {node: idx for idx, node in enumerate(self.node2vec.G.nodes())}
        self.idx_to_node = {idx: node for node, idx in self.node_mapping.items()}

        self.initial_adj_matrix = self.generate_adj_matrix().to(device)
        self.adj_matrix = self.initial_adj_matrix.clone()
        self.node_features = self.generate_node_features().to(device)
        self.new_node_features = self.node_features.clone()
        self.visited_nodes = set()

    def generate_adj_matrix(self) -> torch.Tensor:
        adj_matrix = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32, device=self.device)
        for u, v, data in self.node2vec.G.edges(data=True):
            adj_matrix[self.node_mapping[u], self.node_mapping[v]] = float(data['probability'])
        return adj_matrix

    def generate_node_features(self) -> torch.Tensor:
        w2v = self.node2vec.train(
            self.adj_matrix,
            self.node_mapping,
            self.goal_nodes,
            is_loadmodel=False,
            is_loaddata=False,
        )
        node_features = torch.zeros((self.num_nodes, self.emb_size + 40), dtype=torch.float32, device=self.device)
        node_features[:, :self.emb_size] = w2v.embeddings.to(self.device)
        return node_features

    def update_node_features(self, robot_num: int, current_node: int, goal_node: int):
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

    def update_adj_matrix(self, current_node_idx: int):
        neighbors = torch.where(self.adj_matrix[current_node_idx] > 0)[0]
        for neighbor in neighbors:
            prob = self.adj_matrix[current_node_idx, neighbor].item()
            if 0.0 < prob < 1.0:
                new_prob = 1.0 if torch.rand(1, device=self.device).item() < prob else 0.0
                self.adj_matrix[current_node_idx, neighbor] = new_prob

    def run_simulation(self, model_path: str = None) -> int:
        in_channels = self.emb_size + 40
        hidden_channels = 2 * in_channels
        out_channels = 1
        learning_rate = 1e-4

        model = GAT(
            self.data_name,
            learning_rate,
            in_channels,
            hidden_channels,
            out_channels,
            heads=6,
            use_wandb=False,
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
        model.eval()

        robot_num = self.current_agent
        current_node = self.robots[robot_num - 1]['current_node']
        goal_node = self.robots[robot_num - 1]['goal_node']
        valid_edges = self.adj_matrix[self.node_mapping[current_node]] == 1.0
        next_nodes = torch.where(valid_edges)[0]

        if len(next_nodes) == 0:
            print(f"[Simulation] No valid edges for node {current_node}")
            return current_node

        if len(next_nodes) == 1:
            next_node = next_nodes[0].item()
            output_node = self.idx_to_node[next_node]
        else:
            self.update_node_features(robot_num, current_node, goal_node)
            edge_index = torch.stack(torch.where(self.adj_matrix == 1.0)).to(self.device)
            from_node = self.node_mapping[current_node]
            current_node_indices = torch.tensor([from_node], dtype=torch.long, device=self.device)

            with torch.no_grad():
                output = model(self.node_features, edge_index, current_node_indices)
            next_node_outputs = output[next_nodes]

            max_index = torch.argmax(next_node_outputs)
            selected_next_node = next_nodes[max_index].item()
            output_node = self.idx_to_node[selected_next_node]

        self.update_adj_matrix(self.node_mapping[current_node])
        self.node_features = self.generate_node_features().to(self.device)
        self.new_node_features = self.node_features.clone()

        print(f"Robot {robot_num} | Current node: {current_node} -> Next node: {output_node}")
        return output_node


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.join(base_dir, 'models', 'marvel_simple_gat.pth')

    robots = [
        {'robot_num': 1, 'current_node': 9, 'next_node': 9, 'goal_node': 12},
        {'robot_num': 2, 'current_node': 1, 'next_node': 1, 'goal_node': 8},
    ]

    max_steps = 50
    step_count = 0
    while not all(robot['current_node'] == robot['goal_node'] for robot in robots) and step_count < max_steps:
        for robot in robots:
            current_agent = robot['robot_num']
            if robot['current_node'] != robot['goal_node']:
                sim = Simulation(
                    data_name='Simple',
                    emb_size=128,
                    robots=robots,
                    current_agent=current_agent,
                    device=device,
                )
                output_node = sim.run_simulation(model_path=default_model_path)
                robots[current_agent - 1]['current_node'] = output_node
        step_count += 1

    if all(robot['current_node'] == robot['goal_node'] for robot in robots):
        print("All robots reached their goal nodes.")
    else:
        print(f"Simulation terminated after {max_steps} steps.")
