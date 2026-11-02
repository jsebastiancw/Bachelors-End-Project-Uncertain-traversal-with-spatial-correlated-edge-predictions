from imports import *
from network_operations import get_travel_time
from simulated_annealing import simulated_annealing


def initialize_theta(G, theta, shortest_path):
    """
    Warm-starts `theta` along the shortest path edges.
    Supports both PyTorch (CUDA/CPU) tensors and NumPy arrays.
    """
    edge_to_idx = G.graph.get('edge_to_idx')
    if edge_to_idx is None:
        all_edges = list(G.edges())
        G.graph['all_edges'] = all_edges
        edge_to_idx = {e: idx for idx, e in enumerate(all_edges)}
        G.graph['edge_to_idx'] = edge_to_idx

    for i in range(len(shortest_path) - 1):
        specified_edge = (shortest_path[i], shortest_path[i + 1])
        if specified_edge in edge_to_idx:
            index = edge_to_idx[specified_edge]
            theta[index] = 1.5
    return theta


def initialize_theta2(G, origin, destination, theta, time_limit, shortest_path, alpha):
    shortest_path, shortest_time = simulated_annealing(G, origin, destination, 100, 0.95, 1000)
    time_limit = 1.1 * shortest_time

    edge_to_idx = G.graph.get('edge_to_idx')
    if edge_to_idx is None:
        all_edges = list(G.edges())
        edge_to_idx = {e: idx for idx, e in enumerate(all_edges)}
        G.graph['edge_to_idx'] = edge_to_idx
    num_edges = len(edge_to_idx)

    is_tensor = TORCH_AVAILABLE and isinstance(theta, torch.Tensor)
    theta_np = theta.detach().cpu().numpy().copy() if is_tensor else np.asarray(theta, dtype=np.float64).copy()

    time_budget = float(time_limit)
    grad_time = np.zeros(num_edges, dtype=np.float64)
    grad_base = np.zeros(num_edges, dtype=np.float64)

    for i in range(len(shortest_path) - 1):
        current_node = shortest_path[i]
        next_node = shortest_path[i + 1]
        if time_budget < 0.0:
            break

        available_edges = [v for _, v in G.out_edges(current_node)]
        if not available_edges or next_node not in available_edges:
            break

        avail_indices = np.asarray([edge_to_idx[(current_node, v)] for v in available_edges], dtype=np.int64)
        logits = time_budget * theta_np[avail_indices] + theta_np[num_edges + avail_indices]
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits)

        chosen_index = available_edges.index(next_node)
        chosen_edge_idx = avail_indices[chosen_index]

        grad_time[chosen_edge_idx] += time_budget
        grad_base[chosen_edge_idx] += 1.0
        grad_time[avail_indices] -= probabilities * time_budget
        grad_base[avail_indices] -= probabilities

        travel_time = get_travel_time(G[current_node][next_node]['weight'], G[current_node][next_node]['sigma'])
        time_budget -= travel_time

    gradient = np.concatenate([grad_time, grad_base])
    if is_tensor:
        theta = theta + alpha * torch.as_tensor(gradient, dtype=theta.dtype, device=theta.device)
    else:
        theta = theta + alpha * gradient
    return theta
