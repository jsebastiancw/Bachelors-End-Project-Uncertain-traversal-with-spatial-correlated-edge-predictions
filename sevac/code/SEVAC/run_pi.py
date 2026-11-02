from imports import *
from network_operations import get_travel_time


def _to_numpy_theta(theta) -> np.ndarray:
    if TORCH_AVAILABLE and isinstance(theta, torch.Tensor):
        return theta.detach().cpu().numpy()
    return np.asarray(theta, dtype=np.float64)


class _LazyPhiList:
    """
    Lightweight on-demand representation of the feature vectors `phis` for available edges
    at a given state (time_budget, edge_indices, num_edges).
    Avoids allocating dense 2*|E| arrays unless explicitly indexed by legacy callers.
    """
    def __init__(self, edge_indices: list, time_budget: float, num_edges: int):
        self.edge_indices = edge_indices
        self.time_budget = float(time_budget)
        self.num_edges = num_edges

    def __len__(self):
        return len(self.edge_indices)

    def __getitem__(self, idx: int) -> np.ndarray:
        vec = np.zeros(2 * self.num_edges, dtype=np.float64)
        e_idx = self.edge_indices[idx]
        vec[e_idx] = self.time_budget
        vec[self.num_edges + e_idx] = 1.0
        return vec


def run_pi_one(G, origin, destination, theta_np, time_limit, edge_to_idx, num_edges):
    current_node = origin
    Tau_one = []
    time_budget = float(time_limit)
    visited_steps = 0
    max_steps = max(200, G.number_of_nodes() * 4)

    theta_time = theta_np[:num_edges]
    theta_base = theta_np[num_edges:]

    while current_node != destination and visited_steps < max_steps:
        if time_budget < 0.0:
            break

        available_edges = []
        available_edge_indices = []
        for u, v, data in G.out_edges(current_node, data=True):
            if data['edge_type'] == 'edge1' or (data['edge_type'] == 'edge2' and random.random() < data['p']):
                available_edges.append(v)
                available_edge_indices.append(edge_to_idx[(u, v)])

        if not available_edges:
            break

        idx_arr = np.asarray(available_edge_indices, dtype=np.int64)
        logits = time_budget * theta_time[idx_arr] + theta_base[idx_arr]
        # Numerically stable softmax
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probabilities = exp_logits / np.sum(exp_logits)

        chosen_index = int(np.random.choice(len(available_edges), p=probabilities))
        next_node = available_edges[chosen_index]
        chosen_edge_idx = available_edge_indices[chosen_index]

        edge_data = G[current_node][next_node]
        travel_time = get_travel_time(edge_data['weight'], edge_data['sigma'])

        lazy_phis = _LazyPhiList(available_edge_indices, time_budget, num_edges)
        chosen_fai = lazy_phis  # Compact reference; fast gradient uses element[8] (available_edge_indices)

        # 0: [node, time_budget], 1: edge_data, 2: travel_time, 3: available_edges,
        # 4: phis, 5: chosen_fai, 6: probabilities, 7: chosen_index, 8: available_edge_indices
        Tau_one.append([
            [current_node, time_budget],
            edge_data,
            travel_time,
            available_edges,
            lazy_phis,
            chosen_fai,
            probabilities,
            chosen_index,
            available_edge_indices,
            chosen_edge_idx,
        ])

        time_budget -= travel_time
        current_node = next_node
        visited_steps += 1

    # Terminal state marker
    Tau_one.append([[current_node, time_budget], 0, 0, 0, 0, 0, 0, 0, [], -1])
    return Tau_one


def run_pi(G, origin, destination, theta, time_limit, trajectory_number, current_iteration, device=None):
    """
    Collects `trajectory_number` policy rollout trajectories.
    Supports both PyTorch CUDA/CPU tensors and NumPy arrays for `theta`.
    """
    if 'edge_to_idx' not in G.graph:
        all_edges = list(G.edges())
        G.graph['all_edges'] = all_edges
        G.graph['edge_to_idx'] = {e: idx for idx, e in enumerate(all_edges)}

    edge_to_idx = G.graph['edge_to_idx']
    num_edges = len(edge_to_idx)
    theta_np = _to_numpy_theta(theta)

    Tau = [
        run_pi_one(G, origin, destination, theta_np, time_limit, edge_to_idx, num_edges)
        for _ in range(trajectory_number)
    ]
    return Tau

