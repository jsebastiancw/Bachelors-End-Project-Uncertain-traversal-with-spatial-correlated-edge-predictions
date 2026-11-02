from imports import *


def _ensure_edge_index_map(G):
    edge_to_idx = G.graph.get('edge_to_idx')
    if edge_to_idx is None:
        all_edges = list(G.edges())
        G.graph['all_edges'] = all_edges
        edge_to_idx = {e: idx for idx, e in enumerate(all_edges)}
        G.graph['edge_to_idx'] = edge_to_idx
    return edge_to_idx


def travel_simulate(G, taus, theta, time_limit, origin, destination):
    """
    Off-policy importance-sampled trajectory replay for off_policy_VPG.
    Executes on CUDA tensor if `theta` is a GPU tensor, or CPU/NumPy otherwise.
    """
    edge_to_idx = _ensure_edge_index_map(G)
    num_edges = len(edge_to_idx)

    is_tensor = TORCH_AVAILABLE and isinstance(theta, torch.Tensor)
    theta_np = theta.detach().cpu().numpy() if is_tensor else np.asarray(theta, dtype=np.float64)
    theta_time = theta_np[:num_edges]
    theta_base = theta_np[num_edges:]

    target_time = np.zeros(num_edges, dtype=np.float64)
    target_base = np.zeros(num_edges, dtype=np.float64)
    acc_time = np.zeros(num_edges, dtype=np.float64)
    acc_base = np.zeros(num_edges, dtype=np.float64)

    for tau in taus:
        terminal = tau[-1][0]
        if terminal[1] >= 0.0 and terminal[0] == destination:
            path = [element[0][0] for element in tau]
            log_prob_mu = 0.0
            for element in tau[:-1]:
                p_mu = max(float(element[6][element[7]]), 1e-12)
                log_prob_mu += math.log(p_mu)

            log_prob_pi = 0.0
            time_budget = float(time_limit)

            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                if time_budget < 0.0:
                    break

                available_edges = [v for _, v in G.out_edges(current_node)]
                if not available_edges or next_node not in available_edges:
                    break

                avail_indices = np.asarray([edge_to_idx[(current_node, v)] for v in available_edges], dtype=np.int64)
                logits = time_budget * theta_time[avail_indices] + theta_base[avail_indices]
                logits -= np.max(logits)
                exp_logits = np.exp(logits)
                probabilities = exp_logits / np.sum(exp_logits)

                chosen_index = available_edges.index(next_node)
                chosen_edge_idx = avail_indices[chosen_index]
                log_prob_pi += math.log(max(float(probabilities[chosen_index]), 1e-12))

                acc_time[chosen_edge_idx] += time_budget
                acc_base[chosen_edge_idx] += 1.0
                acc_time[avail_indices] -= probabilities * time_budget
                acc_base[avail_indices] -= probabilities

                travel_time = G[current_node][next_node]['weight']
                time_budget -= travel_time

            bz = math.exp(max(-20.0, min(2.3, log_prob_pi - log_prob_mu)))
            target_time += bz * acc_time
            target_base += bz * acc_base

    target = np.concatenate([target_time, target_base])
    if is_tensor:
        return torch.as_tensor(target, dtype=theta.dtype, device=theta.device)
    return target


def travel_simulate_VAC(G, taus, theta, time_limit, origin, destination):
    """
    Off-policy importance-sampled trajectory replay for SEVAC.
    Executes on CUDA tensor if `theta` is a GPU tensor, or CPU/NumPy otherwise.
    """
    edge_to_idx = _ensure_edge_index_map(G)
    num_edges = len(edge_to_idx)

    is_tensor = TORCH_AVAILABLE and isinstance(theta, torch.Tensor)
    theta_np = theta.detach().cpu().numpy() if is_tensor else np.asarray(theta, dtype=np.float64)
    theta_time = theta_np[:num_edges]
    theta_base = theta_np[num_edges:]

    target_time = np.zeros(num_edges, dtype=np.float64)
    target_base = np.zeros(num_edges, dtype=np.float64)

    for tau in taus:
        path = [element[0][0] for element in tau]
        log_prob_mu = 0.0
        for element in tau[:-1]:
            p_mu = max(float(element[6][element[7]]), 1e-12)
            log_prob_mu += math.log(p_mu)

        log_prob_pi = 0.0
        time_budget = float(time_limit)
        step_time = np.zeros(num_edges, dtype=np.float64)
        step_base = np.zeros(num_edges, dtype=np.float64)

        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            if time_budget < 0.0:
                break

            available_edges = [v for _, v in G.out_edges(current_node)]
            if not available_edges or next_node not in available_edges:
                break

            avail_indices = np.asarray([edge_to_idx[(current_node, v)] for v in available_edges], dtype=np.int64)
            logits = time_budget * theta_time[avail_indices] + theta_base[avail_indices]
            logits -= np.max(logits)
            exp_logits = np.exp(logits)
            probabilities = exp_logits / np.sum(exp_logits)

            chosen_index = available_edges.index(next_node)
            chosen_edge_idx = avail_indices[chosen_index]
            log_prob_pi += math.log(max(float(probabilities[chosen_index]), 1e-12))

            # Match original travel_simulate_VAC where `tmp` holds the last step's score vector
            step_time.fill(0.0)
            step_base.fill(0.0)
            step_time[chosen_edge_idx] += time_budget
            step_base[chosen_edge_idx] += 1.0
            step_time[avail_indices] -= probabilities * time_budget
            step_base[avail_indices] -= probabilities

            travel_time = G[current_node][next_node]['weight']
            time_budget -= travel_time

        bz = math.exp(max(-20.0, min(2.3, log_prob_pi - log_prob_mu)))
        terminal = tau[-1][0]
        if terminal[1] >= 0.0 and terminal[0] == destination:
            target_time += 0.5 * bz * step_time
            target_base += 0.5 * bz * step_base
        else:
            target_time -= 0.5 * bz * step_time
            target_base -= 0.5 * bz * step_base

    target = np.concatenate([target_time, target_base])
    if is_tensor:
        return torch.as_tensor(target, dtype=theta.dtype, device=theta.device)
    return target
