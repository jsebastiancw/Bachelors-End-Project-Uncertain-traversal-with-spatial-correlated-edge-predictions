import json
from scipy.stats import norm
from imports import *
from load_data import load_data, resolve_data_path
from network_operations import create_network, get_travel_time
from simulated_annealing import simulated_annealing
from theta_operations import update_theta
from run_pi import run_pi
from initialize_theta import initialize_theta
from travel_simulate import travel_simulate, travel_simulate_VAC


def _zeros(size: int, device=None):
    target_device = device if device is not None else DEVICE
    if TORCH_AVAILABLE and target_device is not None:
        return torch.zeros(size, dtype=torch.float64, device=target_device)
    return np.zeros(size, dtype=np.float64)


def _compute_tau_score(tau: list, num_edges: int) -> np.ndarray:
    """
    Computes the cumulative policy score sum_{t} (phi(s_t, a_t) - E_{a~pi}[phi(s_t, a)])
    for a single trajectory `tau` in O(steps * out_degree) time.
    """
    grad_time = np.zeros(num_edges, dtype=np.float64)
    grad_base = np.zeros(num_edges, dtype=np.float64)

    for state in tau[:-1]:
        if len(state) >= 10:
            time_budget = float(state[0][1])
            probabilities = state[6]
            avail_indices = state[8]
            chosen_edge_idx = state[9]

            grad_time[chosen_edge_idx] += time_budget
            grad_base[chosen_edge_idx] += 1.0
            grad_time[avail_indices] -= probabilities * time_budget
            grad_base[avail_indices] -= probabilities
        else:
            # Legacy fallback
            tmp = np.array(state[5], dtype=np.float64, copy=True)
            for idx in range(len(state[3])):
                tmp -= state[6][idx] * state[4][idx]
            grad_time += tmp[:num_edges]
            grad_base += tmp[num_edges:]

    return np.concatenate([grad_time, grad_base])


def _to_device_grad(grad_np: np.ndarray, like_theta):
    if TORCH_AVAILABLE and isinstance(like_theta, torch.Tensor):
        return torch.as_tensor(grad_np, dtype=like_theta.dtype, device=like_theta.device)
    return grad_np


def VPG(G, iterations, alpha, start, end, time_budget, shortest_path, device=None):
    time_limit = time_budget if time_budget != 0 else 20.0
    M = 100
    num_edges = len(G.edges())
    thetas = _zeros(2 * num_edges, device=device)

    success_rates = []
    success_count = 0

    for episode in range(iterations):
        if success_count == 0:
            thetas = initialize_theta(G, thetas, shortest_path)

        taus = run_pi(G, start, end, thetas, time_limit, M, episode, device=device)
        grad_np = np.zeros(2 * num_edges, dtype=np.float64)

        for tau in taus:
            terminal = tau[-1][0]
            if terminal[1] >= 0.0:
                if terminal[0] == end:
                    success_count += 1
                grad_np += _compute_tau_score(tau, num_edges)

        gradient = _to_device_grad(grad_np / M, thetas)
        thetas = thetas + alpha * gradient

        ep_num = episode + 1
        success_rate = success_count / (M * ep_num)
        print(f"[VPG] Episode: {ep_num}, Success rate: {success_rate:.2f}")
        success_rates.append(success_rate)

    return success_rates


def VAC(G, iterations, alpha, start, end, time_budget, shortest_path, device=None):
    time_limit = time_budget if time_budget != 0 else 15.0
    M = 100
    num_edges = len(G.edges())
    thetas = _zeros(2 * num_edges, device=device)

    success_rates = []
    success_count = 0

    for episode in range(iterations):
        if success_count == 0:
            thetas = initialize_theta(G, thetas, shortest_path)

        taus = run_pi(G, start, end, thetas, time_limit, M, episode, device=device)
        grad_np = np.zeros(2 * num_edges, dtype=np.float64)

        for tau in taus:
            terminal = tau[-1][0]
            tau_score = _compute_tau_score(tau, num_edges)
            if terminal[1] >= 0.0 and terminal[0] == end:
                success_count += 1
                grad_np += 0.5 * tau_score
            else:
                grad_np -= 0.5 * tau_score

        gradient = _to_device_grad(grad_np / M, thetas)
        thetas = thetas + 2.0 * alpha * gradient

        ep_num = episode + 1
        success_rate = success_count / (M * ep_num)
        print(f"[VAC] Episode: {ep_num}, Success rate: {success_rate:.2f}")
        success_rates.append(success_rate)

    return success_rates


def off_policy_VPG(G, iterations, alpha, start, end, time_budget, shortest_path, device=None):
    time_limit = time_budget if time_budget != 0 else 15.0
    M = 100
    num_edges = len(G.edges())
    thetas = _zeros(2 * num_edges, device=device)

    success_rates = []
    success_count = 0
    taus_last = None

    for episode in range(iterations):
        if success_count == 0:
            thetas = initialize_theta(G, thetas, shortest_path)

        taus = run_pi(G, start, end, thetas, time_limit, M, episode, device=device)
        grad_np = np.zeros(2 * num_edges, dtype=np.float64)

        for tau in taus:
            terminal = tau[-1][0]
            if terminal[1] >= 0.0 and terminal[0] == end:
                success_count += 1
                grad_np += _compute_tau_score(tau, num_edges)

        gradient = _to_device_grad(grad_np, thetas)
        if episode >= 10 and taus_last is not None:
            gradient = gradient + travel_simulate(G, taus_last, thetas, time_limit, start, end)

        taus_last = taus
        gradient = gradient / M
        thetas = thetas + alpha * gradient

        ep_num = episode + 1
        success_rate = success_count / (M * ep_num)
        print(f"[off-policy VPG] Episode: {ep_num}, Success rate: {success_rate:.2f}")
        success_rates.append(success_rate)

    return success_rates


def SEVAC(G, iterations, alpha, start, end, time_budget, shortest_path, device=None):
    time_limit = time_budget if time_budget != 0 else 15.0
    M = 100
    num_edges = len(G.edges())
    thetas = _zeros(2 * num_edges, device=device)

    success_rates = []
    success_count = 0
    taus_last = None

    for episode in range(iterations):
        if success_count == 0:
            thetas = initialize_theta(G, thetas, shortest_path)

        taus = run_pi(G, start, end, thetas, time_limit, M, episode, device=device)
        grad_np = np.zeros(2 * num_edges, dtype=np.float64)

        for tau in taus:
            terminal = tau[-1][0]
            tau_score = _compute_tau_score(tau, num_edges)
            if terminal[1] >= 0.0 and terminal[0] == end:
                success_count += 1
                grad_np += 0.5 * tau_score
            else:
                grad_np -= 0.5 * tau_score

        gradient = _to_device_grad(grad_np, thetas)
        thetas = thetas + 2.0 * alpha * gradient

        if episode >= 10 and taus_last is not None:
            gradient = gradient + travel_simulate_VAC(G, taus_last, thetas, time_limit, start, end)

        taus_last = taus
        gradient = gradient / M
        thetas = thetas + 2.0 * alpha * gradient

        ep_num = episode + 1
        success_rate = success_count / (M * ep_num)
        print(f"[SEVAC] Episode: {ep_num}, Success rate: {success_rate:.2f}")
        success_rates.append(success_rate)

    return success_rates


def compare(network, iteration, alpha, device=None):
    target_device = device if device is not None else DEVICE
    file_path = resolve_data_path(f'Networks/{network}/{network}_network.csv')
    od_path = resolve_data_path(f'Networks/{network}/{network}_OD.csv')

    data = load_data(file_path)
    G = create_network(data)
    ods = pd.read_csv(od_path)

    probs_VPG = []
    probs_VAC = []
    probs_off_policy_VPG = []
    probs_SEVAC = []
    output_VPG = []
    output_VAC = []
    output_off_policy_VPG = []
    output_SEVAC = []

    p1 = p2 = p3 = p4 = p5 = p6 = p7 = p8 = p9 = p10 = p11 = p12 = 0.0
    times = 0

    for _, od in ods.iterrows():
        flag = True
        o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12 = (
            pd.Series(dtype=object) for _ in range(12)
        )
        start = int(od['O'])
        end = int(od['D'])
        try:
            shortest_path = nx.shortest_path(G, start, end, weight='weight')
            shortest_time = sum(G[u][v]['weight'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
            shortest_path_sigma = sum(G[u][v]['sigma'] for u, v in zip(shortest_path[:-1], shortest_path[1:]))
            time_k = 1.0
            time_budget = time_k * shortest_time

            print(f"\n--- Evaluating OD Pair ({start} -> {end}) | LET: {shortest_time:.2f} ---")
            success_rates1 = VPG(G, iteration, alpha, start, end, time_budget, shortest_path, device=target_device)
            success_rates2 = VAC(G, iteration, alpha, start, end, time_budget, shortest_path, device=target_device)
            success_rates3 = off_policy_VPG(G, iteration, alpha, start, end, time_budget, shortest_path, device=target_device)
            success_rates4 = SEVAC(G, iteration, alpha, start, end, time_budget, shortest_path, device=target_device)

            success_rates5 = VPG(G, iteration, alpha, start, end, time_budget * 0.95, shortest_path, device=target_device)
            success_rates6 = VAC(G, iteration, alpha, start, end, time_budget * 0.95, shortest_path, device=target_device)
            success_rates7 = off_policy_VPG(G, iteration, alpha, start, end, time_budget * 0.95, shortest_path, device=target_device)
            success_rates8 = SEVAC(G, iteration, alpha, start, end, time_budget * 0.95, shortest_path, device=target_device)

            success_rates9 = VPG(G, iteration, alpha, start, end, time_budget * 1.05, shortest_path, device=target_device)
            success_rates10 = VAC(G, iteration, alpha, start, end, time_budget * 1.05, shortest_path, device=target_device)
            success_rates11 = off_policy_VPG(G, iteration, alpha, start, end, time_budget * 1.05, shortest_path, device=target_device)
            success_rates12 = SEVAC(G, iteration, alpha, start, end, time_budget * 1.05, shortest_path, device=target_device)
        except Exception as exc:
            print(f"[compare] Skipping OD ({start} -> {end}): {exc}")
            flag = False

        if flag:
            times += 1
            o1['od'] = o2['od'] = o3['od'] = o4['od'] = o5['od'] = o6['od'] = o7['od'] = o8['od'] = o9['od'] = o10['od'] = o11['od'] = o12['od'] = [start, end]
            o1['tf'] = o2['tf'] = o3['tf'] = o4['tf'] = 1.0
            o5['tf'] = o6['tf'] = o7['tf'] = o8['tf'] = 0.95
            o9['tf'] = o10['tf'] = o11['tf'] = o12['tf'] = 1.05
            o1['let'] = o2['let'] = o3['let'] = o4['let'] = o5['let'] = o6['let'] = o7['let'] = o8['let'] = o9['let'] = o10['let'] = o11['let'] = o12['let'] = shortest_time
            o1['prob'] = o2['prob'] = o3['prob'] = o4['prob'] = norm.cdf(shortest_time, shortest_time, max(shortest_path_sigma, 1e-6))
            o5['prob'] = o6['prob'] = o7['prob'] = o8['prob'] = norm.cdf(shortest_time * 0.95, shortest_time, max(shortest_path_sigma, 1e-6))
            o9['prob'] = o10['prob'] = o11['prob'] = o12['prob'] = norm.cdf(shortest_time * 1.05, shortest_time, max(shortest_path_sigma, 1e-6))

            o1['sota'] = success_rates1[-1]
            o2['sota'] = success_rates2[-1]
            o3['sota'] = success_rates3[-1]
            o4['sota'] = success_rates4[-1]
            o5['sota'] = success_rates5[-1]
            o6['sota'] = success_rates6[-1]
            o7['sota'] = success_rates7[-1]
            o8['sota'] = success_rates8[-1]
            o9['sota'] = success_rates9[-1]
            o10['sota'] = success_rates10[-1]
            o11['sota'] = success_rates11[-1]
            o12['sota'] = success_rates12[-1]

            output_VPG.extend([o1, o5, o9])
            output_VAC.extend([o2, o6, o10])
            output_off_policy_VPG.extend([o3, o7, o11])
            output_SEVAC.extend([o4, o8, o12])

            p1 += success_rates1[-1]
            p2 += success_rates2[-1]
            p3 += success_rates3[-1]
            p4 += success_rates4[-1]
            p5 += success_rates5[-1]
            p6 += success_rates6[-1]
            p7 += success_rates7[-1]
            p8 += success_rates8[-1]
            p9 += success_rates9[-1]
            p10 += success_rates10[-1]
            p11 += success_rates11[-1]
            p12 += success_rates12[-1]

    denom = max(1, times)
    p1, p2, p3, p4 = p1 / denom, p2 / denom, p3 / denom, p4 / denom
    p5, p6, p7, p8 = p5 / denom, p6 / denom, p7 / denom, p8 / denom
    p9, p10, p11, p12 = p9 / denom, p10 / denom, p11 / denom, p12 / denom

    pp1, pp2, pp3, pp4, pp5, pp6, pp7, pp8, pp9, pp10, pp11, pp12 = (
        pd.Series(dtype=object) for _ in range(12)
    )
    pp1['tf'] = pp2['tf'] = pp3['tf'] = pp4['tf'] = 1.0
    pp5['tf'] = pp6['tf'] = pp7['tf'] = pp8['tf'] = 0.95
    pp9['tf'] = pp10['tf'] = pp11['tf'] = pp12['tf'] = 1.05

    pp1['prob'], pp2['prob'], pp3['prob'], pp4['prob'] = p1, p2, p3, p4
    pp5['prob'], pp6['prob'], pp7['prob'], pp8['prob'] = p5, p6, p7, p8
    pp9['prob'], pp10['prob'], pp11['prob'], pp12['prob'] = p9, p10, p11, p12

    probs_VPG.extend([pp1, pp5, pp9])
    probs_VAC.extend([pp2, pp6, pp10])
    probs_off_policy_VPG.extend([pp3, pp7, pp11])
    probs_SEVAC.extend([pp4, pp8, pp12])

    output1 = pd.DataFrame(output_VPG)
    output2 = pd.DataFrame(output_VAC)
    output3 = pd.DataFrame(output_off_policy_VPG)
    output4 = pd.DataFrame(output_SEVAC)

    probs1 = pd.DataFrame(probs_VPG)
    probs2 = pd.DataFrame(probs_VAC)
    probs3 = pd.DataFrame(probs_off_policy_VPG)
    probs4 = pd.DataFrame(probs_SEVAC)

    out_dir = os.path.join('Networks', network, 'csv')
    os.makedirs(out_dir, exist_ok=True)

    output1.to_csv(os.path.join(out_dir, f'{network}_VPG.csv'), index=False)
    output2.to_csv(os.path.join(out_dir, f'{network}_VAC.csv'), index=False)
    output3.to_csv(os.path.join(out_dir, f'{network}_off_policy_VPG.csv'), index=False)
    output4.to_csv(os.path.join(out_dir, f'{network}_SEVAC.csv'), index=False)

    probs1.to_csv(os.path.join(out_dir, f'{network}_VPG_mean_prob.csv'), index=False)
    probs2.to_csv(os.path.join(out_dir, f'{network}_VAC_mean_prob.csv'), index=False)
    probs3.to_csv(os.path.join(out_dir, f'{network}_off-policy_VPG_mean_prob.csv'), index=False)
    probs4.to_csv(os.path.join(out_dir, f'{network}_SEVAC_mean_prob.csv'), index=False)
    print(f"[compare] Saved results to {os.path.abspath(out_dir)}")
