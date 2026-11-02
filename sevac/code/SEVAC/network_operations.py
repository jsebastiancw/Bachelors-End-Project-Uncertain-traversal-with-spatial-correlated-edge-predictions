from imports import *


def _extract_sigma(edge_row, cost: float, rng: random.Random) -> float:
    """
    Extracts or synthesizes a positive standard deviation `sigma` for edge travel time.
    Handles CSVs that provide 'sigma', 'Var', 'variance', or omit variance columns.
    """
    for col in ('sigma', 'Sigma'):
        if col in edge_row and pd.notna(edge_row[col]) and float(edge_row[col]) > 0:
            return float(edge_row[col])

    for var_col in ('Var', 'var', 'variance', 'Variance'):
        if var_col in edge_row and pd.notna(edge_row[var_col]) and float(edge_row[var_col]) > 0:
            return float(math.sqrt(float(edge_row[var_col])))

    # Fallback matching generate_net.py when variance column is blank/omitted
    return max(1e-4, rng.uniform(0.05, 0.4) * cost)


def create_network(edges_data: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    rng = random.Random(616)

    prob_col = 'prob' if 'prob' in edges_data.columns else ('P' if 'P' in edges_data.columns else None)

    for _, edge in edges_data.iterrows():
        u = int(edge['From'])
        v = int(edge['To'])
        cost = float(edge['Cost'])
        prob = float(edge[prob_col]) if (prob_col and pd.notna(edge[prob_col])) else 1.0
        sigma = _extract_sigma(edge, cost, rng)
        edge_type = 'edge1' if prob >= 1.0 else 'edge2'

        G.add_edge(u, v, weight=cost, sigma=sigma, p=prob, edge_type=edge_type)

    # Pre-cache edge index table for O(1) lookup during trajectory rollouts and GPU updates
    all_edges = list(G.edges())
    G.graph['all_edges'] = all_edges
    G.graph['edge_to_idx'] = {e: idx for idx, e in enumerate(all_edges)}
    return G


def get_travel_time(mu: float, sigma: float) -> float:
    """
    Samples a positive travel time from N(mu, sigma^2) truncated to (0, +inf).
    """
    if sigma <= 0.0:
        return float(mu)
    # Fast truncated normal draw on (0, +inf)
    for _ in range(8):
        sample = random.gauss(mu, sigma)
        if sample > 0.0:
            return sample
    lower, upper = 0.0, np.inf
    X = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return float(X.rvs())
