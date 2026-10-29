"""
gaussian_disaster_simulator.py
Gaussian Process (GP) Spatial Disaster Hazard Field & Correlated Edge Blockage Generator
based on the Gaussian Traveler Problem (GTP) framework.

References:
- Dey, D., Shankar, K., & Bagnell, J. A. (2014). Gaussian Traveler Problem.
- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist


# ==========================================
# 1. Gaussian Process Kernel & Prior Sampling
# ==========================================

def rbf_kernel(X1, X2, length_scale=0.20, variance=1.5):
    """
    Computes the Squared Exponential (Radial Basis Function) covariance matrix K(X1, X2):
        k(x_i, x_j) = sigma_f^2 * exp( - ||x_i - x_j||^2 / (2 * l^2) )

    Args:
        X1 (np.ndarray): Array of shape (N, 2) with 2D spatial coordinates.
        X2 (np.ndarray): Array of shape (M, 2) with 2D spatial coordinates.
        length_scale (float): Spatial correlation length scale l > 0.
        variance (float): Signal variance sigma_f^2 > 0 controlling hazard amplitude.

    Returns:
        K (np.ndarray): Covariance matrix of shape (N, M).
    """
    X1 = np.atleast_2d(np.asarray(X1, dtype=float))
    X2 = np.atleast_2d(np.asarray(X2, dtype=float))
    sq_dists = cdist(X1, X2, metric='sqeuclidean')
    return variance * np.exp(-0.5 * sq_dists / (length_scale ** 2))


def sample_gp_prior(coordinates, length_scale=0.20, variance=1.5, jitter=1e-6, rng=None):
    """
    Samples a latent disaster intensity vector Y ~ N(0, K) from a zero-mean Gaussian
    Process prior over the given 2D spatial coordinates (Dey et al., 2014).

    Args:
        coordinates (np.ndarray): Array of shape (N, 2) representing spatial query points X.
        length_scale (float): RBF kernel length scale l controlling smoothness.
        variance (float): RBF kernel variance sigma_f^2.
        jitter (float): Small diagonal nugget added to K for positive-definite stability.
        rng (np.random.Generator or None): Optional NumPy random generator for reproducibility.

    Returns:
        Y (np.ndarray): Sampled latent hazard intensities of shape (N,).
        K (np.ndarray): Covariance matrix of shape (N, N).
    """
    coords = np.atleast_2d(np.asarray(coordinates, dtype=float))
    n_points = coords.shape[0]

    K = rbf_kernel(coords, coords, length_scale=length_scale, variance=variance)
    K_stable = K + jitter * np.eye(n_points)

    mean = np.zeros(n_points, dtype=float)
    if rng is None:
        Y = np.random.multivariate_normal(mean, K_stable)
    else:
        Y = rng.multivariate_normal(mean, K_stable)

    return Y, K


# ==========================================
# 2. Invertible Cost / Probability Mapping
# ==========================================

def logistic_cost_mapping(y, k_steepness=1.2, y_offset=0.5, max_blockage_prob=None):
    """
    Invertible logistic (sigmoid) mapping C: R -> (0, 1) transforming unbounded
    latent GP dynamics y_e in R into valid edge blockage probabilities q_e in [0, 1]:
        q_e = C(y_e) = 1 / (1 + exp(-k_steepness * (y_e - y_offset)))

    Its analytical inverse is the logit function:
        C^{-1}(q_e) = y_offset + (1 / k_steepness) * ln(q_e / (1 - q_e))

    Args:
        y (np.ndarray or float): Unbounded latent GP variables y_e.
        k_steepness (float): Logistic slope parameter (> 0).
        y_offset (float): Threshold offset shifting the baseline blockage rate.
        max_blockage_prob (float or None): Optional upper clip bound (e.g., 0.92).

    Returns:
        q (np.ndarray or float): Mapped blockage probabilities in [0, 1].
    """
    y_arr = np.asarray(y, dtype=float)
    q = 1.0 / (1.0 + np.exp(-k_steepness * (y_arr - y_offset)))
    if max_blockage_prob is not None:
        q = np.clip(q, 0.0, max_blockage_prob)
    return float(q) if np.ndim(y) == 0 else q


def inverse_logistic_cost_mapping(q, k_steepness=1.2, y_offset=0.5):
    """
    Exact inverse mapping C^{-1}: (0, 1) -> R (logit function) corresponding to
    logistic_cost_mapping, satisfying the GTP invertible cost requirement (Dey et al., 2014).
    """
    q_clipped = np.clip(np.asarray(q, dtype=float), 1e-9, 1.0 - 1e-9)
    return y_offset + (1.0 / k_steepness) * np.log(q_clipped / (1.0 - q_clipped))


# ==========================================
# 3. Graph Edge Integration & Realization
# ==========================================

def extract_edge_midpoints(G):
    """
    Extracts the ordered list of edges and their 2D spatial midpoints
    m_e = ((x_u + x_v) / 2, (y_u + y_v) / 2) for all e = (u, v) in G.

    Returns:
        edge_list (list of tuple): Ordered list of edges (u, v).
        midpoints (np.ndarray): Array of shape (|E|, 2) with midpoint coordinates.
    """
    pos = nx.get_node_attributes(G, 'pos')
    edge_list = list(G.edges())
    midpoints = np.zeros((len(edge_list), 2), dtype=float)

    for idx, (u, v) in enumerate(edge_list):
        x_u, y_u = pos[u]
        x_v, y_v = pos[v]
        midpoints[idx, 0] = 0.5 * (x_u + x_v)
        midpoints[idx, 1] = 0.5 * (y_u + y_v)

    return edge_list, midpoints


def apply_gp_disaster_to_graph(G, gp_params=None):
    """
    Samples a Gaussian Process prior over the spatial midpoints of all edges in G,
    maps the latent intensities y_e to blockage probabilities q_e via the invertible
    logistic function, and attaches attributes to G and gp_params for downstream
    conditioned visualization and inference.

    Attached edge attributes:
        - 'midpoint': (mx, my) tuple
        - 'latent_intensity': float y_e
        - 'blockage_prob': float q_e in [0, 1]
        - 'noisy_observation': float t_e ~ N(y_e, sigma_obs^2) (Dey et al., 2014)

    Args:
        G (nx.Graph): Spatial graph generated by SpatialGraphGenerator.
        gp_params (dict or None): Dictionary of GP and cost mapping hyperparameters.

    Returns:
        G (nx.Graph): Graph updated with GP latent intensities and blockage probabilities.
        gp_state (dict): Dictionary containing hyperparameters and sampled GP state
                         (midpoints, latent y_e, covariance K, etc.) for visualization.
    """
    if gp_params is None:
        gp_params = {}

    length_scale = gp_params.get('length_scale', 0.22)
    variance = gp_params.get('variance', 1.5)
    noise_variance = gp_params.get('noise_variance', 0.05)  # sigma^2 in p(t|y)
    k_steepness = gp_params.get('k_steepness', 1.2)
    y_offset = gp_params.get('y_offset', 0.5)
    max_blockage_prob = gp_params.get('max_blockage_prob', 0.92)
    jitter = gp_params.get('jitter', 1e-6)
    seed = gp_params.get('seed', None)

    rng = np.random.default_rng(seed) if seed is not None else None

    # 1. Extract edge midpoints X_E
    edge_list, midpoints = extract_edge_midpoints(G)
    if len(edge_list) == 0:
        raise ValueError("Graph G contains no edges.")

    # 2. Sample latent hazard intensities Y_E ~ N(0, K)
    y_edges, K_edges = sample_gp_prior(
        midpoints,
        length_scale=length_scale,
        variance=variance,
        jitter=jitter,
        rng=rng
    )

    # 3. Sample noisy local observations t_e ~ N(y_e, sigma^2) per GTP likelihood model
    if rng is not None:
        obs_noise = rng.normal(0.0, np.sqrt(noise_variance), size=len(edge_list))
    else:
        obs_noise = np.random.normal(0.0, np.sqrt(noise_variance), size=len(edge_list))
    t_edges = y_edges + obs_noise

    # 4. Map latent variables y_e -> blockage probabilities q_e via invertible logistic C
    q_edges = logistic_cost_mapping(
        y_edges,
        k_steepness=k_steepness,
        y_offset=y_offset,
        max_blockage_prob=max_blockage_prob
    )

    # 5. Attach attributes to graph edges
    for idx, (u, v) in enumerate(edge_list):
        G.edges[u, v]['midpoint'] = (float(midpoints[idx, 0]), float(midpoints[idx, 1]))
        G.edges[u, v]['latent_intensity'] = float(y_edges[idx])
        G.edges[u, v]['noisy_observation'] = float(t_edges[idx])
        G.edges[u, v]['blockage_prob'] = float(q_edges[idx])

    # Store full realization state inside gp_state so visualize_gaussian_disaster
    # renders the exact same GP realization across the continuous 2D meshgrid
    gp_state = dict(gp_params)
    gp_state.update({
        'length_scale': length_scale,
        'variance': variance,
        'noise_variance': noise_variance,
        'k_steepness': k_steepness,
        'y_offset': y_offset,
        'max_blockage_prob': max_blockage_prob,
        'jitter': jitter,
        'edge_midpoints': midpoints,
        'edge_latent_y': y_edges,
        'edge_noisy_t': t_edges,
        'edge_blockage_q': q_edges,
        'K_edges': K_edges,
    })

    # Also store on G.graph for convenience if user only passes G or simple gp_params
    G.graph['gp_state'] = gp_state

    return G, gp_state


def sample_gp_disaster_realization(G, max_resamples=1000, allow_relaxation=True):
    """
    Draws Bernoulli trials B_e ~ Bern(q_e) for every edge in G based on the GP-derived
    blockage probabilities q_e. Verifies that an Origin -> Destination path exists
    in G_offline, using multi-stage relaxation and a safest-path fallback if needed.

    Returns:
        G_offline (nx.Graph): Graph with blocked edges removed.
        blocked_edges (list): List of edges (u, v) that failed in this realization.
    """
    nodes = list(G.nodes())
    origins = [n for n, d in G.nodes(data=True) if d.get('is_origin', False)]
    destinations = [n for n, d in G.nodes(data=True) if d.get('is_destination', False)]

    origin = origins[0] if origins else min(nodes)
    destination = destinations[0] if destinations else max(nodes)

    scale = 1.0
    max_relax_stages = 6 if allow_relaxation else 1
    attempts_per_stage = max(1, max_resamples // max_relax_stages) if allow_relaxation else max_resamples

    for _ in range(max_relax_stages):
        for _ in range(attempts_per_stage):
            G_offline = G.copy()
            blocked_edges = []

            for u, v, data in G.edges(data=True):
                q_e = data.get('blockage_prob', 0.0) * scale
                is_blocked = (np.random.rand() < q_e)
                G.edges[u, v]['is_blocked'] = int(is_blocked)

                if is_blocked:
                    blocked_edges.append((u, v))
                    G_offline.remove_edge(u, v)

            if nx.has_path(G_offline, origin, destination):
                return G_offline, blocked_edges

        scale *= 0.80

    if allow_relaxation and nx.has_path(G, origin, destination):
        def edge_risk_weight(u, v, d):
            q = min(0.999, d.get('blockage_prob', 0.0))
            return -np.log(1.0 - q)

        safest_path = nx.shortest_path(G, origin, destination, weight=edge_risk_weight)
        protected_edges = {
            (min(safest_path[k], safest_path[k + 1]), max(safest_path[k], safest_path[k + 1]))
            for k in range(len(safest_path) - 1)
        }

        G_offline = G.copy()
        blocked_edges = []
        for u, v, data in G.edges(data=True):
            edge_key = (min(u, v), max(u, v))
            if edge_key in protected_edges:
                is_blocked = False
            else:
                q_e = data.get('blockage_prob', 0.0)
                is_blocked = (np.random.rand() < q_e)

            G.edges[u, v]['is_blocked'] = int(is_blocked)
            if is_blocked:
                blocked_edges.append((u, v))
                G_offline.remove_edge(u, v)

        return G_offline, blocked_edges

    raise RuntimeError(
        f"Failed to sample an O->D connected G_offline after {max_resamples} attempts."
    )


# ==========================================
# 4. Continuous Meshgrid GP Evaluation & Visualization
# ==========================================

def evaluate_gp_on_meshgrid(grid_coords, gp_state):
    """
    Evaluates the continuous 2D hazard field over meshgrid coordinates X_grid
    consistent with the exact GP realization on the graph edges.
    Uses GP conditional projection (kriging) from the sampled edge midpoints
    so the background contour seamlessly matches the edge realizations.
    """
    length_scale = gp_state.get('length_scale', 0.22)
    variance = gp_state.get('variance', 1.5)
    jitter = gp_state.get('jitter', 1e-6)

    X_edges = gp_state.get('edge_midpoints')
    y_edges = gp_state.get('edge_latent_y')

    if X_edges is None or y_edges is None:
        # If no edge realization exists yet, sample directly from the prior
        y_grid, _ = sample_gp_prior(grid_coords, length_scale=length_scale, variance=variance, jitter=jitter)
        return y_grid

    K_ee = gp_state.get('K_edges')
    if K_ee is None:
        K_ee = rbf_kernel(X_edges, X_edges, length_scale=length_scale, variance=variance)
    K_ee_reg = K_ee + jitter * np.eye(len(X_edges))

    # Cross-covariance between meshgrid points and edge midpoints: K(X_grid, X_edges)
    K_ge = rbf_kernel(grid_coords, X_edges, length_scale=length_scale, variance=variance)

    # Posterior mean E[Y_grid | Y_edges] = K_ge @ K_ee^{-1} @ y_edges
    alpha = np.linalg.solve(K_ee_reg, y_edges)
    y_grid = K_ge @ alpha
    return y_grid


def visualize_gaussian_disaster(G, gp_params=None, show_labels=True, grid_res=80):
    """
    Renders the spatial network overlaid on the continuous 2D Gaussian Process
    hazard field heatmap:
    - Background: Continuous GP-derived blockage probability field C(Y(x)) over [0, 1]^2
      rendered via matplotlib.pyplot.contourf with the 'RdYlGn_r' colormap.
    - Nodes: Origin (O) as a Star ('*'), Destination (D) as an 'X', and intermediate nodes.
    - Edges: Color-coded using a 'YlOrRd' gradient based on their blockage probability q_e,
      with numerical q_e text labels annotated alongside each edge.
    """
    # Ensure GP realization has been applied to G
    gp_state = G.graph.get('gp_state', {})
    if gp_params is not None:
        merged_params = dict(gp_state)
        merged_params.update(gp_params)
        gp_state = merged_params

    # If G does not yet have 'blockage_prob' on edges, compute it now
    if len(G.edges()) > 0 and 'blockage_prob' not in next(iter(G.edges(data=True)))[2]:
        G, gp_state = apply_gp_disaster_to_graph(G, gp_state)

    k_steepness = gp_state.get('k_steepness', 1.2)
    y_offset = gp_state.get('y_offset', 0.5)
    max_blockage_prob = gp_state.get('max_blockage_prob', 0.92)
    length_scale = gp_state.get('length_scale', 0.22)
    variance = gp_state.get('variance', 1.5)

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_title(
        f"Gaussian Process Disaster Hazard Field & Road Network ($l={length_scale:.2f}, \\sigma_f^2={variance:.2f}$)",
        fontsize=14,
        fontweight='bold'
    )

    # 1. Create 2D Meshgrid over [0, 1]^2 and evaluate GP realization
    gx = np.linspace(0.0, 1.0, grid_res)
    gy = np.linspace(0.0, 1.0, grid_res)
    X, Y = np.meshgrid(gx, gy)
    grid_coords = np.column_stack([X.ravel(), Y.ravel()])

    y_grid = evaluate_gp_on_meshgrid(grid_coords, gp_state)
    q_grid = logistic_cost_mapping(
        y_grid,
        k_steepness=k_steepness,
        y_offset=y_offset,
        max_blockage_prob=max_blockage_prob
    ).reshape(grid_res, grid_res)

    # Render continuous background contour plot using 'RdYlGn_r'
    bg_cmap = mpl.colormaps['RdYlGn_r']
    contour = ax.contourf(
        X, Y, q_grid,
        levels=50,
        cmap=bg_cmap,
        alpha=0.58,
        vmin=0.0,
        vmax=1.0
    )
    cbar_bg = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.02)
    cbar_bg.set_label("Continuous GP Hazard Field $C(Y(\\mathbf{x}))$", fontsize=10)

    # 2. Extract Node Positions & Categories
    pos = nx.get_node_attributes(G, 'pos')
    origins = [n for n, attr in G.nodes(data=True) if attr.get('is_origin')]
    destinations = [n for n, attr in G.nodes(data=True) if attr.get('is_destination')]
    standard_nodes = [n for n in G.nodes() if n not in origins and n not in destinations]

    # 3. Draw Edges Color-Coded by Blockage Probability q_e ('YlOrRd')
    edge_cmap = mpl.colormaps['YlOrRd']
    edge_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    segments = []
    q_values = []
    linewidths = []

    for u, v, attr in G.edges(data=True):
        segments.append([pos[u], pos[v]])
        q_e = attr.get('blockage_prob', 0.0)
        q_values.append(q_e)
        linewidths.append(2.8 if attr.get('is_highway', 0) == 1 else 1.8)

    lc = LineCollection(
        segments,
        cmap=edge_cmap,
        norm=edge_norm,
        linewidths=linewidths,
        alpha=0.95,
        zorder=3
    )
    lc.set_array(np.array(q_values))
    ax.add_collection(lc)

    cbar_edge = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.08)
    cbar_edge.set_label("Edge Blockage Probability $q_e$", fontsize=10)

    # 4. Annotate Edge Blockage Probabilities (q_e)
    if show_labels:
        edge_labels = {
            (u, v): f"{attr.get('blockage_prob', 0.0):.2f}"
            for u, v, attr in G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=7,
            font_color='black',
            font_weight='bold',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.0),
            ax=ax
        )

    # 5. Overlay Nodes (Origin as Star, Destination as X)
    nx.draw_networkx_nodes(
        G, pos, nodelist=standard_nodes, node_size=140,
        node_color='white', edgecolors='black', linewidths=1.2, ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=origins, node_size=380,
        node_color='dodgerblue', node_shape='*', edgecolors='black', linewidths=1.5, ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=destinations, node_size=320,
        node_color='crimson', node_shape='X', edgecolors='black', linewidths=1.5, ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax)

    # 6. Plot Formatting & Legend
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.35)

    legend_handles = [
        Line2D([0], [0], marker='*', color='w', label='Origin (O)',
               markerfacecolor='dodgerblue', markersize=14, markeredgecolor='k'),
        Line2D([0], [0], marker='X', color='w', label='Destination (D)',
               markerfacecolor='crimson', markersize=12, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Intermediate Node',
               markerfacecolor='white', markersize=8, markeredgecolor='k'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.show()


# ==========================================
# 5. Example Execution Script
# ==========================================
if __name__ == "__main__":
    from graph_builder import SpatialGraphGenerator

    # 1. Build a synthetic spatial network
    builder = SpatialGraphGenerator(n_nodes=40, d_min=0.04)
    G, _ = builder.build_graph(regime='multidistrict', C=4)

    # 2. Define Gaussian Process (GTP) hyperparameters (Dey et al., 2014)
    gp_params = {
        'length_scale': 0.22,       # Spatial correlation length scale l
        'variance': 1.8,            # Prior signal variance sigma_f^2
        'noise_variance': 0.05,     # Observation noise variance sigma^2 in p(t|y)
        'k_steepness': 1.2,         # Logistic cost steepness
        'y_offset': 0.4,            # Logistic inflection threshold
        'max_blockage_prob': 0.92,  # Upper bound on edge blockage probability
    }

    # 3. Sample GP prior over edge midpoints and map to blockage probabilities q_e
    G, gp_state = apply_gp_disaster_to_graph(G, gp_params)

    # 4. Sample an O -> D connected disaster realization
    G_offline, blocked_edges = sample_gp_disaster_realization(G)
    print(
        f"Total Edges: {G.number_of_edges()} | "
        f"Blocked in GP Realization: {len(blocked_edges)} | "
        f"Remaining in G_offline: {G_offline.number_of_edges()}"
    )

    # 5. Render the GP Hazard Field, Network Topology, and Edge Blockage Probabilities q_e
    visualize_gaussian_disaster(G, gp_state, show_labels=True)

