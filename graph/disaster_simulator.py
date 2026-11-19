"""
disaster_simulator.py
Spatial Disaster Hazard Field & Correlated Edge Blockage Generator for CTP.

References:
- Silverman, B. W. (1986). Density Estimation for Statistics and Data Analysis.
- Wang, Y. H. (1993). On the number of successes in independent trials. Statistica Sinica.
- den Hollander, J. (2025). Construction of CTP Instances with Spatially Correlated Blockages.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


# ==========================================
# 1. Mathematical Kernels & Transformations
# ==========================================

def inverse_ellipsoid_distance(x, y, x0, y0, a, b, phi_deg):
    """
    Applies the inverse rotation and scaling mapping g^{-1}(x, y) (den Hollander, 2025)
    to transform 2D coordinates into ellipsoidal radial distance r from epicenter (x0, y0).
    Supports both scalar coordinates and NumPy meshgrid arrays.
    """
    phi = np.radians(phi_deg)
    dx = x - x0
    dy = y - y0
    
    # Inverse rotation matrix P^{-1} followed by axis scaling (a, b)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    x_rot = dx * cos_p + dy * sin_p
    y_rot = -dx * sin_p + dy * cos_p
    
    x_hat = x_rot / a
    y_hat = y_rot / b
    
    return np.sqrt(x_hat**2 + y_hat**2)


def evaluate_kernel(r, kernel_type, h, gamma):
    """
    Evaluates generalized attenuation kernels f(r; h, gamma) mapping transformed 
    distance r to point failure probability in [0, gamma] (Silverman, 1986; den Hollander, 2025).
    """
    k_type = kernel_type.lower()
    r = np.asarray(r, dtype=float)
    
    if k_type in ['gaussian', 'normal']:
        # Unbounded domain: h represents the ~99%/95% effective cutoff radius
        return gamma * np.exp(-(r**2) / (2.0 * (h / 3.0)**2))
        
    elif k_type == 'exponential':
        return gamma * np.exp(-(3.0 * r) / h)
        
    elif k_type == 'epanechnikov':
        val = np.zeros_like(r)
        mask = r <= h
        val[mask] = gamma * (1.0 - (r[mask] / h)**2)
        return val
        
    elif k_type == 'biweight':
        val = np.zeros_like(r)
        mask = r <= h
        val[mask] = gamma * ((1.0 - (r[mask] / h)**2)**2)
        return val
        
    elif k_type == 'triangular':
        val = np.zeros_like(r)
        mask = r <= h
        val[mask] = gamma * (1.0 - (r[mask] / h))
        return val
        
    elif k_type == 'uniform':
        val = np.zeros_like(r)
        mask = r <= h
        val[mask] = gamma
        return val
        
    else:
        raise ValueError(f"Unsupported kernel type: '{kernel_type}'")


def evaluate_hazard_field(x, y, disaster_config):
    """
    Computes the cumulative point failure probability at coordinates (x, y) 
    by summing effects across all K disaster epicenters and clipping to [0, 1].
    """
    total_risk = np.zeros_like(np.asarray(x, dtype=float))
    
    for epi in disaster_config:
        r = inverse_ellipsoid_distance(
            x, y,
            x0=epi['x0'], y0=epi['y0'],
            a=epi.get('a', 1.0), b=epi.get('b', 1.0),
            phi_deg=epi.get('phi', 0.0)
        )
        point_prob = evaluate_kernel(
            r,
            kernel_type=epi['kernel'],
            h=epi['h'],
            gamma=epi['gamma']
        )
        total_risk = total_risk + point_prob
        
    return np.clip(total_risk, 0.0, 1.0)


# ==========================================
# 2. Required Core Functions
# ==========================================

def apply_disaster_probabilities(G, disaster_config, num_edge_samples=10, max_blockage_prob=0.92):
    """
    Evaluates m equidistant sample points along every edge e = (u, v) in G,
    computes point failure risks q_{e,j}, and aggregates them into discrete edge 
    blockage probabilities q_e via the Poisson Binomial zero-failure formula:
        q_e = 1 - prod_{j=1}^m (1 - q_{e,j}) (Wang, 1993; den Hollander, 2025).
        
    Attaches 'blockage_prob' (q_e) and 'point_probs' to each edge in G.
    """
    pos = nx.get_node_attributes(G, 'pos')
    
    for u, v in G.edges():
        x_u, y_u = pos[u]
        x_v, y_v = pos[v]
        
        # Sample m equidistant points from origin node (j=1) to destination node (j=m)
        xs = np.linspace(x_u, x_v, num_edge_samples)
        ys = np.linspace(y_u, y_v, num_edge_samples)
        
        # Compute point failure probabilities q_{e,j}
        q_ej = evaluate_hazard_field(xs, ys, disaster_config)
        
        # Poisson Binomial aggregation (Wang, 1993)
        survival_prob = np.prod(1.0 - q_ej)
        q_e = float(np.clip(1.0 - survival_prob, 0.0, max_blockage_prob))
        
        G.edges[u, v]['blockage_prob'] = q_e
        G.edges[u, v]['point_probs'] = q_ej.tolist()
        
    return G


def _resolve_od_pairs(G, od_pairs=None):
    """
    Resolves the active list of (O_k, D_k) pairs for single- or multi-agent routing.
    Defaults to `G.graph.get("od_pairs")` or `[(origin, destination)]`.
    """
    if od_pairs is not None and len(od_pairs) > 0:
        return [(int(o), int(d)) for o, d in od_pairs]

    graph_pairs = G.graph.get('od_pairs')
    if graph_pairs and len(graph_pairs) > 0:
        return [(int(o), int(d)) for o, d in graph_pairs]

    if 'origin' in G.graph and 'destination' in G.graph:
        return [(int(G.graph['origin']), int(G.graph['destination']))]

    nodes = list(G.nodes())
    origins = [n for n, d in G.nodes(data=True) if d.get('is_origin', False)]
    destinations = [n for n, d in G.nodes(data=True) if d.get('is_destination', False)]

    if origins and destinations and len(origins) == len(destinations):
        origins_sorted = sorted(origins, key=lambda n: (G.nodes[n].get('origin_agent_idx') is None, G.nodes[n].get('origin_agent_idx', 0)))
        dests_sorted = sorted(destinations, key=lambda n: (G.nodes[n].get('destination_agent_idx') is None, G.nodes[n].get('destination_agent_idx', 0)))
        return [(int(o), int(d)) for o, d in zip(origins_sorted, dests_sorted)]

    origin = int(origins[0]) if origins else int(min(nodes))
    destination = int(destinations[0]) if destinations else int(max(nodes))
    return [(origin, destination)]


def sample_disaster_realization(G, max_resamples=1000, allow_relaxation=True, od_pairs=None):
    """
    Draws independent Bernoulli trials B_e ~ Bern(q_e) for every edge in G.
    Verifies if every (O_k, D_k) in `od_pairs` (defaulting to G.graph.get("od_pairs")
    or [(origin, destination)]) remains connected in G_offline.
    Resamples edge statuses until a traversable O_k -> D_k path exists for all agents.
    If max_resamples is exceeded and allow_relaxation=True, gradually scales down
    blockage probabilities or preserves the safest O_k -> D_k paths so extreme disaster
    configurations remain playable without raising an exception.
    
    Returns:
        G_offline (nx.Graph): Graph with blocked edges removed.
        blocked_edges (list): List of edges (u, v) that failed in this realization.
    """
    pairs = _resolve_od_pairs(G, od_pairs=od_pairs)
    
    scale = 1.0
    max_relax_stages = 6 if allow_relaxation else 1
    attempts_per_stage = max(1, max_resamples // max_relax_stages) if allow_relaxation else max_resamples
    
    for stage in range(max_relax_stages):
        for _ in range(attempts_per_stage):
            G_offline = G.copy()
            blocked_edges = []
            
            for u, v, data in G.edges(data=True):
                q_e = data.get('blockage_prob', 0.0) * scale
                is_blocked = (np.random.rand() < q_e)
                
                # Store realized status on the original graph as well
                G.edges[u, v]['is_blocked'] = int(is_blocked)
                
                if is_blocked:
                    blocked_edges.append((u, v))
                    G_offline.remove_edge(u, v)
                    
            if all(nx.has_path(G_offline, o_k, d_k) for o_k, d_k in pairs):
                return G_offline, blocked_edges
                
        scale *= 0.80
        
    if allow_relaxation and all(nx.has_path(G, o_k, d_k) for o_k, d_k in pairs):
        # Fallback: preserve the minimum-risk O_k -> D_k path for every agent to guarantee connectivity
        def edge_risk_weight(u, v, d):
            q = min(0.999, d.get('blockage_prob', 0.0))
            return -np.log(1.0 - q)
            
        protected_edges = set()
        for o_k, d_k in pairs:
            safest_path = nx.shortest_path(G, o_k, d_k, weight=edge_risk_weight)
            for idx in range(len(safest_path) - 1):
                u_p, v_p = safest_path[idx], safest_path[idx + 1]
                protected_edges.add((min(u_p, v_p), max(u_p, v_p)))
        
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
        f"Failed to sample an O->D connected G_offline for pairs {pairs} after {max_resamples} attempts. "
        "Consider reducing disaster intensity (gamma) or domain radius (h)."
    )


def visualize_graph_with_disaster(G, disaster_config, show_labels=True, grid_res=200):
    """
    Renders the spatial network overlaid on the continuous 2D hazard heatmap:
    - Background: Continuous point-failure probability field f(r) using RdYlGn_r / YlOrRd.
    - Edges: Color-coded by aggregate blockage probability q_e (YlOrRd gradient).
    - Labels: Annotates numerical q_e values next to each edge segment.
    - Nodes: Plots intermediate nodes and single or multiple (O_k, D_k) pairs.
    """
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_title("Spatial Road Network & Disaster Hazard Field ($q_e$)", fontsize=15, fontweight='bold')
    
    # 1. Background Continuous Hazard Field Meshgrid
    gx = np.linspace(0.0, 1.0, grid_res)
    gy = np.linspace(0.0, 1.0, grid_res)
    X, Y = np.meshgrid(gx, gy)
    Z = evaluate_hazard_field(X, Y, disaster_config)
    
    # Render filled contour background (Green = safe, Yellow/Red = high point risk)
    bg_cmap = mpl.colormaps['RdYlGn_r']
    contour = ax.contourf(X, Y, Z, levels=50, cmap=bg_cmap, alpha=0.55, vmin=0.0, vmax=max(0.15, Z.max()))
    cbar_bg = fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.02)
    cbar_bg.set_label("Point Failure Probability $f(r)$", fontsize=10)
    
    # Plot Epicenter markers
    epi_x = [epi['x0'] for epi in disaster_config]
    epi_y = [epi['y0'] for epi in disaster_config]
    ax.scatter(epi_x, epi_y, s=140, facecolors='none', edgecolors='black', linewidths=2.0, zorder=6, label='Epicenters')
    
    # 2. Extract Node Positions & Categories
    pos = nx.get_node_attributes(G, 'pos')
    pairs = _resolve_od_pairs(G)
    od_node_set = {n for pair in pairs for n in pair}
    standard_nodes = [n for n in G.nodes() if n not in od_node_set]
    
    # 3. Draw Edges Color-Coded by Blockage Probability q_e
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
        
    lc = LineCollection(segments, cmap=edge_cmap, norm=edge_norm, linewidths=linewidths, alpha=0.95, zorder=3)
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
        
    # 5. Draw Nodes
    nx.draw_networkx_nodes(
        G, pos, nodelist=standard_nodes, node_size=140,
        node_color='white', edgecolors='black', linewidths=1.2, ax=ax
    )

    agent_palette = [
        ('dodgerblue', 'crimson'),
        ('limegreen', 'darkorange'),
        ('magenta', 'gold'),
        ('cyan', 'blueviolet'),
    ]
    legend_handles = []
    role_tags = {}

    for k_idx, (o_node, d_node) in enumerate(pairs):
        o_col, d_col = agent_palette[k_idx % len(agent_palette)]
        if len(pairs) > 1:
            o_col = d_col = ['dodgerblue', 'limegreen', 'magenta', 'darkorange', 'cyan'][k_idx % 5]
        nx.draw_networkx_nodes(
            G, pos, nodelist=[o_node], node_size=380,
            node_color=o_col, node_shape='*', edgecolors='black', linewidths=1.5, ax=ax
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=[d_node], node_size=320,
            node_color=d_col, node_shape='X', edgecolors='black', linewidths=1.5, ax=ax
        )
        tag_o = f"O{k_idx + 1}" if len(pairs) > 1 else "O"
        tag_d = f"D{k_idx + 1}" if len(pairs) > 1 else "D"
        role_tags.setdefault(o_node, []).append(tag_o)
        role_tags.setdefault(d_node, []).append(tag_d)

        legend_handles.append(
            Line2D([0], [0], marker='*', color='w', label=f'Origin {tag_o} (Node {o_node})',
                   markerfacecolor=o_col, markersize=14, markeredgecolor='k')
        )
        legend_handles.append(
            Line2D([0], [0], marker='X', color='w', label=f'Destination {tag_d} (Node {d_node})',
                   markerfacecolor=d_col, markersize=12, markeredgecolor='k')
        )

    nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax)
    for node_id, tags in role_tags.items():
        x_n, y_n = pos[node_id]
        ax.annotate(
            "/".join(tags),
            xy=(x_n, y_n),
            xytext=(6, 8),
            textcoords='offset points',
            fontsize=8.5,
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.18', facecolor='yellow', alpha=0.85, edgecolor='black', lw=0.7),
            zorder=8,
        )
    
    # 6. Plot Formatting & Legend
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.35)
    
    legend_handles.extend([
        Line2D([0], [0], marker='o', color='w', label='Intermediate Node', markerfacecolor='white', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Disaster Epicenter', markerfacecolor='none', markersize=10, markeredgecolor='k', markeredgewidth=2)
    ])
    ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    plt.show()

def generate_random_disaster(num_epicenters=3, G=None, safe_buffer=0.12):
    """
    Procedurally generates random disaster epicenters and ellipsoids.
    Scales epicenter radius and intensity softly when num_epicenters is large,
    and optionally keeps epicenters a safe buffer away from Origin/Destination.
    """
    kernels = ['gaussian', 'exponential', 'epanechnikov', 'biweight', 'triangular', 'uniform']
    disaster_config = []
    
    # Scale intensity and radius slightly when many epicenters are requested
    scale_factor = 1.0 / np.sqrt(max(1.0, num_epicenters / 3.0))
    h_max = max(0.18, 0.40 * scale_factor)
    gamma_max = max(0.08, 0.20 * scale_factor)
    
    protected_coords = []
    if G is not None:
        pos = nx.get_node_attributes(G, 'pos')
        pairs = _resolve_od_pairs(G)
        for o_n, d_n in pairs:
            protected_coords.append(np.array(pos[o_n]))
            protected_coords.append(np.array(pos[d_n]))
    
    for _ in range(num_epicenters):
        for _ in range(100):
            x0 = float(np.random.uniform(0.1, 0.9))
            y0 = float(np.random.uniform(0.1, 0.9))
            if not protected_coords or all(
                np.linalg.norm(np.array([x0, y0]) - pc) >= safe_buffer for pc in protected_coords
            ):
                break
                
        disaster_config.append({
            'x0': x0,
            'y0': y0,
            'kernel': str(np.random.choice(kernels)),
            'h': float(np.random.uniform(0.15, h_max)),
            'gamma': float(np.random.uniform(0.05, gamma_max)),
            'a': 1.0,
            'b': float(np.random.uniform(0.3, 0.9)),
            'phi': float(np.random.uniform(0.0, 180.0))
        })
    return disaster_config

# ==========================================
# 3. Example Execution Script
# ==========================================
if __name__ == "__main__":
    from graph_builder import SpatialGraphGenerator, sample_od_pairs
    
    # 1. Build a synthetic spatial network
    builder = SpatialGraphGenerator(n_nodes=45, d_min=0.04)
    G_base, _ = builder.build_graph(regime='multidistrict', C=4)
    
    # 2. Sample fresh 1-agent OD pairs across multiple disaster realizations
    print("=== Single-Agent Multi-Realization Ellipsoidal Disaster Verification ===")
    for trial in range(1, 4):
        G_trial = G_base.copy()
        od_1 = sample_od_pairs(G_trial, num_agents=1, rng=10 + trial)
        disaster_cfg = generate_random_disaster(num_epicenters=3, G=G_trial)
        G_trial = apply_disaster_probabilities(G_trial, disaster_cfg, num_edge_samples=10)
        G_offline, blocked_edges = sample_disaster_realization(G_trial, od_pairs=od_1)
        o1, d1 = od_1[0]
        print(
            f"[Trial {trial} | 1-Agent OD={od_1[0]}] Total Edges: {G_trial.number_of_edges()} | "
            f"Blocked: {len(blocked_edges)} | Remaining: {G_offline.number_of_edges()} | "
            f"Connected(O1->D1): {nx.has_path(G_offline, o1, d1)}"
        )

    # 3. Sample 2-agent and 3-agent perimeter OD pairs and verify multi-agent G_offline connectivity
    print("\n=== Multi-Agent Ellipsoidal Disaster Connectivity Verification ===")
    for num_agents in [2, 3]:
        G_multi = G_base.copy()
        od_k = sample_od_pairs(G_multi, num_agents=num_agents, avoid_od_overlap=True, rng=50 + num_agents)
        disaster_config = generate_random_disaster(num_epicenters=3, G=G_multi)
        G_multi = apply_disaster_probabilities(G_multi, disaster_config, num_edge_samples=10)
        G_offline, blocked_edges = sample_disaster_realization(G_multi, od_pairs=od_k)
        conn_status = [nx.has_path(G_offline, o, d) for o, d in od_k]
        print(
            f"[{num_agents}-Agent Realization] od_pairs={od_k} | "
            f"Blocked: {len(blocked_edges)}/{G_multi.number_of_edges()} | "
            f"All Agents Connected in G_offline: {all(conn_status)} ({conn_status})"
        )
        visualize_graph_with_disaster(G_multi, disaster_config, show_labels=True)