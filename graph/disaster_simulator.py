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

def apply_disaster_probabilities(G, disaster_config, num_edge_samples=10):
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
        q_e = float(1.0 - survival_prob)
        
        G.edges[u, v]['blockage_prob'] = q_e
        G.edges[u, v]['point_probs'] = q_ej.tolist()
        
    return G


def sample_disaster_realization(G, max_resamples=1000):
    """
    Draws independent Bernoulli trials B_e ~ Bern(q_e) for every edge in G.
    Verifies if Origin (0) -> Destination (N-1) remains connected in G_offline.
    Resamples edge statuses until a traversable O -> D path exists (den Hollander, 2025).
    
    Returns:
        G_offline (nx.Graph): Graph with blocked edges removed.
        blocked_edges (list): List of edges (u, v) that failed in this realization.
    """
    nodes = list(G.nodes())
    origins = [n for n, d in G.nodes(data=True) if d.get('is_origin', False)]
    destinations = [n for n, d in G.nodes(data=True) if d.get('is_destination', False)]
    
    origin = origins[0] if origins else min(nodes)
    destination = destinations[0] if destinations else max(nodes)
    
    for attempt in range(1, max_resamples + 1):
        G_offline = G.copy()
        blocked_edges = []
        
        for u, v, data in G.edges(data=True):
            q_e = data.get('blockage_prob', 0.0)
            is_blocked = (np.random.rand() < q_e)
            
            # Store realized status on the original graph as well
            G.edges[u, v]['is_blocked'] = int(is_blocked)
            
            if is_blocked:
                blocked_edges.append((u, v))
                G_offline.remove_edge(u, v)
                
        if nx.has_path(G_offline, origin, destination):
            return G_offline, blocked_edges
            
    raise RuntimeError(
        f"Failed to sample an O->D connected G_offline after {max_resamples} attempts. "
        "Consider reducing disaster intensity (gamma) or domain radius (h)."
    )


def visualize_graph_with_disaster(G, disaster_config, show_labels=True, grid_res=200):
    """
    Renders the spatial network overlaid on the continuous 2D hazard heatmap:
    - Background: Continuous point-failure probability field f(r) using RdYlGn_r / YlOrRd.
    - Edges: Color-coded by aggregate blockage probability q_e (YlOrRd gradient).
    - Labels: Annotates numerical q_e values next to each edge segment.
    - Nodes: Plots intermediate nodes, Origin (O), and Destination (D).
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
    origins = [n for n, attr in G.nodes(data=True) if attr.get('is_origin')]
    destinations = [n for n, attr in G.nodes(data=True) if attr.get('is_destination')]
    standard_nodes = [n for n in G.nodes() if n not in origins and n not in destinations]
    
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
        Line2D([0], [0], marker='*', color='w', label='Origin (O)', markerfacecolor='dodgerblue', markersize=14, markeredgecolor='k'),
        Line2D([0], [0], marker='X', color='w', label='Destination (D)', markerfacecolor='crimson', markersize=12, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Intermediate Node', markerfacecolor='white', markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', label='Disaster Epicenter', markerfacecolor='none', markersize=10, markeredgecolor='k', markeredgewidth=2)
    ]
    ax.legend(handles=legend_handles, loc='upper left', framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    plt.show()

def generate_random_disaster(num_epicenters=3):
    """Procedurally generates random disaster epicenters and ellipsoids."""
    kernels = ['gaussian', 'exponential', 'epanechnikov', 'biweight', 'triangular', 'uniform']
    disaster_config = []
    
    for _ in range(num_epicenters):
        disaster_config.append({
            'x0': float(np.random.uniform(0.1, 0.9)),
            'y0': float(np.random.uniform(0.1, 0.9)),
            'kernel': str(np.random.choice(kernels)),
            'h': float(np.random.uniform(0.15, 0.40)),
            'gamma': float(np.random.uniform(0.05, 0.20)),
            'a': 1.0,
            'b': float(np.random.uniform(0.3, 0.9)),
            'phi': float(np.random.uniform(0.0, 180.0))
        })
    return disaster_config

# ==========================================
# 3. Example Execution Script
# ==========================================
if __name__ == "__main__":
    from graph_builder import SpatialGraphGenerator
    
    #np.random.seed(42)
    
    # 1. Build a 50-node synthetic spatial network (den Hollander Base Case scale)
    builder = SpatialGraphGenerator(n_nodes=50, d_min=0.03)
    G, _ = builder.build_graph(regime='multidistrict', C=3)
    
    # 2. Define the 4-Epicenter Ellipsoidal Disaster Scenario (den Hollander, 2025 - Eq. 8.1)
    # K = {(x0, y0, f, h, gamma, a, b, phi)_k}
    disaster_config = generate_random_disaster(num_epicenters=4)
    
    # 3. Compute Point-to-Edge Risks and Poisson Binomial Edge Blockage Probabilities (m = 10)
    G = apply_disaster_probabilities(G, disaster_config, num_edge_samples=10)
    
    # 4. Sample a valid O -> D connected disaster realization
    G_offline, blocked_edges = sample_disaster_realization(G)
    print(f"Total Edges: {G.number_of_edges()} | Blocked in Realization: {len(blocked_edges)} | Remaining in G_offline: {G_offline.number_of_edges()}")
    
    # 5. Render the Hazard Field, Network Topology, and Edge Blockage Probabilities q_e
    visualize_graph_with_disaster(G, disaster_config, show_labels=True)