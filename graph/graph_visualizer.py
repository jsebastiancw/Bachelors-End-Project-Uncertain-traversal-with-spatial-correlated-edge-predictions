# Script to visualize a graph using matplotlib and networkx

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
from graph_builder import SpatialGraphGenerator, sample_od_pairs


def _resolve_od_pairs(G, od_pairs=None):
    """
    Resolves the list of (O_k, D_k) pairs from explicit argument, G.graph["od_pairs"],
    G.graph["origin"]/["destination"], or node attributes.
    """
    if od_pairs is not None and len(od_pairs) > 0:
        return [(int(o), int(d)) for o, d in od_pairs]

    graph_pairs = G.graph.get('od_pairs')
    if graph_pairs and len(graph_pairs) > 0:
        return [(int(o), int(d)) for o, d in graph_pairs]

    if 'origin' in G.graph and 'destination' in G.graph:
        return [(int(G.graph['origin']), int(G.graph['destination']))]

    nodes = list(G.nodes())
    origins = [n for n, attr in G.nodes(data=True) if attr.get('is_origin')]
    destinations = [n for n, attr in G.nodes(data=True) if attr.get('is_destination')]

    if origins and destinations and len(origins) == len(destinations):
        origins_sorted = sorted(origins, key=lambda n: (G.nodes[n].get('origin_agent_idx') is None, G.nodes[n].get('origin_agent_idx', 0)))
        dests_sorted = sorted(destinations, key=lambda n: (G.nodes[n].get('destination_agent_idx') is None, G.nodes[n].get('destination_agent_idx', 0)))
        return [(int(o), int(d)) for o, d in zip(origins_sorted, dests_sorted)]

    origin = int(origins[0]) if origins else int(min(nodes))
    destination = int(destinations[0]) if destinations else int(max(nodes))
    return [(origin, destination)]


def visualize_spatial_network(G, title="Spatial Network Visualization", od_pairs=None, show_edge_labels=True):
    """
    Visualizes the CTP spatial graph, highlighting single- or multi-agent (O_k, D_k)
    pairs with distinct colors/markers and clear labels (e.g., O1/D1, O2/D2, O3/D3),
    community clusters, and edge attributes (highways vs local).
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Extract positions and resolve OD pairs
    pos = nx.get_node_attributes(G, 'pos')
    pairs = _resolve_od_pairs(G, od_pairs=od_pairs)
    od_node_set = {n for pair in pairs for n in pair}
    standard_nodes = [n for n in G.nodes() if n not in od_node_set]
    
    # Extract cluster IDs for color mapping
    clusters = [G.nodes[n].get('cluster_id', 0) for n in standard_nodes]
    n_colors = max(clusters) + 1 if clusters else 1
    cmap = plt.colormaps['tab10'].resampled(max(1, n_colors))
    
    # Draw Intermediate Nodes
    if standard_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=standard_nodes, node_size=150,
            node_color=clusters, cmap=cmap, alpha=0.8,
            edgecolors='black', ax=ax, label="Intermediate Nodes"
        )

    # Distinct colors for each agent pair (O_k, D_k)
    # Single-agent uses classic lime star / red X; multi-agent uses distinct per-agent colors
    multi_agent_colors = ['limegreen', 'dodgerblue', 'magenta', 'darkorange', 'cyan', 'gold']
    od_legend_handles = []
    role_tags = {}

    for idx, (o_node, d_node) in enumerate(pairs):
        if len(pairs) == 1:
            o_color, d_color = 'lime', 'red'
            o_tag, d_tag = 'O1', 'D1'
            o_legend_lbl = f"Origin O1 (Node {o_node})"
            d_legend_lbl = f"Destination D1 (Node {d_node})"
        else:
            pair_color = multi_agent_colors[idx % len(multi_agent_colors)]
            o_color, d_color = pair_color, pair_color
            o_tag, d_tag = f"O{idx + 1}", f"D{idx + 1}"
            o_legend_lbl = f"Agent {idx + 1} Origin ({o_tag}: Node {o_node})"
            d_legend_lbl = f"Agent {idx + 1} Destination ({d_tag}: Node {d_node})"

        nx.draw_networkx_nodes(
            G, pos, nodelist=[o_node], node_size=420,
            node_color=o_color, node_shape='*', edgecolors='black',
            linewidths=1.5, ax=ax
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=[d_node], node_size=360,
            node_color=d_color, node_shape='X', edgecolors='black',
            linewidths=1.5, ax=ax
        )

        role_tags.setdefault(o_node, []).append(o_tag)
        role_tags.setdefault(d_node, []).append(d_tag)

        od_legend_handles.append(
            Line2D([0], [0], marker='*', color='w', label=o_legend_lbl,
                   markerfacecolor=o_color, markersize=15, markeredgecolor='k')
        )
        od_legend_handles.append(
            Line2D([0], [0], marker='X', color='w', label=d_legend_lbl,
                   markerfacecolor=d_color, markersize=13, markeredgecolor='k')
        )

    # Categorize edges
    highways = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('is_highway') == 1]
    local_roads = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('is_highway') == 0]
    
    # Draw Edges
    nx.draw_networkx_edges(G, pos, edgelist=local_roads, width=1.5, 
                           alpha=0.65, edge_color='dimgray', ax=ax)
                           
    nx.draw_networkx_edges(G, pos, edgelist=highways, width=2.2, 
                           alpha=0.8, edge_color='darkblue', style='dashed', ax=ax)

    # Edge Labels (Length and Travel Time)
    if show_edge_labels:
        edge_labels = {}
        for u, v, attr in G.edges(data=True):
            length = attr.get('length', 0)
            time = attr.get('travel_time', 0)
            edge_labels[(u, v)] = f"L:{length:.2f}\nT:{time:.2f}"
            
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels,
            font_size=7, font_color='darkred',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            ax=ax
        )

    # Draw Node Labels (IDs) and explicit O_k / D_k callout badges
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', ax=ax)
    for node_id, tags in role_tags.items():
        x_n, y_n = pos[node_id]
        ax.annotate(
            "/".join(tags),
            xy=(x_n, y_n),
            xytext=(7, 9),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.88, edgecolor='black', lw=0.8),
            zorder=8,
        )

    # Formatting
    ax.set_facecolor('#f9f9f9')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Custom Legend for Edges and OD Pairs
    legend_elements = [
        Line2D([0], [0], color='darkblue', lw=2.5, linestyle='dashed', label='Highway (Inter-cluster)'),
        Line2D([0], [0], color='gray', lw=1.0, label='Local Street (Intra-cluster)'),
    ] + od_legend_handles
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9.5, framealpha=0.9)

    plt.tight_layout()
    plt.show()


# ==========================================
# Example Usage & Multi-Agent Verification
# ==========================================
if __name__ == "__main__":
    from gaussian_disaster_simulator import set_disaster_type, sample_gp_disaster_realization

    generator = SpatialGraphGenerator(n_nodes=50, d_min=0.05)
    G_multi, _ = generator.build_graph(regime='multidistrict', C=6)
    pos = nx.get_node_attributes(G_multi, 'pos')
    center = np.array([0.5, 0.5])

    # 1. Verify 1-agent OD resampling across multiple disaster realizations on a single base graph
    print("=== 1-Agent OD Resampling Across Multiple Disaster Realizations ===")
    for r_idx in range(1, 4):
        G_run = G_multi.copy()
        od_1 = sample_od_pairs(G_run, num_agents=1, rng=r_idx)
        G_run, _ = set_disaster_type(G_run, disaster_type='earthquake', num_epicenters=1, seed=r_idx)
        G_off, blocked = sample_gp_disaster_realization(G_run, od_pairs=od_1)
        o1, d1 = od_1[0]
        dist_od = float(np.linalg.norm(np.asarray(pos[o1]) - np.asarray(pos[d1])))
        print(
            f"  Scenario {r_idx}: OD={od_1[0]} | dist(O1,D1)={dist_od:.3f} | "
            f"Blocked={len(blocked)} | G_offline connected={nx.has_path(G_off, o1, d1)}"
        )

    # 2. Verify 2-agent and 3-agent perimeter OD sampling, distance metrics, and visualization
    for k_agents in [1, 2, 3]:
        od_k = sample_od_pairs(G_multi, num_agents=k_agents, avoid_od_overlap=True, rng=40 + k_agents)
        print(f"\n=== {k_agents}-Agent Perimeter OD Metrics ===")
        for idx, (o_n, d_n) in enumerate(od_k, start=1):
            po, pd = np.asarray(pos[o_n]), np.asarray(pos[d_n])
            print(
                f"  Agent {idx}: O{idx}={o_n} (dist_center={np.linalg.norm(po - center):.3f}) -> "
                f"D{idx}={d_n} (dist_center={np.linalg.norm(pd - center):.3f}) | "
                f"dist(O{idx}, D{idx})={np.linalg.norm(po - pd):.3f}"
            )
        if k_agents > 1:
            o_seps = [
                float(np.linalg.norm(np.asarray(pos[od_k[i][0]]) - np.asarray(pos[od_k[j][0]])))
                for i in range(k_agents) for j in range(i + 1, k_agents)
            ]
            d_seps = [
                float(np.linalg.norm(np.asarray(pos[od_k[i][1]]) - np.asarray(pos[od_k[j][1]])))
                for i in range(k_agents) for j in range(i + 1, k_agents)
            ]
            print(f"  Min Inter-Origin Separation: {min(o_seps):.3f} | Min Inter-Destination Separation: {min(d_seps):.3f}")

        visualize_spatial_network(
            G_multi,
            title=f"Multidistrict Spatial Network ({k_agents}-Agent Perimeter OD Pairs)",
            od_pairs=od_k,
        )