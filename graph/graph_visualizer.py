# Script to visualize a graph using matplotlib and networkx

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from graph_builder import SpatialGraphGenerator

def visualize_spatial_network(G, title="Spatial Network Visualization"):
    """
    Visualizes the CTP spatial graph, highlighting node roles, 
    community clusters, and edge attributes (highways vs local).
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Extract positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Categorize nodes
    origins = [n for n, attr in G.nodes(data=True) if attr.get('is_origin')]
    destinations = [n for n, attr in G.nodes(data=True) if attr.get('is_destination')]
    standard_nodes = [n for n in G.nodes() if n not in origins and n not in destinations]
    
    # Extract cluster IDs for color mapping
    clusters = [G.nodes[n]['cluster_id'] for n in standard_nodes]
    n_colors = max(clusters) + 1 if clusters else 1
    cmap = plt.colormaps['tab10'].resampled(n_colors)
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, nodelist=standard_nodes, node_size=150, 
                           node_color=clusters, cmap=cmap, alpha=0.8, 
                           edgecolors='black', ax=ax, label="Intermediate Nodes")
                           
    # Draw Origin
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=origins, 
        node_size=400, 
        node_color='lime', 
        node_shape='*', 
        edgecolors='black', 
        ax=ax, 
        label="Origin (O)"
    )
                           
    # Draw Destination
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=destinations, 
        node_size=400, 
        node_color='red', 
        node_shape='X', 
        edgecolors='black', 
        ax=ax, 
        label="Destination (D)"
    )

    # Categorize edges
    highways = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('is_highway') == 1]
    local_roads = [(u, v) for u, v, attr in G.edges(data=True) if attr.get('is_highway') == 0]
    
    # Draw Edges
    nx.draw_networkx_edges(G, pos, edgelist=local_roads, width=1.0, 
                           alpha=0.5, edge_color='gray', ax=ax)
                           
    nx.draw_networkx_edges(G, pos, edgelist=highways, width=2.5, 
                           alpha=0.8, edge_color='darkblue', style='dashed', ax=ax)

    # Edge Labels (Length and Travel Time)
    edge_labels = {}
    for u, v, attr in G.edges(data=True):
        length = attr.get('length', 0)
        time = attr.get('travel_time', 0)
        edge_labels[(u, v)] = f"L:{length:.2f}\nT:{time:.2f}"
        
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                 font_size=7, font_color='darkred', 
                                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Draw Node Labels (IDs)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif', ax=ax)

    # Formatting
    ax.set_facecolor('#f9f9f9')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Custom Legend for Edges
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='darkblue', lw=2.5, linestyle='dashed', label='Highway (Inter-cluster)'),
        Line2D([0], [0], color='gray', lw=1.0, label='Local Street (Intra-cluster)'),
        Line2D([0], [0], marker='*', color='w', label='Origin', markerfacecolor='lime', markersize=15, markeredgecolor='k'),
        Line2D([0], [0], marker='X', color='w', label='Destination', markerfacecolor='red', markersize=15, markeredgecolor='k')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.show()

# example usage
if __name__ == "__main__":
    # Generate a sample spatial graph
    generator = SpatialGraphGenerator(n_nodes=30, d_min=0.03)
    G_multi, _ = generator.build_graph(regime='multidistrict', C=4)
    
    # Visualize the generated graph
    visualize_spatial_network(G_multi, title="Sample Multidistrict Spatial Network")