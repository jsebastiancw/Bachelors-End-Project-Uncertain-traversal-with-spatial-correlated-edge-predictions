"""
graph_builder.py
Procedural Spatial Graph Generator for DRL under Canadian Traveller Problem (CTP).
"""

import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
import torch
from torch_geometric.data import Data

class SpatialGraphGenerator:
    """
    Generates spatially embedded planar graphs in R^2 based on Barthélemy (2011) and 
    Geometric Block Models (Galhotra et al., 2018).
    """
    def __init__(self, n_nodes=50, d_min=0.05, max_attempts=10000):
        self.N = n_nodes
        self.d_min = d_min
        self.max_attempts = max_attempts
        
    def _sample_nodes(self, num_nodes, bounds, existing_nodes=None, d_min=None, allow_relaxation=True):
        """
        Rejection sampling algorithm (den Hollander, 2025) to place nodes.
        Enforces minimum node separation distance d_min, with optional adaptive
        relaxation to prevent jamming failures at high node/cluster densities.
        """
        nodes = [] if existing_nodes is None else existing_nodes.tolist()
        target_len = num_nodes if existing_nodes is None else len(existing_nodes) + num_nodes
        x_min, x_max, y_min, y_max = bounds
        current_d_min = self.d_min if d_min is None else d_min
        
        attempts = 0
        relaxations = 0
        max_relaxations = 25
        
        while len(nodes) < target_len:
            if attempts > self.max_attempts:
                if allow_relaxation and relaxations < max_relaxations:
                    current_d_min *= 0.85
                    attempts = 0
                    relaxations += 1
                else:
                    raise ValueError("Max attempts reached. Decrease d_min or N.")
                
            candidate = [np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)]
            if not nodes:
                nodes.append(candidate)
                continue
                
            dists = np.linalg.norm(np.array(nodes) - np.array(candidate), axis=1)
            if np.all(dists >= current_d_min):
                nodes.append(candidate)
            attempts += 1
            
        return np.array(nodes[-num_nodes:])

    def _generate_node_placements(self, regime, C=3):
        """
        Places the Origin (O), Destination (D), and intermediate nodes.
        O is restricted to [0, 1/3]^2 and D to [2/3, 1]^2.
        """
        pos = np.zeros((self.N, 2))
        cluster_ids = np.zeros(self.N, dtype=int)
        
        # 1. Place Origin (Index 0) and Destination (Index N-1)
        pos[0] = self._sample_nodes(1, [0.0, 0.33, 0.0, 0.33])[0]
        pos[self.N-1] = self._sample_nodes(1, [0.66, 1.0, 0.66, 1.0], existing_nodes=pos[:1])[0]
        
        if regime in ['sparse', 'dense']:
            # Uniformly distribute remaining nodes across the full [0,1]^2 space
            pos[1:self.N-1] = self._sample_nodes(
                self.N - 2, [0.0, 1.0, 0.0, 1.0], existing_nodes=pos[[0, self.N-1]]
            )
            
        elif regime == 'multidistrict':
            # Geometric Block Model: Partition space into C well-separated centroids (Galhotra et al., 2018)
            centroid_d_min = min(0.45, 0.65 / np.sqrt(max(1, C)))
            centroids = self._sample_nodes(C, [0.12, 0.88, 0.12, 0.88], d_min=centroid_d_min)
            
            # Scale cluster radius with C so districts remain visually compact and distinct
            cluster_radius = min(0.14, 0.28 / np.sqrt(max(1, C)))
            
            intermediate_count = self.N - 2
            base_count = intermediate_count // C
            remainder = intermediate_count % C
            
            # Include both Origin and Destination in existing_nodes from the start
            placed_nodes = [pos[0].tolist(), pos[self.N-1].tolist()]
            
            idx = 1
            for c in range(C):
                n_cluster = base_count + (1 if c < remainder else 0)
                if n_cluster <= 0:
                    continue
                cluster_ids[idx:idx+n_cluster] = c + 1
                cx, cy = centroids[c]
                # Sample locally within compact bounds around each district centroid
                bounds = [
                    max(0.0, cx - cluster_radius), min(1.0, cx + cluster_radius),
                    max(0.0, cy - cluster_radius), min(1.0, cy + cluster_radius)
                ]
                sampled = self._sample_nodes(
                    n_cluster, bounds, existing_nodes=np.array(placed_nodes)
                )
                pos[idx:idx+n_cluster] = sampled
                placed_nodes.extend(sampled.tolist())
                idx += n_cluster
                
            # Assign Origin and Destination to their nearest district cluster
            cluster_ids[0] = int(np.argmin(np.linalg.norm(centroids - pos[0], axis=1)) + 1)
            cluster_ids[self.N-1] = int(np.argmin(np.linalg.norm(centroids - pos[self.N-1], axis=1)) + 1)
                
        return pos, cluster_ids

    def _draw_edges(self, pos, cluster_ids, regime, params):
        """
        Calculates physical lengths and draws edges via Bernoulli trials based on Beta distributions.
        """
        dist_mat = distance_matrix(pos, pos)
        G = nx.Graph()
        
        for i in range(self.N):
            G.add_node(i, 
                       pos=(pos[i][0], pos[i][1]), 
                       is_origin=(i==0), 
                       is_destination=(i==self.N-1),
                       cluster_id=cluster_ids[i])
            
        for i in range(self.N):
            for j in range(i + 1, self.N):
                w_e = dist_mat[i, j]
                
                # Retrieve regime-specific thresholds and Beta parameters
                if regime in ['sparse', 'dense']:
                    l_max = params['l_max']
                    alpha, beta = params['alpha'], params['beta']
                    v_e = params['v_e']
                    is_highway = 1 if regime == 'sparse' else 0
                else: # Multi-district (SBM + RGG)
                    if cluster_ids[i] == cluster_ids[j]:
                        l_max, alpha, beta, v_e = params['r_s'], params['alpha_local'], params['beta'], params['v_local']
                        is_highway = 0
                    else:
                        l_max, alpha, beta, v_e = params['r_d'], params['alpha_highway'], params['beta'], params['v_highway']
                        is_highway = 1
                
                # Deterministic length cutoff
                if w_e > l_max:
                    continue
                    
                # Stochastic existence via Beta-Bernoulli draw
                p_e = np.random.beta(alpha, beta)
                if np.random.rand() < p_e:
                    c_e = w_e / v_e # Physical travel cost (Barthélemy, 2011)
                    G.add_edge(i, j, length=w_e, travel_time=c_e, is_highway=is_highway)
                    
        return G

    def build_graph(self, regime='dense', C=3):
        """
        Full construction pipeline verifying global O -> D connectivity.
        """
        # Regime hyperparameters
        regime_params = {
            'sparse': {'l_max': 0.22, 'alpha': 2, 'beta': 5, 'v_e': 2.0},
            'dense': {'l_max': 0.28, 'alpha': 5, 'beta': 2, 'v_e': 1.0},
            'multidistrict': {
                'r_s': 0.22, 'alpha_local': 5, 'v_local': 1.0,     # Intra-cluster (r_s)
                'r_d': 0.45, 'alpha_highway': 1.5, 'v_highway': 2.5, # Inter-cluster (r_d)
                'beta': 2
            }
        }
        params = dict(regime_params[regime])
        
        connected = False
        pos, cluster_ids = self._generate_node_placements(regime, C)
        
        # Resample edge draws until O -> D is traversable
        attempts = 0
        placement_resets = 0
        while not connected:
            if attempts > 50:
                # Reset node placement if edge density is too low to connect
                pos, cluster_ids = self._generate_node_placements(regime, C)
                attempts = 0
                placement_resets += 1
                # Gradually expand cutoff radius if extreme parameters prevent O->D connectivity
                if placement_resets % 3 == 0:
                    if 'l_max' in params:
                        params['l_max'] = min(1.0, params['l_max'] * 1.15)
                    else:
                        params['r_d'] = min(1.0, params['r_d'] * 1.15)
                        params['r_s'] = min(1.0, params['r_s'] * 1.10)
                
            G = self._draw_edges(pos, cluster_ids, regime, params)
            connected = nx.has_path(G, 0, self.N - 1)
            attempts += 1
            
        G_directed = G.to_directed()
        return G, G_directed

    def to_pyg(self, G):
        """
        Converts the NetworkX graph into PyTorch Geometric format for MARVEL/SEVAC architectures.
        """
        # Node Features [x, y, is_origin, is_destination, cluster_id]
        x = []
        for i in range(self.N):
            node = G.nodes[i]
            x.append([node['pos'][0], node['pos'][1], float(node['is_origin']), float(node['is_destination']), float(node['cluster_id'])])
        x = torch.tensor(x, dtype=torch.float)
        
        # Edge Index and Edge Features [length, travel_time, is_highway]
        edge_index = []
        edge_attr = []
        for u, v, data in G.edges(data=True):
            edge_index.append([u, v])
            edge_index.append([v, u]) # Undirected equivalence in PyG
            attr = [data['length'], data['travel_time'], float(data['is_highway'])]
            edge_attr.append(attr)
            edge_attr.append(attr)
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    builder = SpatialGraphGenerator(n_nodes=100, d_min=0.03)
    
    # Generate networks
    G_sparse, _ = builder.build_graph(regime='sparse')
    G_dense, _ = builder.build_graph(regime='dense')
    G_multi, _ = builder.build_graph(regime='multidistrict', C=4)
    
    # Convert to DRL tensor format
    pyg_data = builder.to_pyg(G_multi)
    print(f"PyG Data Object generated: {pyg_data}")