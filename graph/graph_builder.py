"""
graph_builder.py
Procedural Spatial Graph Generator for DRL under Canadian Traveller Problem (CTP).
"""

import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
import torch
from torch_geometric.data import Data

def sample_od_pairs(
    G,
    num_agents: int = 1,
    perim_radius: float = 0.35,
    min_od_dist: float = 0.70,
    min_agent_sep: float = 0.25,
    avoid_od_overlap: bool = False,
    max_retries: int = 50,
    rng=None,
) -> list[tuple[int, int]]:
    """
    Standalone randomized perimeter multi-agent Origin-Destination (OD) sampler.

    Samples K = num_agents OD pairs (O_k, D_k) from the outer perimeter of an
    already-generated spatial graph G = (V, E) by excluding the interior disk of
    radius `perim_radius` centered at (0.5, 0.5), enforcing minimum intra-pair
    Euclidean separation `min_od_dist`, minimum inter-agent separation `min_agent_sep`,
    and topological reachability in G.

    Updates G in-place for backward compatibility:
      - G.graph["od_pairs"] = [(O_1, D_1), ..., (O_K, D_K)]
      - G.graph["origin"] = O_1
      - G.graph["destination"] = D_1
      - Node attributes: 'is_origin', 'is_destination', 'origin_agent_idx', 'destination_agent_idx'
    """
    if num_agents < 1:
        raise ValueError("num_agents must be at least 1.")
    if G.number_of_nodes() < 2 * num_agents:
        raise ValueError(
            f"Graph has {G.number_of_nodes()} nodes, which is fewer than 2 * num_agents ({2 * num_agents})."
        )

    if rng is None:
        rng_gen = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng_gen = np.random.default_rng(int(rng))
    else:
        rng_gen = rng

    pos = nx.get_node_attributes(G, "pos")
    nodes_list = list(G.nodes())
    coords = {v: np.asarray(pos[v], dtype=float) for v in nodes_list}
    center = np.array([0.5, 0.5], dtype=float)
    dist_center = {v: float(np.linalg.norm(coords[v] - center)) for v in nodes_list}

    # 1. Perimeter Pool (Center-Exclusion Filter) with iterative safeguard
    curr_perim_radius = float(perim_radius)
    perimeter_nodes = [v for v in nodes_list if dist_center[v] >= curr_perim_radius]
    while len(perimeter_nodes) < 2 * num_agents and curr_perim_radius > 0.0:
        curr_perim_radius = max(0.0, round(curr_perim_radius - 0.02, 6))
        perimeter_nodes = [v for v in nodes_list if dist_center[v] >= curr_perim_radius]

    # 2. Iterative Multi-Agent Sampling (k = 1 ... K) with Backtracking & Relaxation
    curr_min_od_dist = float(min_od_dist)
    curr_min_agent_sep = float(min_agent_sep)
    od_pairs: list[tuple[int, int]] = []

    relaxation_round = 0
    while True:
        found_all = False
        for _ in range(max_retries):
            od_pairs = []
            used_origins: list[int] = []
            used_destinations: list[int] = []
            used_nodes: set[int] = set()
            success = True

            for _k in range(num_agents):
                ref_o_nodes = used_nodes if avoid_od_overlap else used_origins
                valid_O = [
                    v for v in perimeter_nodes
                    if v not in used_nodes
                    and all(
                        float(np.linalg.norm(coords[v] - coords[u])) >= curr_min_agent_sep
                        for u in ref_o_nodes
                    )
                ]
                if not valid_O:
                    success = False
                    break

                perm = rng_gen.permutation(len(valid_O))
                valid_O_shuffled = [valid_O[int(i)] for i in perm]

                paired_k = False
                for O_k in valid_O_shuffled:
                    ref_d_nodes = used_nodes if avoid_od_overlap else used_destinations
                    valid_D = [
                        v for v in perimeter_nodes
                        if v not in used_nodes
                        and v != O_k
                        and float(np.linalg.norm(coords[O_k] - coords[v])) >= curr_min_od_dist
                        and all(
                            float(np.linalg.norm(coords[v] - coords[d])) >= curr_min_agent_sep
                            for d in ref_d_nodes
                        )
                        and nx.has_path(G, O_k, v)
                    ]
                    if valid_D:
                        chosen_idx = int(rng_gen.choice(len(valid_D)))
                        D_k = valid_D[chosen_idx]
                        od_pairs.append((int(O_k), int(D_k)))
                        used_origins.append(int(O_k))
                        used_destinations.append(int(D_k))
                        used_nodes.add(int(O_k))
                        used_nodes.add(int(D_k))
                        paired_k = True
                        break

                if not paired_k:
                    success = False
                    break

            if success and len(od_pairs) == num_agents:
                found_all = True
                break

        if found_all:
            break

        # Decay min_od_dist and min_agent_sep by 5% and retry
        curr_min_od_dist *= 0.95
        curr_min_agent_sep *= 0.95
        relaxation_round += 1

        if relaxation_round % 5 == 0 and curr_perim_radius > 0.0:
            curr_perim_radius = max(0.0, round(curr_perim_radius - 0.02, 6))
            perimeter_nodes = [v for v in nodes_list if dist_center[v] >= curr_perim_radius]

        if relaxation_round > 200:
            raise RuntimeError(
                f"Failed to sample {num_agents} connected OD pairs after 200 relaxation stages."
            )

    # 3. Backward Compatibility & Node/Graph Metadata Update
    G.graph["od_pairs"] = od_pairs
    G.graph["origin"] = od_pairs[0][0]
    G.graph["destination"] = od_pairs[0][1]

    for v in G.nodes():
        G.nodes[v]["is_origin"] = False
        G.nodes[v]["is_destination"] = False
        G.nodes[v]["origin_agent_idx"] = None
        G.nodes[v]["destination_agent_idx"] = None

    for agent_idx, (o_node, d_node) in enumerate(od_pairs):
        G.nodes[o_node]["is_origin"] = True
        G.nodes[d_node]["is_destination"] = True
        G.nodes[o_node]["origin_agent_idx"] = agent_idx
        G.nodes[d_node]["destination_agent_idx"] = agent_idx

    return od_pairs


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
        Places all N nodes across [0, 1]^2 without locking any node to fixed
        origin/destination corner boxes. Preserves minimum node separation d_min
        and SBM/cluster structure.
        """
        pos = np.zeros((self.N, 2))
        cluster_ids = np.zeros(self.N, dtype=int)
        
        if regime in ['sparse', 'dense']:
            # Uniformly distribute all N nodes across the full [0, 1]^2 space
            pos[:] = self._sample_nodes(self.N, [0.0, 1.0, 0.0, 1.0])
            
        elif regime == 'multidistrict':
            # Geometric Block Model: Partition space into C well-separated centroids (Galhotra et al., 2018)
            centroid_d_min = min(0.45, 0.65 / np.sqrt(max(1, C)))
            centroids = self._sample_nodes(C, [0.12, 0.88, 0.12, 0.88], d_min=centroid_d_min)
            
            # Scale cluster radius with C so districts remain visually compact and distinct
            cluster_radius = min(0.14, 0.28 / np.sqrt(max(1, C)))
            
            base_count = self.N // C
            remainder = self.N % C
            
            placed_nodes = []
            idx = 0
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
                existing = np.array(placed_nodes) if placed_nodes else None
                sampled = self._sample_nodes(
                    n_cluster, bounds, existing_nodes=existing
                )
                pos[idx:idx+n_cluster] = sampled
                placed_nodes.extend(sampled.tolist())
                idx += n_cluster
                
        return pos, cluster_ids

    def _draw_edges(self, pos, cluster_ids, regime, params):
        """
        Calculates physical lengths and draws edges via Bernoulli trials based on Beta distributions.
        Constructs the complete road graph G_n = (V, E_n) prior to OD assignment.
        """
        dist_mat = distance_matrix(pos, pos)
        G = nx.Graph()
        
        for i in range(self.N):
            G.add_node(i, 
                       pos=(float(pos[i][0]), float(pos[i][1])), 
                       is_origin=False, 
                       is_destination=False,
                       origin_agent_idx=None,
                       destination_agent_idx=None,
                       cluster_id=int(cluster_ids[i]))
            
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

    def build_graph(
        self,
        regime='dense',
        C=3,
        num_agents: int = 1,
        perim_radius: float = 0.35,
        min_od_dist: float = 0.70,
        min_agent_sep: float = 0.25,
        avoid_od_overlap: bool = False,
        rng=None,
    ):
        """
        Full construction pipeline:
        1. Samples all N nodes and builds the complete connected road graph G_n = (V, E_n)
           before assigning any origins or destinations.
        2. Calls sample_od_pairs(G, ...) to assign perimeter OD pairs (default num_agents=1).
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
        
        # Resample edge draws until the road network G_n = (V, E_n) is connected
        attempts = 0
        placement_resets = 0
        while not connected:
            if attempts > 50:
                # Reset node placement if edge density is too low to connect
                pos, cluster_ids = self._generate_node_placements(regime, C)
                attempts = 0
                placement_resets += 1
                # Gradually expand cutoff radius if extreme parameters prevent global connectivity
                if placement_resets % 3 == 0:
                    if 'l_max' in params:
                        params['l_max'] = min(1.0, params['l_max'] * 1.15)
                    else:
                        params['r_d'] = min(1.0, params['r_d'] * 1.15)
                        params['r_s'] = min(1.0, params['r_s'] * 1.10)
                
            G = self._draw_edges(pos, cluster_ids, regime, params)
            connected = nx.is_connected(G)
            attempts += 1

        # Sample initial OD pairs on the already-constructed road graph G_n
        sample_od_pairs(
            G,
            num_agents=num_agents,
            perim_radius=perim_radius,
            min_od_dist=min_od_dist,
            min_agent_sep=min_agent_sep,
            avoid_od_overlap=avoid_od_overlap,
            rng=rng,
        )
            
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
# Execution Example & Multi-Agent OD Verification
# ==========================================
if __name__ == "__main__":
    def _print_od_metrics(G, od_pairs, label=""):
        pos = nx.get_node_attributes(G, "pos")
        center = np.array([0.5, 0.5])
        print(f"\n--- {label} (K={len(od_pairs)}) ---")
        print(f"G.graph['od_pairs'] = {G.graph.get('od_pairs')} | origin={G.graph.get('origin')}, destination={G.graph.get('destination')}")
        for k, (o, d) in enumerate(od_pairs, start=1):
            po, pd = np.asarray(pos[o]), np.asarray(pos[d])
            d_o_center = float(np.linalg.norm(po - center))
            d_d_center = float(np.linalg.norm(pd - center))
            d_od = float(np.linalg.norm(po - pd))
            has_p = nx.has_path(G, o, d)
            print(
                f"  Agent {k}: O{k}={o} ({po[0]:.2f}, {po[1]:.2f}) [dist_center={d_o_center:.3f}] -> "
                f"D{k}={d} ({pd[0]:.2f}, {pd[1]:.2f}) [dist_center={d_d_center:.3f}] | "
                f"dist(O{k}, D{k})={d_od:.3f} | connected={has_p}"
            )
        if len(od_pairs) > 1:
            o_seps = [
                float(np.linalg.norm(np.asarray(pos[od_pairs[i][0]]) - np.asarray(pos[od_pairs[j][0]])))
                for i in range(len(od_pairs)) for j in range(i + 1, len(od_pairs))
            ]
            d_seps = [
                float(np.linalg.norm(np.asarray(pos[od_pairs[i][1]]) - np.asarray(pos[od_pairs[j][1]])))
                for i in range(len(od_pairs)) for j in range(i + 1, len(od_pairs))
            ]
            print(f"  Min Inter-Origin Sep: {min(o_seps):.3f} | Min Inter-Destination Sep: {min(d_seps):.3f}")

    builder = SpatialGraphGenerator(n_nodes=80, d_min=0.035)

    # 1. Generate base networks across regimes
    G_sparse, _ = builder.build_graph(regime='sparse')
    G_dense, _ = builder.build_graph(regime='dense')
    G_multi, _ = builder.build_graph(regime='multidistrict', C=4)

    # 2. Sample fresh 1-agent OD pairs independently on the same base graph
    for trial in range(1, 4):
        pairs_1 = sample_od_pairs(G_multi, num_agents=1, rng=trial)
        _print_od_metrics(G_multi, pairs_1, label=f"1-Agent Resample Trial {trial}")

    # 3. Sample 2-agent and 3-agent perimeter OD pairs on the same base graph
    pairs_2 = sample_od_pairs(G_multi, num_agents=2, rng=102)
    _print_od_metrics(G_multi, pairs_2, label="2-Agent Perimeter OD Sampling")

    pairs_3 = sample_od_pairs(G_multi, num_agents=3, avoid_od_overlap=True, rng=203)
    _print_od_metrics(G_multi, pairs_3, label="3-Agent Perimeter OD Sampling (avoid_od_overlap=True)")

    # 4. Convert to DRL tensor format
    pyg_data = builder.to_pyg(G_multi)
    print(f"\nPyG Data Object generated: {pyg_data}")