"""
seal/model.py
Inductive SEAL (Subgraph, Embeddings, and Attributes for Link Prediction)
Spatial Edge Blockage Prediction Module in PyTorch & PyTorch Geometric.

Reference:
- Zhang, M., & Chen, Y. (2018). Link Prediction Based on Graph Neural Networks (SEAL).
  Advances in Neural Information Processing Systems (NeurIPS 2018).
"""

import os
import sys
import math
import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINEConv
from torch_geometric.nn.aggr import SortAggregation
from torch_geometric.utils import degree

# Status Constants
STATUS_CLEAR = 0
STATUS_BLOCKED = 1
STATUS_UNKNOWN = 2

MAX_DRNL_LABEL = 16
NUM_CLUSTER_BUCKETS = 8
IDW_SCALES = (0.06, 0.12, 0.22, 0.40)

# Node feature dimension:
#   DRNL one-hot (17) + DRNL scalar & flags (4) + coords & relative geometry (11)
#   + cluster features (11) + local observed degree & multi-scale IDW stats (14) = 57
NODE_FEATURE_DIM = (MAX_DRNL_LABEL + 1) + 4 + 11 + (3 + NUM_CLUSTER_BUCKETS) + (6 + 2 * len(IDW_SCALES))

# Edge feature dimension:
#   [length_raw, length_norm, speed_norm, travel_time_norm, is_highway, same_cluster] (6)
#   + one-hot status [Clear, Blocked, Unknown] (3)
#   + relative geometry to target edge [dx, dy, dist, cos_angle, is_target_edge] (5) = 14
EDGE_FEATURE_DIM = 14


def parse_edge_status(val) -> int:
    """
    Normalizes heterogeneous edge status representations into:
      0 -> STATUS_CLEAR (Observed Open / Traversable)
      1 -> STATUS_BLOCKED (Observed Blocked / Failed)
      2 -> STATUS_UNKNOWN (Unobserved)
    """
    if val is None:
        return STATUS_UNKNOWN
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("clear", "open", "passable", "traversable", "free", "0", "false"):
            return STATUS_CLEAR
        if s in ("blocked", "closed", "failed", "impassable", "1", "true"):
            return STATUS_BLOCKED
        return STATUS_UNKNOWN
    if isinstance(val, (list, tuple, np.ndarray, torch.Tensor)):
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 3 and np.sum(arr) > 0:
            return int(np.argmax(arr))
        if arr.size == 1:
            val = float(arr[0])
        else:
            return STATUS_UNKNOWN
    if isinstance(val, bool):
        return STATUS_BLOCKED if val else STATUS_CLEAR
    if isinstance(val, (int, np.integer, float, np.floating)):
        if np.isnan(val):
            return STATUS_UNKNOWN
        ival = int(round(float(val)))
        if ival == 0:
            return STATUS_CLEAR
        if ival == 1:
            return STATUS_BLOCKED
        return STATUS_UNKNOWN
    return STATUS_UNKNOWN


def normalize_edge_status_dict(
    nx_graph: Union[nx.Graph, nx.DiGraph],
    edge_status_dict: Optional[Dict[Tuple[int, int], Union[int, str, float, Sequence[float]]]],
) -> Dict[Tuple[int, int], int]:
    """
    Builds a canonical lookup mapping every undirected edge (min(u,v), max(u,v))
    and directed pair (u, v), (v, u) in `nx_graph` to {STATUS_CLEAR, STATUS_BLOCKED, STATUS_UNKNOWN}.
    """
    normalized: Dict[Tuple[int, int], int] = {}
    if edge_status_dict:
        for k, v in edge_status_dict.items():
            if not isinstance(k, tuple) or len(k) != 2:
                continue
            u, w = k
            st = parse_edge_status(v)
            normalized[(u, w)] = st
            if (w, u) not in normalized or normalized[(w, u)] == STATUS_UNKNOWN:
                normalized[(w, u)] = st

    # Populate missing graph edges as STATUS_UNKNOWN
    for u, v in nx_graph.edges():
        if (u, v) not in normalized:
            st = normalized.get((v, u), STATUS_UNKNOWN)
            normalized[(u, v)] = st
        if (v, u) not in normalized:
            normalized[(v, u)] = normalized[(u, v)]

    return normalized


def prepare_spatial_graph(
    nx_graph: Union[nx.Graph, nx.DiGraph],
    seed: int = 42,
    copy_graph: bool = False,
) -> nx.Graph:
    """
    Ensures an arbitrary procedural or canonical NetworkX graph (Graph or DiGraph)
    has standardized spatial node and edge attributes required for SEAL & GP simulation:
      - Node attributes:
          'pos': (x, y) normalized to [0.05, 0.95]^2 (preserved if already in [0, 1]^2)
          'cluster_id': int >= 0
          'is_origin': bool
          'is_destination': bool
      - Edge attributes:
          'length': Euclidean length w_e > 0
          'travel_time': Travel cost c_e > 0
          'speed': Effective traversal speed v_e = w_e / c_e
          'is_highway': 0 or 1
          'midpoint': ((x_u + x_v)/2, (y_u + y_v)/2)

    Converts DiGraphs into an equivalent undirected spatial `nx.Graph` for topological
    subgraph extraction while retaining fast lookup for directed edge queries.
    """
    if isinstance(nx_graph, nx.DiGraph):
        G = nx.Graph()
        for n, d in nx_graph.nodes(data=True):
            G.add_node(n, **dict(d))
        for u, v, d in nx_graph.edges(data=True):
            if G.has_edge(u, v):
                # Merge attributes conservatively
                existing = G.edges[u, v]
                for k, val in d.items():
                    if k not in existing:
                        existing[k] = val
            else:
                G.add_edge(u, v, **dict(d))
        G.graph.update(nx_graph.graph)
    else:
        G = nx_graph.copy() if copy_graph else nx_graph

    nodes = list(G.nodes())
    n_nodes = len(nodes)
    if n_nodes == 0:
        return G

    # 1. Ensure 2D spatial coordinates 'pos' in [0, 1]^2
    pos = nx.get_node_attributes(G, "pos")
    if len(pos) < n_nodes:
        # Deterministic spectral + spring layout using edge cost/weight
        try:
            if n_nodes <= 600:
                raw_pos = nx.spectral_layout(G, weight=None)
                raw_pos = nx.spring_layout(
                    G,
                    pos=raw_pos,
                    seed=seed,
                    iterations=60,
                    k=1.2 / math.sqrt(max(1, n_nodes)),
                )
            else:
                raw_pos = nx.spring_layout(
                    G,
                    seed=seed,
                    iterations=40,
                    k=1.2 / math.sqrt(max(1, n_nodes)),
                )
        except Exception:
            rng = np.random.default_rng(seed)
            raw_pos = {n: rng.uniform(0.05, 0.95, size=2) for n in nodes}

        coords = np.vstack([np.asarray(raw_pos[n], dtype=float) for n in nodes])
        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        span = np.where((c_max - c_min) > 1e-8, c_max - c_min, 1.0)
        norm_coords = 0.06 + 0.88 * (coords - c_min) / span
        for idx, n in enumerate(nodes):
            G.nodes[n]["pos"] = (float(norm_coords[idx, 0]), float(norm_coords[idx, 1]))
        pos = nx.get_node_attributes(G, "pos")
    else:
        # Ensure all pos tuples are float pairs normalized inside [0, 1]^2 if outside bounds
        coords = np.vstack([np.asarray(pos[n], dtype=float) for n in nodes])
        if np.min(coords) < -1e-3 or np.max(coords) > 1.0 + 1e-3:
            c_min = coords.min(axis=0)
            c_max = coords.max(axis=0)
            span = np.where((c_max - c_min) > 1e-8, c_max - c_min, 1.0)
            norm_coords = 0.05 + 0.90 * (coords - c_min) / span
            for idx, n in enumerate(nodes):
                G.nodes[n]["pos"] = (float(norm_coords[idx, 0]), float(norm_coords[idx, 1]))
            pos = nx.get_node_attributes(G, "pos")

    # 2. Ensure Origin / Destination flags
    has_origin = any(G.nodes[n].get("is_origin", False) for n in nodes)
    has_dest = any(G.nodes[n].get("is_destination", False) for n in nodes)
    if not has_origin or not has_dest:
        coords = np.vstack([np.asarray(pos[n], dtype=float) for n in nodes])
        dists_bl = np.linalg.norm(coords - np.array([0.0, 0.0]), axis=1)
        dists_tr = np.linalg.norm(coords - np.array([1.0, 1.0]), axis=1)
        o_idx = int(np.argmin(dists_bl))
        d_idx = int(np.argmin(dists_tr))
        if d_idx == o_idx and n_nodes > 1:
            d_idx = (o_idx + 1) % n_nodes
        for idx, n in enumerate(nodes):
            if not has_origin:
                G.nodes[n]["is_origin"] = (idx == o_idx)
            if not has_dest:
                G.nodes[n]["is_destination"] = (idx == d_idx)

    # 3. Ensure 'cluster_id' (if missing or all 0, assign spatial grid/sector cluster IDs 1..4)
    cluster_ids = [G.nodes[n].get("cluster_id", None) for n in nodes]
    if any(c is None for c in cluster_ids):
        for n in nodes:
            if "cluster_id" not in G.nodes[n]:
                x_n, y_n = pos[n]
                qx = 0 if x_n < 0.5 else 1
                qy = 0 if y_n < 0.5 else 1
                G.nodes[n]["cluster_id"] = int(1 + qx + 2 * qy)

    # 4. Ensure edge attributes: length (w_e), travel_time (c_e), speed (v_e), is_highway, midpoint
    raw_costs = []
    for u, v, d in G.edges(data=True):
        c = d.get("travel_time", d.get("weight", d.get("Cost", None)))
        if c is not None:
            raw_costs.append(float(c))
    median_cost = float(np.median(raw_costs)) if raw_costs else 1.0
    cost_scale = median_cost if median_cost > 1e-6 else 1.0

    for u, v, d in G.edges(data=True):
        xu, yu = pos[u]
        xv, yv = pos[v]
        euclid_len = float(math.hypot(xu - xv, yu - yv))
        w_e = float(d.get("length", max(1e-4, euclid_len)))
        d["length"] = max(1e-4, w_e)
        d["midpoint"] = (0.5 * (xu + xv), 0.5 * (yu + yv))

        if "travel_time" in d and d["travel_time"] is not None:
            c_e = float(d["travel_time"])
        elif "weight" in d and d["weight"] is not None:
            # Normalize large raw road costs so speed v_e = w_e / c_e stays well-scaled
            c_e = float(d["weight"]) / cost_scale * d["length"]
            d["travel_time"] = max(1e-4, c_e)
        elif "Cost" in d and d["Cost"] is not None:
            c_e = float(d["Cost"]) / cost_scale * d["length"]
            d["travel_time"] = max(1e-4, c_e)
        else:
            c_e = d["length"]
            d["travel_time"] = max(1e-4, c_e)

        c_e = max(1e-5, float(d["travel_time"]))
        v_e = float(d.get("speed", d["length"] / c_e))
        d["speed"] = float(np.clip(v_e, 0.05, 20.0))

        if "is_highway" not in d:
            cu = G.nodes[u].get("cluster_id", 0)
            cv = G.nodes[v].get("cluster_id", 0)
            d["is_highway"] = 1 if (cu != cv or v_e > 1.5) else 0

    return G


def load_canonical_network(
    network_name: str,
    workspace_root: Optional[str] = None,
    seed: int = 42,
) -> nx.Graph:
    """
    Loads a canonical benchmark network (e.g. 'SiouxFalls', 'Anaheim',
    'Friedrichshain', 'Winnipeg', 'Barcelona', 'Chicago_Sketch') from
    `marvel_modernized/Networks/` or `sevac/data/` and prepares its spatial
    node/edge attributes for SEAL and GP disaster simulation.
    """
    if workspace_root is None:
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    clean_name = network_name.replace(" ", "").strip()
    name_variants = [clean_name, network_name.strip()]
    if clean_name.lower() == "siouxfalls":
        name_variants = ["SiouxFalls", "sioux"]

    search_paths = []
    for nv in name_variants:
        search_paths.extend([
            os.path.join(workspace_root, "marvel_modernized", "Networks", nv, f"{nv}_network.csv"),
            os.path.join(workspace_root, "sevac", "data", nv, f"{nv}_network.csv"),
            os.path.join(workspace_root, "Networks", nv, f"{nv}_network.csv"),
        ])

    csv_path = None
    for p in search_paths:
        if os.path.exists(p):
            csv_path = p
            break

    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find canonical network '{network_name}'. Checked: {search_paths[:4]}"
        )

    df = pd.read_csv(csv_path)
    G = nx.Graph()
    for _, row in df.iterrows():
        u = int(row["From"])
        v = int(row["To"])
        if u == v:
            continue
        cost = float(row["Cost"]) if "Cost" in row and pd.notna(row["Cost"]) else 1.0
        vol = float(row["Volume"]) if "Volume" in row and pd.notna(row["Volume"]) else 0.0
        if G.has_edge(u, v):
            G.edges[u, v]["weight"] = 0.5 * (G.edges[u, v]["weight"] + cost)
        else:
            G.add_edge(u, v, weight=cost, volume=vol)

    # Keep largest connected component if isolated artifacts exist
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    G = prepare_spatial_graph(G, seed=seed, copy_graph=False)
    G.graph["name"] = clean_name
    G.graph["source_csv"] = csv_path
    return G


def compute_drnl(
    adj_list: Dict[int, List[int]],
    sub_nodes: Sequence[int],
    u: int,
    v: int,
    max_hop: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Double-Radius Node Labeling (DRNL) as defined in Zhang & Chen (NeurIPS 2018)
    for all nodes in `sub_nodes` with respect to target link endpoints (u, v).
    Crucially, the direct target edge (u, v) is excluded during BFS shortest-path
    calculation so the target edge never leaks its own topological presence.

    Formula:
      - f_l(u) = 1, f_l(v) = 1
      - For i != u, v with distances d_u = d(i, u), d_v = d(i, v) and d = d_u + d_v:
          f_l(i) = 1 + min(d_u, d_v) + (d // 2) * ((d // 2) + (d % 2) - 1)
      - Unreachable nodes (d_u = inf or d_v = inf) receive label 0.

    Returns:
      labels (np.ndarray): Integer DRNL labels in [0, MAX_DRNL_LABEL] of shape (|V_sub|,).
      dist_u (np.ndarray): Hop distances to u (capped at max_hop + 2) of shape (|V_sub|,).
      dist_v (np.ndarray): Hop distances to v (capped at max_hop + 2) of shape (|V_sub|,).
    """
    sub_set = set(sub_nodes)
    inf_dist = max_hop + 2

    def _bfs_excluding_target(src: int, other: int) -> Dict[int, int]:
        dists = {src: 0}
        queue = deque([src])
        while queue:
            curr = queue.popleft()
            d_curr = dists[curr]
            if d_curr >= max_hop + 1:
                continue
            for nbr in adj_list.get(curr, ()):
                if nbr not in sub_set:
                    continue
                if (curr == u and nbr == v) or (curr == v and nbr == u):
                    continue
                if nbr not in dists:
                    dists[nbr] = d_curr + 1
                    queue.append(nbr)
        return dists

    du_map = _bfs_excluding_target(u, v)
    dv_map = _bfs_excluding_target(v, u)

    n_sub = len(sub_nodes)
    labels = np.zeros(n_sub, dtype=np.int64)
    dist_u = np.full(n_sub, inf_dist, dtype=np.float32)
    dist_v = np.full(n_sub, inf_dist, dtype=np.float32)

    for idx, node in enumerate(sub_nodes):
        if node == u:
            labels[idx] = 1
            dist_u[idx] = 0.0
            dist_v[idx] = float(dv_map.get(u, 1))
        elif node == v:
            labels[idx] = 1
            dist_u[idx] = float(du_map.get(v, 1))
            dist_v[idx] = 0.0
        else:
            du = du_map.get(node, None)
            dv = dv_map.get(node, None)
            if du is not None:
                dist_u[idx] = float(du)
            if dv is not None:
                dist_v[idx] = float(dv)

            if du is None or dv is None:
                labels[idx] = 0
            else:
                d = du + dv
                half_d = d // 2
                rem_d = d % 2
                lbl = 1 + min(du, dv) + half_d * (half_d + rem_d - 1)
                labels[idx] = min(MAX_DRNL_LABEL, max(2, int(lbl)))

    return labels, dist_u, dist_v


class _GraphExtractionCache:
    """
    Precomputes arrays and adjacency lists for a NetworkX graph so extracting
    hundreds of k-hop enclosing subgraphs takes microseconds per edge.
    """
    def __init__(self, G: nx.Graph, normalized_status: Dict[Tuple[int, int], int]):
        self.G = G
        self.nodes = list(G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.adj_list: Dict[int, List[int]] = {n: list(G.neighbors(n)) for n in self.nodes}

        # Node arrays
        self.pos_map: Dict[int, Tuple[float, float]] = {}
        self.cluster_map: Dict[int, int] = {}
        for n in self.nodes:
            nd = G.nodes[n]
            p = nd.get("pos", (0.5, 0.5))
            self.pos_map[n] = (float(p[0]), float(p[1]))
            self.cluster_map[n] = int(nd.get("cluster_id", 0))

        # Edge attribute cache keyed by canonical (min_node, max_node) or (u, v)
        self.edge_info: Dict[Tuple[int, int], Tuple[float, float, float, float, int, float, float, int]] = {}
        lengths = [float(d.get("length", 0.1)) for _, _, d in G.edges(data=True)]
        times = [float(d.get("travel_time", 0.1)) for _, _, d in G.edges(data=True)]
        speeds = [float(d.get("speed", 1.0)) for _, _, d in G.edges(data=True)]

        self.max_len = max(1e-4, float(np.percentile(lengths, 95)) if lengths else 1.0)
        self.max_time = max(1e-4, float(np.percentile(times, 95)) if times else 1.0)
        self.max_speed = max(1e-4, float(np.max(speeds)) if speeds else 2.5)

        # Precompute per-node incident observed status counts (excluding any specific query edge later)
        self.node_deg: Dict[int, int] = {n: len(self.adj_list[n]) for n in self.nodes}
        self.node_obs_clear: Dict[int, int] = {n: 0 for n in self.nodes}
        self.node_obs_blocked: Dict[int, int] = {n: 0 for n in self.nodes}

        # Collect all observed edges and their midpoints for fast spatial IDW context
        obs_midpoints = []
        obs_is_blocked = []
        obs_edge_pairs = []

        for u, v, d in G.edges(data=True):
            xu, yu = self.pos_map[u]
            xv, yv = self.pos_map[v]
            mx, my = 0.5 * (xu + xv), 0.5 * (yu + yv)
            w_e = float(d.get("length", math.hypot(xu - xv, yu - yv)))
            c_e = float(d.get("travel_time", w_e))
            v_e = float(d.get("speed", w_e / max(1e-5, c_e)))
            hw = int(d.get("is_highway", 0))
            st = normalized_status.get((u, v), normalized_status.get((v, u), STATUS_UNKNOWN))

            info = (
                w_e,
                min(2.0, w_e / self.max_len),
                min(2.0, v_e / self.max_speed),
                min(2.0, c_e / self.max_time),
                hw,
                mx,
                my,
                st,
            )
            self.edge_info[(u, v)] = info
            self.edge_info[(v, u)] = info

            if st == STATUS_CLEAR:
                self.node_obs_clear[u] += 1
                self.node_obs_clear[v] += 1
                obs_midpoints.append((mx, my))
                obs_is_blocked.append(0.0)
                obs_edge_pairs.append((u, v))
            elif st == STATUS_BLOCKED:
                self.node_obs_blocked[u] += 1
                self.node_obs_blocked[v] += 1
                obs_midpoints.append((mx, my))
                obs_is_blocked.append(1.0)
                obs_edge_pairs.append((u, v))

        if obs_midpoints:
            self.obs_midpoints = np.asarray(obs_midpoints, dtype=np.float32)
            self.obs_is_blocked = np.asarray(obs_is_blocked, dtype=np.float32)
            self.obs_edge_pairs = obs_edge_pairs
        else:
            self.obs_midpoints = np.zeros((0, 2), dtype=np.float32)
            self.obs_is_blocked = np.zeros((0,), dtype=np.float32)
            self.obs_edge_pairs = []


def extract_enclosing_subgraph(
    cache: _GraphExtractionCache,
    target_edge: Tuple[int, int],
    k_hop: int = 2,
    y_prob: Optional[float] = None,
    y_bin: Optional[float] = None,
    max_sub_nodes: int = 80,
) -> Data:
    """
    Extracts the k-hop enclosing subgraph G_{(u, v)}^k around `target_edge = (u, v)`
    and constructs the inductive node feature matrix X and edge feature matrix E.
    """
    u, v = target_edge

    # 1. k-hop BFS expansion from {u, v}
    visited = {u: 0, v: 0}
    queue = deque([(u, 0), (v, 0)])
    sub_nodes = [u, v] if u != v else [u]

    while queue:
        curr, depth = queue.popleft()
        if depth >= k_hop:
            continue
        for nbr in cache.adj_list.get(curr, ()):
            if nbr not in visited:
                visited[nbr] = depth + 1
                sub_nodes.append(nbr)
                if len(sub_nodes) >= max_sub_nodes:
                    break
                queue.append((nbr, depth + 1))
        if len(sub_nodes) >= max_sub_nodes:
            break

    n_sub = len(sub_nodes)
    local_idx = {node: idx for idx, node in enumerate(sub_nodes)}

    # 2. Double-Radius Node Labeling (DRNL) with target edge (u, v) masked
    drnl_labels, dist_u, dist_v = compute_drnl(cache.adj_list, sub_nodes, u, v, max_hop=k_hop)

    # 3. Target edge geometry & cluster context
    xu, yu = cache.pos_map[u]
    xv, yv = cache.pos_map[v]
    mx_t, my_t = 0.5 * (xu + xv), 0.5 * (yu + yv)
    dx_t, dy_t = xv - xu, yv - yu
    len_t = max(1e-5, math.hypot(dx_t, dy_t))
    ux_axis, uy_axis = dx_t / len_t, dy_t / len_t  # Unit tangent along u -> v
    nx_axis, ny_axis = -uy_axis, ux_axis            # Unit normal perpendicular to u -> v

    cu = cache.cluster_map[u]
    cv = cache.cluster_map[v]

    # Check if target edge itself was in the observed list (if caller passed a known edge during training/eval)
    t_info = cache.edge_info.get((u, v), (len_t, min(2.0, len_t / cache.max_len), 1.0, min(2.0, len_t / cache.max_time), 0, mx_t, my_t, STATUS_UNKNOWN))
    t_orig_status = t_info[7]

    # 4. Build Node Feature Matrix X of shape (n_sub, NODE_FEATURE_DIM)
    node_coords = np.zeros((n_sub, 2), dtype=np.float32)
    for idx, node in enumerate(sub_nodes):
        node_coords[idx, 0], node_coords[idx, 1] = cache.pos_map[node]

    # Multi-scale spatial Gaussian/IDW kernel features from observed non-target edges
    # Strictly masks out target edge (u, v) if it happened to be observed
    idw_feats = np.zeros((n_sub, 2 * len(IDW_SCALES) + 2), dtype=np.float32)
    if len(cache.obs_midpoints) > 0:
        valid_mask = np.ones(len(cache.obs_midpoints), dtype=bool)
        if t_orig_status != STATUS_UNKNOWN:
            for obs_idx, (ou, ov) in enumerate(cache.obs_edge_pairs):
                if (ou == u and ov == v) or (ou == v and ov == u):
                    valid_mask[obs_idx] = False
        obs_pts = cache.obs_midpoints[valid_mask]
        obs_blk = cache.obs_is_blocked[valid_mask]

        if len(obs_pts) > 0:
            # Squared Euclidean distances from all subgraph nodes to observed edge midpoints: (n_sub, n_obs)
            diff_x = node_coords[:, 0:1] - obs_pts[:, 0:1].T
            diff_y = node_coords[:, 1:2] - obs_pts[:, 1:2].T
            sq_dists = diff_x * diff_x + diff_y * diff_y
            min_dists = np.sqrt(np.min(sq_dists, axis=1))
            nearest_idx = np.argmin(sq_dists, axis=1)

            for s_idx, scale in enumerate(IDW_SCALES):
                weights = np.exp(-0.5 * sq_dists / (scale * scale))  # (n_sub, n_obs)
                w_sum = np.sum(weights, axis=1)
                blk_weighted = np.sum(weights * obs_blk[None, :], axis=1)
                # Posterior-like smoothed blockage ratio and observation density at length scale `scale`
                idw_feats[:, 2 * s_idx] = np.where(w_sum > 1e-6, blk_weighted / (w_sum + 1e-4), 0.25)
                idw_feats[:, 2 * s_idx + 1] = np.tanh(w_sum / 4.0)

            idw_feats[:, -2] = np.exp(-min_dists / 0.15)
            idw_feats[:, -1] = obs_blk[nearest_idx]
        else:
            idw_feats[:, 0::2] = 0.25
    else:
        idw_feats[:, 0::2] = 0.25

    x_mat = np.zeros((n_sub, NODE_FEATURE_DIM), dtype=np.float32)
    denom_hop = float(k_hop + 1)

    for idx, node in enumerate(sub_nodes):
        lbl = int(drnl_labels[idx])
        # (a) DRNL One-hot [0 .. MAX_DRNL_LABEL] (17 dims)
        x_mat[idx, lbl] = 1.0

        # (b) DRNL scalar & hop distances (4 dims: offset 17..21)
        off = MAX_DRNL_LABEL + 1
        x_mat[idx, off + 0] = float(lbl) / float(MAX_DRNL_LABEL)
        x_mat[idx, off + 1] = float(dist_u[idx]) / denom_hop
        x_mat[idx, off + 2] = float(dist_v[idx]) / denom_hop
        x_mat[idx, off + 3] = 1.0 if (node == u or node == v) else 0.0
        off += 4

        # (c) 2D coordinates & relative geometry to target edge (11 dims: offset 21..32)
        xn, yn = node_coords[idx, 0], node_coords[idx, 1]
        rx, ry = xn - mx_t, yn - my_t
        r_mid = math.hypot(rx, ry)
        r_u = math.hypot(xn - xu, yn - yu)
        r_v = math.hypot(xn - xv, yn - yv)
        proj_para = rx * ux_axis + ry * uy_axis
        proj_perp = abs(rx * nx_axis + ry * ny_axis)

        x_mat[idx, off + 0] = xn
        x_mat[idx, off + 1] = yn
        x_mat[idx, off + 2] = rx
        x_mat[idx, off + 3] = ry
        x_mat[idx, off + 4] = r_mid
        x_mat[idx, off + 5] = r_u
        x_mat[idx, off + 6] = r_v
        x_mat[idx, off + 7] = proj_para
        x_mat[idx, off + 8] = proj_perp
        x_mat[idx, off + 9] = math.exp(-0.5 * (r_mid / 0.12) ** 2)
        x_mat[idx, off + 10] = math.exp(-0.5 * (r_mid / 0.25) ** 2)
        off += 11

        # (d) District / Cluster features (11 dims: offset 32..43)
        c_node = cache.cluster_map[node]
        x_mat[idx, off + 0] = 1.0 if c_node == cu else 0.0
        x_mat[idx, off + 1] = 1.0 if c_node == cv else 0.0
        x_mat[idx, off + 2] = float(c_node) / 10.0
        bucket = int(c_node) % NUM_CLUSTER_BUCKETS
        x_mat[idx, off + 3 + bucket] = 1.0
        off += 3 + NUM_CLUSTER_BUCKETS

        # (e) Local incident observed status ratios & multi-scale IDW features (14 dims: offset 43..57)
        deg = max(1, cache.node_deg[node])
        n_clr = cache.node_obs_clear[node]
        n_blk = cache.node_obs_blocked[node]
        # Subtract target edge if it was counted in cache
        if node in (u, v):
            if t_orig_status == STATUS_CLEAR:
                n_clr = max(0, n_clr - 1)
            elif t_orig_status == STATUS_BLOCKED:
                n_blk = max(0, n_blk - 1)
        n_obs = n_clr + n_blk
        n_unk = max(0, deg - n_obs)

        x_mat[idx, off + 0] = float(deg) / 10.0
        x_mat[idx, off + 1] = float(n_clr) / float(deg)
        x_mat[idx, off + 2] = float(n_blk) / float(deg)
        x_mat[idx, off + 3] = float(n_unk) / float(deg)
        x_mat[idx, off + 4] = float(n_blk) / float(max(1, n_obs)) if n_obs > 0 else 0.25
        x_mat[idx, off + 5] = 1.0 if n_obs > 0 else 0.0
        x_mat[idx, off + 6 : off + 6 + idw_feats.shape[1]] = idw_feats[idx]

    # 5. Build Subgraph Edge Index & Edge Attributes (both directions for undirected message passing)
    edge_src = []
    edge_dst = []
    edge_attr_list = []

    sub_set = set(sub_nodes)
    seen_pairs = set()

    for node_a in sub_nodes:
        ia = local_idx[node_a]
        for node_b in cache.adj_list.get(node_a, ()):
            if node_b not in sub_set:
                continue
            pair_key = (min(node_a, node_b), max(node_a, node_b))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            ib = local_idx[node_b]

            is_target = (node_a == u and node_b == v) or (node_a == v and node_b == u)
            info = cache.edge_info.get((node_a, node_b))
            if info is None:
                continue

            w_e, w_norm, v_norm, c_norm, hw, mx_e, my_e, st = info
            # Target edge is strictly masked as STATUS_UNKNOWN to prevent label leakage
            eff_st = STATUS_UNKNOWN if is_target else st

            oh_clear = 1.0 if eff_st == STATUS_CLEAR else 0.0
            oh_blocked = 1.0 if eff_st == STATUS_BLOCKED else 0.0
            oh_unknown = 1.0 if eff_st == STATUS_UNKNOWN else 0.0

            same_clust = 1.0 if cache.cluster_map[node_a] == cache.cluster_map[node_b] else 0.0
            dmx, dmy = mx_e - mx_t, my_e - my_t
            dist_mid = math.hypot(dmx, dmy)

            xa, ya = cache.pos_map[node_a]
            xb, yb = cache.pos_map[node_b]
            elen = max(1e-5, math.hypot(xb - xa, yb - ya))
            cos_ang = abs(((xb - xa) * ux_axis + (yb - ya) * uy_axis) / elen)

            e_vec = [
                w_e,
                w_norm,
                v_norm,
                c_norm,
                float(hw),
                same_clust,
                oh_clear,
                oh_blocked,
                oh_unknown,
                dmx,
                dmy,
                dist_mid,
                cos_ang,
                1.0 if is_target else 0.0,
            ]

            edge_src.extend([ia, ib])
            edge_dst.extend([ib, ia])
            edge_attr_list.append(e_vec)
            edge_attr_list.append(e_vec)

    # Ensure target edge (0, 1) is present in the subgraph even if querying a virtual pair
    if (min(u, v), max(u, v)) not in seen_pairs and u != v:
        ia, ib = local_idx[u], local_idx[v]
        w_e, w_norm, v_norm, c_norm, hw, mx_e, my_t_val, _ = t_info
        same_clust = 1.0 if cu == cv else 0.0
        e_vec = [
            w_e, w_norm, v_norm, c_norm, float(hw), same_clust,
            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 1.0, 1.0,
        ]
        edge_src.extend([ia, ib])
        edge_dst.extend([ib, ia])
        edge_attr_list.append(e_vec)
        edge_attr_list.append(e_vec)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
    x_tensor = torch.from_numpy(x_mat)

    # Target edge global context vector (12 dims) for skip readout
    target_edge_feat = torch.tensor([
        t_info[0],
        t_info[1],
        t_info[2],
        t_info[3],
        float(t_info[4]),
        1.0 if cu == cv else 0.0,
        mx_t,
        my_t,
        0.5 * (idw_feats[0, 0] + idw_feats[min(1, n_sub - 1), 0]),
        0.5 * (idw_feats[0, 2] + idw_feats[min(1, n_sub - 1), 2]),
        0.5 * (idw_feats[0, 4] + idw_feats[min(1, n_sub - 1), 4]),
        0.5 * (idw_feats[0, -2] + idw_feats[min(1, n_sub - 1), -2]),
    ], dtype=torch.float32).unsqueeze(0)

    # Distance in hops to nearest observed edge (for diagnostic hop breakdown h=1 vs h=2)
    u_has_obs = (x_mat[0, MAX_DRNL_LABEL + 1 + 4 + 11 + 3 + NUM_CLUSTER_BUCKETS + 5] > 0.5)
    v_has_obs = (x_mat[min(1, n_sub - 1), MAX_DRNL_LABEL + 1 + 4 + 11 + 3 + NUM_CLUSTER_BUCKETS + 5] > 0.5)
    nearest_obs_hop = 1 if (u_has_obs or v_has_obs) else 2

    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        target_edge_feat=target_edge_feat,
        drnl=torch.from_numpy(drnl_labels),
        nearest_obs_hop=torch.tensor([nearest_obs_hop], dtype=torch.long),
    )
    if y_prob is not None:
        data.y_prob = torch.tensor([float(y_prob)], dtype=torch.float32)
    if y_bin is not None:
        data.y_bin = torch.tensor([float(y_bin)], dtype=torch.float32)

    return data


def extract_batch_subgraphs(
    nx_graph: Union[nx.Graph, nx.DiGraph],
    edge_status_dict: Optional[Dict[Tuple[int, int], Union[int, str, float]]],
    target_edges: Sequence[Tuple[int, int]],
    k_hop: int = 2,
    gp_probs: Optional[Dict[Tuple[int, int], float]] = None,
    bin_labels: Optional[Dict[Tuple[int, int], int]] = None,
) -> List[Data]:
    """
    Vectorized helper that prepares the graph cache once and extracts enclosing
    subgraphs for all `target_edges`.
    """
    G = prepare_spatial_graph(nx_graph, copy_graph=False)
    norm_status = normalize_edge_status_dict(G, edge_status_dict)
    cache = _GraphExtractionCache(G, norm_status)

    data_list: List[Data] = []
    for e in target_edges:
        u, v = e
        y_p = None
        if gp_probs is not None:
            y_p = gp_probs.get((u, v), gp_probs.get((v, u), None))
        y_b = None
        if bin_labels is not None:
            y_b = bin_labels.get((u, v), bin_labels.get((v, u), None))

        sub_data = extract_enclosing_subgraph(
            cache=cache,
            target_edge=(u, v),
            k_hop=k_hop,
            y_prob=y_p,
            y_bin=y_b,
        )
        data_list.append(sub_data)

    return data_list


class SEALModel(nn.Module):
    """
    Inductive SEAL Architecture for Spatial Edge Blockage Prediction:
      1. Node & Edge Linear Encoders
      2. L layers of Edge-Aware GINEConv Message Passing with Jumping Knowledge
      3. PyG SortAggregation (SortPooling) + 1D CNN Feature Extractor (DGCNN style)
      4. Target Endpoint Pairwise Readout + MLP + Sigmoid Head -> q_hat_e in [0, 1]
    """
    def __init__(
        self,
        node_in_dim: int = NODE_FEATURE_DIM,
        edge_in_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 3,
        sortpool_k: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sortpool_k = sortpool_k
        self.dropout = dropout

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, train_eps=True, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        total_gnn_dim = hidden_dim * num_layers

        # SortPooling (Zhang et al., AAAI 2018 / NeurIPS 2018 SEAL)
        self.sort_pool = SortAggregation(k=sortpool_k)
        self.conv1d_1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=total_gnn_dim,
            stride=total_gnn_dim,
        )
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        conv1d_out_len = max(1, sortpool_k // 2)
        sortpool_flat_dim = 32 * conv1d_out_len

        # Target endpoint readout + SortPooling + target edge spatial context
        # Endpoint features: (h_u + h_v) [total_gnn_dim] + |h_u - h_v| [total_gnn_dim] + raw endpoint x [2 * node_in_dim]
        readout_dim = sortpool_flat_dim + 2 * total_gnn_dim + 2 * node_in_dim + 12

        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_logits(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Computes raw blockage logits for a PyG `Data` or `Batch` of enclosing subgraphs.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        ptr = data.ptr if hasattr(data, "ptr") and data.ptr is not None else torch.tensor([0, x.size(0)], dtype=torch.long, device=x.device)

        h = self.node_encoder(x)
        e_emb = self.edge_encoder(edge_attr)

        layer_outputs = []
        for conv, norm in zip(self.convs, self.norms):
            h_next = conv(h, edge_index, e_emb)
            h = norm(h + F.silu(h_next))
            layer_outputs.append(h)

        # Jumping Knowledge concatenation across all GINEConv layers: (TotalNodes, L * hidden_dim)
        h_cat = torch.cat(layer_outputs, dim=-1)

        # Add small deterministic structural tie-breaker to last channel for stable SortAggregation ordering
        if hasattr(data, "drnl") and data.drnl is not None:
            drnl_bonus = -0.01 * data.drnl.float().unsqueeze(-1)
            h_for_sort = torch.cat([h_cat[:, :-1], h_cat[:, -1:] + drnl_bonus], dim=-1)
        else:
            h_for_sort = h_cat

        # 1. SortPooling + 1D CNN
        pooled = self.sort_pool(h_for_sort, batch)  # (B, sortpool_k * total_gnn_dim)
        bsz = pooled.size(0)
        pooled_1d = pooled.view(bsz, 1, self.sortpool_k * h_cat.size(-1))
        c1 = F.silu(self.conv1d_1(pooled_1d))       # (B, 16, sortpool_k)
        c1 = self.maxpool1d(c1)                     # (B, 16, sortpool_k // 2)
        c2 = F.silu(self.conv1d_2(c1))              # (B, 32, sortpool_k // 2)
        sortpool_vec = c2.reshape(bsz, -1)

        # 2. Target Link Endpoint Embeddings (indices ptr[:-1] and ptr[:-1] + 1 are always u and v)
        u_idx = ptr[:-1]
        sub_sizes = ptr[1:] - ptr[:-1]
        v_idx = torch.where(sub_sizes > 1, u_idx + 1, u_idx)

        h_u = h_cat[u_idx]
        h_v = h_cat[v_idx]
        endpoint_sum = h_u + h_v
        endpoint_diff = torch.abs(h_u - h_v)

        x_u = x[u_idx]
        x_v = x[v_idx]
        raw_pair = torch.cat([0.5 * (x_u + x_v), torch.abs(x_u - x_v)], dim=-1)

        target_feat = data.target_edge_feat.view(bsz, -1)

        combined = torch.cat([sortpool_vec, endpoint_sum, endpoint_diff, raw_pair, target_feat], dim=-1)
        logits = self.head(combined).squeeze(-1)
        return logits

    def forward(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Returns predicted blockage probabilities q_hat_e in [0, 1].
        """
        return torch.sigmoid(self.forward_logits(data))

    @torch.no_grad()
    def predict_edges(
        self,
        nx_graph: Union[nx.Graph, nx.DiGraph],
        edge_status_dict: Optional[Dict[Tuple[int, int], Union[int, str, float]]],
        target_edges: Optional[Sequence[Tuple[int, int]]] = None,
        k_hop: int = 2,
        batch_size: int = 256,
    ) -> Dict[Tuple[int, int], float]:
        """
        Fast, batched inference method predicting blockage probabilities q_hat_e in [0, 1]
        for all `target_edges` in `nx_graph` conditioned on `edge_status_dict`.
        """
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device

        G = prepare_spatial_graph(nx_graph, copy_graph=False)
        norm_status = normalize_edge_status_dict(G, edge_status_dict)

        if target_edges is None:
            target_edges = [
                (u, v) for u, v in nx_graph.edges()
                if norm_status.get((u, v), STATUS_UNKNOWN) == STATUS_UNKNOWN
            ]
        else:
            target_edges = list(target_edges)

        if len(target_edges) == 0:
            if was_training:
                self.train()
            return {}

        cache = _GraphExtractionCache(G, norm_status)

        # Deduplicate undirected canonical queries so directed graphs (u,v) & (v,u) only compute once
        unique_queries: List[Tuple[int, int]] = []
        canon_to_idx: Dict[Tuple[int, int], int] = {}
        query_map: List[int] = []

        for u, v in target_edges:
            ckey = (min(u, v), max(u, v)) if not isinstance(u, str) else (u, v)
            if ckey not in canon_to_idx:
                canon_to_idx[ckey] = len(unique_queries)
                unique_queries.append((u, v))
            query_map.append(canon_to_idx[ckey])

        subgraphs = [
            extract_enclosing_subgraph(cache, q_edge, k_hop=k_hop)
            for q_edge in unique_queries
        ]

        preds_unique: List[float] = []
        for start in range(0, len(subgraphs), batch_size):
            batch_data = Batch.from_data_list(subgraphs[start : start + batch_size]).to(device)
            probs = self.forward(batch_data).detach().cpu().numpy().ravel()
            preds_unique.extend(float(np.clip(p, 0.0, 1.0)) for p in probs)

        if was_training:
            self.train()

        return {
            edge: preds_unique[query_map[idx]]
            for idx, edge in enumerate(target_edges)
        }


class SEALPredictor:
    """
    High-level wrapper around `SEALModel` that loads trained weights (if available)
    and exposes `.predict_edges(...)` along with integration utilities for MARVEL
    and SEVAC.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        hidden_dim: int = 64,
        num_layers: int = 3,
        sortpool_k: int = 16,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path is None:
            default_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "best_seal_model.pt")
            if os.path.exists(default_ckpt):
                checkpoint_path = default_ckpt

        self.checkpoint_path = checkpoint_path
        self.model = SEALModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            sortpool_k=sortpool_k,
        ).to(self.device)

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                cfg = ckpt.get("config", {})
                if cfg:
                    self.model = SEALModel(
                        node_in_dim=cfg.get("node_in_dim", NODE_FEATURE_DIM),
                        edge_in_dim=cfg.get("edge_in_dim", EDGE_FEATURE_DIM),
                        hidden_dim=cfg.get("hidden_dim", hidden_dim),
                        num_layers=cfg.get("num_layers", num_layers),
                        sortpool_k=cfg.get("sortpool_k", sortpool_k),
                        dropout=cfg.get("dropout", 0.1),
                    ).to(self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)
        self.model.eval()

    def predict_edges(
        self,
        nx_graph: Union[nx.Graph, nx.DiGraph],
        edge_status_dict: Optional[Dict[Tuple[int, int], Union[int, str, float]]],
        target_edges: Optional[Sequence[Tuple[int, int]]] = None,
        k_hop: int = 2,
        batch_size: int = 256,
    ) -> Dict[Tuple[int, int], float]:
        return self.model.predict_edges(
            nx_graph=nx_graph,
            edge_status_dict=edge_status_dict,
            target_edges=target_edges,
            k_hop=k_hop,
            batch_size=batch_size,
        )

    def update_marvel_adj_and_entropy(
        self,
        nx_graph: Union[nx.Graph, nx.DiGraph],
        adj_matrix: torch.Tensor,
        node_mapping: Dict[int, int],
        edge_status_dict: Dict[Tuple[int, int], Union[int, str, float]],
        target_edges: Optional[Sequence[Tuple[int, int]]] = None,
        k_hop: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[Tuple[int, int], float]]:
        """
        Seamless adapter for `marvel_modernized/MARVEL/Env_724.py`:
        Predicts blockage probabilities q_hat_e for unobserved k-hop edges, converts them
        to traversal probabilities p_hat_e = 1 - q_hat_e, updates MARVEL's dynamic
        `adj_matrix`, and returns the updated graph entropy H(G).
        """
        q_hat = self.predict_edges(nx_graph, edge_status_dict, target_edges=target_edges, k_hop=k_hop)
        updated_adj = adj_matrix.clone()
        eps = 1e-8
        total_entropy = torch.tensor(0.0, dtype=torch.float32, device=adj_matrix.device)

        for (u, v), q_e in q_hat.items():
            p_pass = float(np.clip(1.0 - q_e, 0.0, 1.0))
            if u in node_mapping and v in node_mapping:
                iu, iv = node_mapping[u], node_mapping[v]
                updated_adj[iu, iv] = p_pass
                if not nx_graph.is_directed():
                    updated_adj[iv, iu] = p_pass
            if 0.0 < p_pass < 1.0:
                h_e = -(p_pass * math.log(p_pass + eps) + (1.0 - p_pass) * math.log(1.0 - p_pass + eps))
                total_entropy = total_entropy + h_e

        return updated_adj, total_entropy, q_hat

    def build_sevac_edge_features(
        self,
        nx_graph: Union[nx.Graph, nx.DiGraph],
        edge_status_dict: Dict[Tuple[int, int], Union[int, str, float]],
        time_budget: float,
        k_hop: int = 2,
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
        """
        Seamless adapter for `sevac/code/SEVAC/`:
        Computes predicted blockage probabilities q_hat_e for all edges and constructs
        an augmented state-action feature vector [time_budget * (1 - q_hat_e), (1 - q_hat_e), q_hat_e].
        """
        all_edges = list(nx_graph.edges())
        norm_status = normalize_edge_status_dict(nx_graph, edge_status_dict)
        unknown_edges = [e for e in all_edges if norm_status.get(e, STATUS_UNKNOWN) == STATUS_UNKNOWN]
        q_hat = self.predict_edges(nx_graph, edge_status_dict, target_edges=unknown_edges, k_hop=k_hop)

        full_q: Dict[Tuple[int, int], float] = {}
        surv_probs = np.zeros(len(all_edges), dtype=np.float64)
        for idx, e in enumerate(all_edges):
            st = norm_status.get(e, STATUS_UNKNOWN)
            if st == STATUS_CLEAR:
                q_val = 0.0
            elif st == STATUS_BLOCKED:
                q_val = 1.0
            else:
                q_val = float(q_hat.get(e, 0.25))
            full_q[e] = q_val
            surv_probs[idx] = 1.0 - q_val

        phi_augmented = np.concatenate([
            float(time_budget) * surv_probs,
            surv_probs,
            1.0 - surv_probs,
        ])
        return phi_augmented, full_q


_DEFAULT_PREDICTOR: Optional[SEALPredictor] = None


def predict_edges(
    nx_graph: Union[nx.Graph, nx.DiGraph],
    edge_status_dict: Optional[Dict[Tuple[int, int], Union[int, str, float]]],
    target_edges: Optional[Sequence[Tuple[int, int]]] = None,
    k_hop: int = 2,
    model: Optional[Union[SEALModel, SEALPredictor]] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[Tuple[int, int], float]:
    """
    Top-level functional inference API matching:
        predict_edges(nx_graph, edge_status_dict, target_edges, k_hop=2) -> Dict[edge, float]

    Returns predicted blockage probabilities q_hat_e in [0, 1] for each edge in `target_edges`.
    """
    global _DEFAULT_PREDICTOR
    if model is not None:
        return model.predict_edges(
            nx_graph=nx_graph,
            edge_status_dict=edge_status_dict,
            target_edges=target_edges,
            k_hop=k_hop,
        )
    if checkpoint_path is not None:
        predictor = SEALPredictor(checkpoint_path=checkpoint_path)
        return predictor.predict_edges(
            nx_graph=nx_graph,
            edge_status_dict=edge_status_dict,
            target_edges=target_edges,
            k_hop=k_hop,
        )
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = SEALPredictor()
    return _DEFAULT_PREDICTOR.predict_edges(
        nx_graph=nx_graph,
        edge_status_dict=edge_status_dict,
        target_edges=target_edges,
        k_hop=k_hop,
    )
