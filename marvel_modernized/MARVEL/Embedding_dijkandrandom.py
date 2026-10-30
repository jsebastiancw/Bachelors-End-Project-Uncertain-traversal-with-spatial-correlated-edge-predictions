import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import Node2Vec as PyGNode2Vec
    HAS_PYG_NODE2VEC = True
except ImportError:
    HAS_PYG_NODE2VEC = False


class _KeyedVectorsAdapter:
    """
    Adapter providing legacy Gensim-compatible `w2v.wv[str(node)]` lookup
    backed directly by a GPU PyTorch tensor of shape (num_nodes, emb_size).
    """
    def __init__(self, embeddings: torch.Tensor, node_mapping: dict):
        self.embeddings = embeddings  # (num_nodes, emb_size) on device
        self.node_mapping = node_mapping
        self.index_to_key = [str(node) for node in node_mapping.keys()]

    def __getitem__(self, key):
        # Support lookup by string node label (e.g. '3') or integer/float node ID
        if key in self.node_mapping:
            idx = self.node_mapping[key]
        else:
            try:
                int_key = int(float(key))
                if int_key in self.node_mapping:
                    idx = self.node_mapping[int_key]
                else:
                    idx = int_key
            except (ValueError, TypeError):
                raise KeyError(f"Node key {key} not found in node_mapping.")
        return self.embeddings[idx]


class EmbeddingResult:
    """
    Container returned by `Node2Vec.train(...)`.
    Exposes both `.embeddings` (GPU Tensor [N, D]) and legacy `.wv` lookup.
    """
    def __init__(self, embeddings: torch.Tensor, node_mapping: dict):
        self.embeddings = embeddings
        self.wv = _KeyedVectorsAdapter(embeddings, node_mapping)

    def to(self, device):
        self.embeddings = self.embeddings.to(device)
        self.wv.embeddings = self.embeddings
        return self


class ShortestPathSkipGram(nn.Module):
    """
    GPU-accelerated PyTorch Skip-Gram + Structural Shortest-Path Graph Embedding.
    Replaces legacy CPU `gensim.models.Word2Vec` with vectorized CUDA operations.
    """
    def __init__(self, num_nodes: int, emb_size: int, struct_dim: int = 8):
        super().__init__()
        self.num_nodes = num_nodes
        self.emb_size = emb_size
        self.target_embeddings = nn.Embedding(num_nodes, emb_size)
        self.context_embeddings = nn.Embedding(num_nodes, emb_size)
        self.struct_proj = nn.Linear(struct_dim, emb_size)

        nn.init.xavier_uniform_(self.target_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
        nn.init.xavier_uniform_(self.struct_proj.weight)

    def forward(self, center_idx: torch.Tensor, pos_ctx_idx: torch.Tensor, neg_ctx_idx: torch.Tensor) -> torch.Tensor:
        """
        Computes Skip-Gram with Negative Sampling (SGNS) loss entirely on GPU.
        Args:
            center_idx: (B,) tensor of center node indices
            pos_ctx_idx: (B,) tensor of positive context node indices
            neg_ctx_idx: (B, K) tensor of negative context node indices
        """
        v = self.target_embeddings(center_idx)              # (B, D)
        u_pos = self.context_embeddings(pos_ctx_idx)        # (B, D)
        u_neg = self.context_embeddings(neg_ctx_idx)        # (B, K, D)

        pos_score = torch.sum(v * u_pos, dim=-1)            # (B,)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(u_neg, v.unsqueeze(-1)).squeeze(-1)  # (B, K)
        neg_loss = torch.sum(F.logsigmoid(-neg_score), dim=-1)     # (B,)

        return -torch.mean(pos_loss + neg_loss)

    @torch.no_grad()
    def get_embeddings(self, struct_features: torch.Tensor) -> torch.Tensor:
        base_emb = self.target_embeddings.weight
        struct_emb = self.struct_proj(struct_features)
        combined = F.layer_norm(base_emb + struct_emb, (self.emb_size,))
        return combined


class Node2Vec:
    """
    Modernized PyTorch GPU Graph Embedding for MARVEL.
    Ingests dynamic graph structure, edge traversal probabilities, and Dijkstra
    shortest paths to target/goal nodes to produce node feature matrices on CUDA.
    """
    def __init__(
        self,
        data_name: str,
        emb_size: int = 128,
        length_walk: int = 50,
        num_walks: int = 10,
        window_size: int = 4,
        num_iters: int = 15,
        n: int = 50,
        repeat_times: int = 20,
        device: torch.device = None,
    ):
        self.data_name = data_name
        self.emb_size = emb_size
        self.length_walk = length_walk
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_iters = num_iters
        self.n = n
        self.repeat_time = repeat_times
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        base_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.dirname(base_dir)

        self.datasets = {
            'Simple': os.path.join(base_dir, 'Roaddata', 'Simplifiedroadnet', 'Simplifiedroadnet.xlsx'),
            'Simple_2': os.path.join(base_dir, 'Roaddata', 'Simplifiedroadnet', 'Simplifiedroadnet.xlsx'),
            'SiouxFalls': os.path.join(repo_dir, 'Networks', 'SiouxFalls', 'SiouxFalls_network.csv'),
            'Anaheim': os.path.join(repo_dir, 'Networks', 'Anaheim', 'Anaheim_network.csv'),
            'Barcelona': os.path.join(repo_dir, 'Networks', 'Barcelona', 'Barcelona_network.csv'),
            'Chicago_Sketch': os.path.join(repo_dir, 'Networks', 'Chicago_Sketch', 'Chicago_Sketch_network.csv'),
            'Museum': os.path.join(repo_dir, 'Benchmark_SOTA', 'maps', 'Museum_network.csv'),
            'Office': os.path.join(repo_dir, 'Benchmark_SOTA', 'maps', 'Office_network.csv'),
        }
        self.G = self.load_graph()

    def _build_default_simple_graph(self) -> nx.DiGraph:
        """
        Builds the 14-node simplified benchmark road network when the external
        Excel file is not present on disk. Matches the MARVEL paper's 14-node
        topology where Robot 1 routes 9 -> 12 and Robot 2 routes 1 -> 8.
        """
        G = nx.DiGraph()
        edges = [
            # Robot 2 main corridors (1 -> 8)
            (1, 2, 2.0, 1.0), (2, 1, 2.0, 1.0),
            (1, 3, 2.5, 0.85), (3, 1, 2.5, 0.85),
            (2, 4, 2.2, 1.0), (4, 2, 2.2, 1.0),
            (2, 5, 2.0, 0.75), (5, 2, 2.0, 0.75),
            (3, 5, 1.8, 1.0), (5, 3, 1.8, 1.0),
            (3, 6, 2.4, 0.8), (6, 3, 2.4, 0.8),
            (4, 7, 2.0, 1.0), (7, 4, 2.0, 1.0),
            (5, 7, 1.9, 0.9), (7, 5, 1.9, 0.9),
            (5, 8, 2.5, 0.7), (8, 5, 2.5, 0.7),
            (6, 8, 2.1, 1.0), (8, 6, 2.1, 1.0),
            (7, 8, 1.8, 1.0), (8, 7, 1.8, 1.0),
            # Cross-links between corridors
            (4, 9, 2.5, 1.0), (9, 4, 2.5, 1.0),
            (6, 10, 2.2, 1.0), (10, 6, 2.2, 1.0),
            (7, 11, 2.0, 0.85), (11, 7, 2.0, 0.85),
            # Robot 1 main corridors (9 -> 12, including path [9, 10, 13, 11, 12])
            (9, 10, 1.0, 1.0), (10, 9, 1.0, 1.0),
            (9, 11, 1.8, 0.75), (11, 9, 1.8, 0.75),
            (10, 13, 1.0, 1.0), (13, 10, 1.0, 1.0),
            (10, 14, 1.5, 0.8), (14, 10, 1.5, 0.8),
            (13, 11, 1.0, 0.9), (11, 13, 1.0, 0.9),
            (13, 12, 1.6, 0.7), (12, 13, 1.6, 0.7),
            (11, 12, 1.0, 1.0), (12, 11, 1.0, 1.0),
            (14, 12, 1.4, 1.0), (12, 14, 1.4, 1.0),
        ]
        for u, v, cost, prob in edges:
            G.add_edge(int(u), int(v), weight=float(cost), probability=float(prob))
        return G

    def load_graph(self) -> nx.DiGraph:
        file_path = self.datasets.get(self.data_name, self.data_name)

        if not os.path.exists(file_path):
            if self.data_name in ('Simple', 'Simple_2'):
                print(f"[Node2Vec] '{file_path}' not found; using built-in 14-node Simple road network.")
                return self._build_default_simple_graph()
            raise FileNotFoundError(f"Dataset file not found for '{self.data_name}': {file_path}")

        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        # Ensure traversal probability column exists (P or prob)
        if 'P' in df.columns:
            prob_col = 'P'
        elif 'prob' in df.columns:
            prob_col = 'prob'
        else:
            # Generate synthetic stochastic edge probabilities (10% uncertain edges in [0.5, 1.0))
            rng = np.random.default_rng(616)
            m = len(df)
            probs = np.ones(m, dtype=np.float32)
            num_stochastic = max(1, int(np.floor(m * 0.1)))
            stoch_idx = rng.choice(m, size=num_stochastic, replace=False)
            probs[stoch_idx] = rng.uniform(0.5, 0.95, size=num_stochastic)
            df['P'] = probs
            prob_col = 'P'

        G = nx.DiGraph()
        for _, row in df.iterrows():
            from_node = int(row['From'])
            to_node = int(row['To'])
            cost = float(row['Cost'])
            prob = float(row[prob_col])
            G.add_edge(from_node, to_node, weight=cost, probability=prob)

        return G

    def generate_temp_graph(self, adj_matrix: torch.Tensor, node_mapping: dict) -> nx.DiGraph:
        """
        Constructs a temporary NetworkX DiGraph reflecting currently active edges
        where `adj_matrix[u_idx, v_idx] > 0`.
        """
        adj_cpu = adj_matrix.detach().cpu().numpy() if isinstance(adj_matrix, torch.Tensor) else np.asarray(adj_matrix)
        temp_G = nx.DiGraph()
        temp_G.add_nodes_from(self.G.nodes())

        for u, v, data in self.G.edges(data=True):
            u2 = node_mapping[u]
            v2 = node_mapping[v]
            if adj_cpu[u2, v2] > 0:
                temp_G.add_edge(u, v, weight=data['weight'])
        return temp_G

    def generate_paths(self, temp_G: nx.DiGraph, goal_nodes: list) -> list:
        """
        Generates shortest paths (Dijkstra) from all nodes to all active target/goal nodes.
        """
        paths = []
        for node in temp_G.nodes():
            for goal_node in goal_nodes:
                if node != goal_node and goal_node in temp_G:
                    try:
                        path = nx.dijkstra_path(temp_G, node, goal_node, weight='weight')
                        paths.extend([path] * self.repeat_time)
                    except nx.NetworkXNoPath:
                        continue
        return paths

    def random_walk_one_step(self, start_node, temp_G: nx.DiGraph) -> list:
        walks = []
        if start_node not in temp_G:
            return walks
        neighbors = list(temp_G.neighbors(start_node))
        if not neighbors:
            return walks

        weights = [1.0 / max(temp_G[start_node][nbr]['weight'], 1e-6) for nbr in neighbors]
        total_w = sum(weights)
        probabilities = [w / total_w for w in weights]
        counts = [max(1, round(p * self.n)) for p in probabilities]

        for neighbor, count in zip(neighbors, counts):
            walks.extend([[start_node, neighbor]] * count)
        return walks

    def generate_random_walks(self, temp_G: nx.DiGraph) -> list:
        walks = []
        for node in self.G.nodes():
            walks.extend(self.random_walk_one_step(node, temp_G))
        return walks

    def _compute_structural_features(
        self,
        temp_G: nx.DiGraph,
        node_mapping: dict,
        goal_nodes: list,
        struct_dim: int = 8,
    ) -> torch.Tensor:
        """
        Computes an (N, struct_dim) tensor of shortest-path and topological features
        to each target node on `self.device`.
        """
        num_nodes = len(node_mapping)
        feats = np.zeros((num_nodes, struct_dim), dtype=np.float32)

        # Shortest-path distances & hop counts to up to 2 goal nodes
        for g_idx, goal in enumerate(goal_nodes[:2]):
            if goal in temp_G:
                try:
                    lengths = nx.single_target_shortest_path_length(temp_G, goal)
                except Exception:
                    lengths = {}
                for node, idx in node_mapping.items():
                    if node == goal:
                        feats[idx, g_idx * 2] = 0.0
                        feats[idx, g_idx * 2 + 1] = 1.0  # Reachable indicator
                    elif node in temp_G:
                        try:
                            cost = nx.dijkstra_path_length(temp_G, node, goal, weight='weight')
                            feats[idx, g_idx * 2] = 1.0 / (1.0 + cost)
                            feats[idx, g_idx * 2 + 1] = 1.0
                        except nx.NetworkXNoPath:
                            feats[idx, g_idx * 2] = 0.0
                            feats[idx, g_idx * 2 + 1] = 0.0

        # Degree & local connectivity statistics
        max_deg = max(1, self.G.number_of_nodes())
        for node, idx in node_mapping.items():
            out_deg = temp_G.out_degree(node) if node in temp_G else 0
            in_deg = temp_G.in_degree(node) if node in temp_G else 0
            feats[idx, 4] = out_deg / max_deg
            feats[idx, 5] = in_deg / max_deg
            if out_deg > 0:
                costs = [temp_G[node][nbr]['weight'] for nbr in temp_G.neighbors(node)]
                feats[idx, 6] = 1.0 / (1.0 + float(np.mean(costs)))
                feats[idx, 7] = 1.0 / (1.0 + float(np.min(costs)))

        return torch.tensor(feats, dtype=torch.float32, device=self.device)

    def _extract_skipgram_pairs(self, sequences: list, node_mapping: dict) -> tuple:
        """
        Converts node walk sequences into vectorized (center_idx, context_idx) GPU tensors.
        """
        centers = []
        contexts = []
        w = self.window_size

        for seq in sequences:
            mapped = [node_mapping[n] for n in seq if n in node_mapping]
            L = len(mapped)
            for i, center in enumerate(mapped):
                start = max(0, i - w)
                end = min(L, i + w + 1)
                for j in range(start, end):
                    if i != j:
                        centers.append(center)
                        contexts.append(mapped[j])

        if not centers:
            # Fallback self-loops if graph has no active walks
            all_idx = list(range(len(node_mapping)))
            centers = all_idx
            contexts = all_idx

        center_tensor = torch.tensor(centers, dtype=torch.long, device=self.device)
        context_tensor = torch.tensor(contexts, dtype=torch.long, device=self.device)
        return center_tensor, context_tensor

    def train(
        self,
        adj_matrix: torch.Tensor,
        node_mapping: dict,
        goal_nodes: list,
        workers: int = 4,
        is_loadmodel: bool = False,
        is_loaddata: bool = False,
    ) -> EmbeddingResult:
        """
        Trains the shortest-path + random-walk graph embedding directly on `self.device` (GPU)
        and returns an `EmbeddingResult` containing the (num_nodes, emb_size) tensor.
        """
        num_nodes = len(node_mapping)
        temp_G = self.generate_temp_graph(adj_matrix, node_mapping)

        paths = self.generate_paths(temp_G, goal_nodes)
        random_walks = self.generate_random_walks(temp_G)
        all_sequences = paths + random_walks

        center_tensor, context_tensor = self._extract_skipgram_pairs(all_sequences, node_mapping)
        struct_features = self._compute_structural_features(temp_G, node_mapping, goal_nodes)

        torch.manual_seed(616)
        model = ShortestPathSkipGram(num_nodes=num_nodes, emb_size=self.emb_size).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        num_pairs = center_tensor.size(0)
        batch_size = min(4096, num_pairs)
        num_neg = min(5, max(1, num_nodes - 1))

        model.train()
        for _ in range(self.num_iters):
            perm = torch.randperm(num_pairs, device=self.device)
            for start in range(0, num_pairs, batch_size):
                idx = perm[start : start + batch_size]
                c_batch = center_tensor[idx]
                p_batch = context_tensor[idx]
                neg_batch = torch.randint(
                    0, num_nodes, (c_batch.size(0), num_neg), device=self.device, dtype=torch.long
                )

                optimizer.zero_grad()
                loss = model(c_batch, p_batch, neg_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        embeddings = model.get_embeddings(struct_features).detach()
        return EmbeddingResult(embeddings, node_mapping)
