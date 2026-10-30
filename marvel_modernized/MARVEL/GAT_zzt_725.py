import os
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
import matplotlib.pyplot as plt

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class GAT(nn.Module):
    """
    Modernized Graph Attention Policy Network (GAT) for MARVEL.
    Processes dynamic node feature matrices and topological adjacency/edge_index
    tensors on CUDA to output routing decision logits across candidate next nodes.
    """
    def __init__(
        self,
        data_name: str,
        learning_rate: float,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        heads: int = 6,
        dropout: float = 0.2,
        alpha: float = 0.2,
        use_wandb: bool = False,
    ):
        super().__init__()
        self.data_name = data_name
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.use_wandb = use_wandb and HAS_WANDB

        head_dim = max(1, in_channels // heads)
        gat_out_dim = head_dim * heads

        # Multi-head Graph Attention Layers (PyG GATConv)
        self.gat_conv1 = GATConv(
            in_channels=in_channels,
            out_channels=head_dim,
            heads=heads,
            concat=True,
            negative_slope=alpha,
            dropout=dropout,
            add_self_loops=True,
        )
        self.proj1 = nn.Linear(gat_out_dim, in_channels) if gat_out_dim != in_channels else nn.Identity()
        self.norm1 = nn.LayerNorm(in_channels)

        self.gat_conv2 = GATConv(
            in_channels=in_channels,
            out_channels=head_dim,
            heads=heads,
            concat=True,
            negative_slope=alpha,
            dropout=dropout,
            add_self_loops=True,
        )
        self.proj2 = nn.Linear(gat_out_dim, in_channels) if gat_out_dim != in_channels else nn.Identity()
        self.norm2 = nn.LayerNorm(in_channels)

        # Graph Convolutional refinement layers
        self.gcn3 = GCNConv(in_channels, 2 * in_channels)
        self.gcn4 = GCNConv(2 * in_channels, 4 * in_channels)

        if self.use_wandb:
            wandb.init(project='MACTP_MARVEL', name=time.strftime('%m%d%H%M%S_GAT'))
            wandb.config.update({
                "data_name": data_name,
                "learning_rate": learning_rate,
                "in_channels": in_channels,
                "heads": heads,
            })

    @staticmethod
    def _ensure_edge_index(edge_index_or_adj: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Accepts either a PyG `edge_index` tensor of shape (2, E) or a dense
        adjacency matrix `(N, N)` and returns a `(2, E)` `torch.long` tensor on `device`.
        """
        tensor = edge_index_or_adj.to(device)
        if tensor.dim() == 2 and tensor.size(0) == 2 and tensor.dtype in (torch.long, torch.int64, torch.int32):
            return tensor.long()
        # Dense adjacency matrix -> sparse edge_index where traversable (adj == 1 or adj > 0)
        row, col = torch.where(tensor > 0)
        return torch.stack([row, col], dim=0).long()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        current_node_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass computing routing logits from `current_node_indices` to all graph nodes.
        Args:
            x: Dynamic node feature matrix of shape (N, in_channels) on device.
            edge_index: Edge index (2, E) or adjacency matrix (N, N) on device.
            current_node_indices: Tensor of shape (1,) containing the active agent's node index.
        Returns:
            out: 1D Tensor of shape (N,) with routing scores for each node.
        """
        device = x.device
        x = x.to(device)
        edge_index = self._ensure_edge_index(edge_index, device)
        current_node_indices = current_node_indices.to(device).long()

        # Layer 1: Multi-head GAT + Residual + LayerNorm
        residual = x
        h = self.gat_conv1(x, edge_index)
        h = self.proj1(h)
        x = self.norm1(h + residual)

        # Layer 2: Multi-head GAT + Residual + LayerNorm
        residual = x
        h = self.gat_conv2(x, edge_index)
        h = self.proj2(h)
        x = self.norm2(h + residual)

        # Layer 3 & 4: GCN message passing
        x = F.relu(self.gcn3(x, edge_index))
        x = self.gcn4(x, edge_index)

        # Pairwise compatibility score matrix: (N, N)
        similarity = torch.matmul(x, x.t())

        # Extract routing logits for the current node: (1, N) -> (N,)
        query_idx = current_node_indices.view(-1, 1).expand(-1, similarity.size(1))
        optimal = similarity.gather(0, query_idx)
        out = optimal.view(-1)
        return out

    def train_model(
        self,
        env,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_episodes: int,
        device: torch.device,
        scheduler,
        model_path: str,
        emb_size: int,
        save_interval: int = 500,
    ):
        self.to(device)
        success_rates = []
        total_losses = []
        success_count = 0

        if model_path and model_path != 'your_model_path':
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)

        for episode in range(num_episodes):
            env.reset()
            total_reward = 0.0
            total_loss = 0.0
            done = False

            while not done:
                reward, done, loss, success = env.step(
                    self, optimizer, criterion, device, model_path, emb_size
                )
                total_reward += float(reward)
                total_loss += float(loss.item() if isinstance(loss, torch.Tensor) else loss)
                if success:
                    success_count += 1

            success_rate = success_count / (episode + 1)
            success_rates.append(success_rate)
            total_losses.append(total_loss)

            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Success Rate: {success_rate:.4f} | "
                f"Total Loss: {total_loss:.4f}"
            )

            if scheduler is not None:
                scheduler.step()

            if self.use_wandb:
                wandb.log({
                    "episode": episode + 1,
                    "success_rate": success_rate,
                    "total_loss": total_loss,
                })

            if model_path and model_path != 'your_model_path' and (episode + 1) % save_interval == 0:
                torch.save(self.state_dict(), model_path)

        if model_path and model_path != 'your_model_path':
            torch.save(self.state_dict(), model_path)

        if self.use_wandb:
            wandb.finish()

        return success_rates, total_losses

    def plot_success_rates(self, success_rates):
        plt.figure(figsize=(8, 5))
        plt.plot(success_rates)
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.title('MARVEL GAT Policy — Success Rate per Episode')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
