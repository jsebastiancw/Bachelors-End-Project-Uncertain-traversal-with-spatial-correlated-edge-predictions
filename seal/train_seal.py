"""
seal/train_seal.py
Deliverable 1: Standalone Supervised Teacher-Student Training Script for SEAL.

Generates diverse procedural spatial graphs (sparse, dense, multidistrict) via
`graph/graph_builder.py` and correlated Gaussian Process disaster fields
(earthquake epicenters, coastal tsunamis, meandering river floods) via
`graph/gaussian_disaster_simulator.py`.

Simulates partial observability by randomly masking 40%-90% of edges as Unknown
and revealing the remaining 10%-60% as ground-truth Clear/Blocked.
Trains SEAL jointly against continuous GP teacher probabilities q_e^{GP} (MSE + KL)
and binary realizations B_e (BCE), logging verbose epoch-by-epoch diagnostics.
"""

import os
import sys
import json
import time
import math
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Ensure project root is on sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from graph.graph_builder import SpatialGraphGenerator
from graph.gaussian_disaster_simulator import (
    set_disaster_type,
    sample_gp_disaster_realization,
)
from seal.model import (
    STATUS_CLEAR,
    STATUS_BLOCKED,
    STATUS_UNKNOWN,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    SEALModel,
    prepare_spatial_graph,
    normalize_edge_status_dict,
    _GraphExtractionCache,
    extract_enclosing_subgraph,
)


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_bernoulli_kl(q_pred: torch.Tensor, q_true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes Bernoulli KL divergence D_KL(Bern(q_true) || Bern(q_pred)).
    """
    p = torch.clamp(q_true, eps, 1.0 - eps)
    q = torch.clamp(q_pred, eps, 1.0 - eps)
    kl = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
    return torch.mean(kl)


def generate_masked_scenario_subgraphs(
    G_base,
    disaster_type: str,
    rng: np.random.Generator,
    k_hop: int = 2,
    mask_ratios: Tuple[float, ...] = (0.40, 0.60, 0.75, 0.90),
    max_targets_per_mask: int = 48,
    max_sub_nodes: int = 150,
) -> List[Data]:
    """
    Applies a GP disaster scenario to `G_base`, samples a valid binary realization B_e,
    and generates partial observability masks with 40%-90% of edges masked as Unknown.
    Extracts k-hop enclosing subgraphs for masked target edges with both teacher labels
    q_e^{GP} and student binary realizations B_e.
    """
    G = G_base.copy()
    boundaries = ["right", "left", "top", "bottom"]
    boundary = str(rng.choice(boundaries))
    num_epi = int(rng.choice([1, 2]))
    scenario_seed = int(rng.integers(0, 1_000_000))

    G, _ = set_disaster_type(
        G,
        disaster_type=disaster_type,
        num_epicenters=num_epi,
        boundary=boundary,
        safe_y=-3.0,
        k_steepness=0.6,
        y_offset=1.5,
        max_blockage_prob=0.92,
        seed=scenario_seed,
    )
    G = prepare_spatial_graph(G, copy_graph=False)
    sample_gp_disaster_realization(G, max_resamples=300, allow_relaxation=True)

    all_edges = list(G.edges())
    if len(all_edges) < 4:
        return []

    gp_probs: Dict[Tuple[int, int], float] = {}
    bin_realizations: Dict[Tuple[int, int], int] = {}
    for u, v, d in G.edges(data=True):
        q_e = float(d.get("blockage_prob", 0.0))
        b_e = int(d.get("is_blocked", 0))
        gp_probs[(u, v)] = q_e
        gp_probs[(v, u)] = q_e
        bin_realizations[(u, v)] = b_e
        bin_realizations[(v, u)] = b_e

    subgraphs: List[Data] = []

    for mask_ratio in mask_ratios:
        # Jitter mask ratio inside [0.40, 0.90]
        m_ratio = float(np.clip(mask_ratio + rng.uniform(-0.05, 0.05), 0.40, 0.90))
        num_edges = len(all_edges)
        num_masked = max(1, min(num_edges - 1, int(round(m_ratio * num_edges))))

        perm = rng.permutation(num_edges)
        masked_indices = set(perm[:num_masked].tolist())

        edge_status_dict: Dict[Tuple[int, int], int] = {}
        masked_edges: List[Tuple[int, int]] = []

        for idx, e in enumerate(all_edges):
            if idx in masked_indices:
                edge_status_dict[e] = STATUS_UNKNOWN
                masked_edges.append(e)
            else:
                edge_status_dict[e] = STATUS_BLOCKED if bin_realizations[e] == 1 else STATUS_CLEAR

        norm_status = normalize_edge_status_dict(G, edge_status_dict)
        cache = _GraphExtractionCache(G, norm_status)

        if len(masked_edges) > max_targets_per_mask:
            sel_idx = rng.choice(len(masked_edges), size=max_targets_per_mask, replace=False)
            target_subset = [masked_edges[i] for i in sel_idx]
        else:
            target_subset = masked_edges

        for u, v in target_subset:
            sub = extract_enclosing_subgraph(
                cache=cache,
                target_edge=(u, v),
                k_hop=k_hop,
                y_prob=gp_probs[(u, v)],
                y_bin=float(bin_realizations[(u, v)]),
                max_sub_nodes=max_sub_nodes,
            )
            subgraphs.append(sub)

    return subgraphs


def build_procedural_dataset(
    num_graphs: int,
    k_hop: int = 2,
    seed: int = 42,
    desc: str = "Train",
    max_sub_nodes: int = 150,
) -> List[Data]:
    """
    Builds a diverse dataset of enclosing subgraphs across procedural graph regimes
    ('sparse', 'dense', 'multidistrict') and GP disaster topologies
    ('earthquake', 'tsunami', 'flood').
    """
    rng = np.random.default_rng(seed)
    regimes = ["sparse", "dense", "multidistrict"]
    disasters = ["earthquake", "tsunami", "flood"]
    dataset: List[Data] = []

    t0 = time.perf_counter()
    for g_idx in range(num_graphs):
        regime = regimes[g_idx % len(regimes)]
        n_nodes = int(rng.integers(32, 65))
        c_clusters = int(rng.integers(3, 6))
        d_min = 0.04 if n_nodes <= 45 else 0.03

        builder = SpatialGraphGenerator(n_nodes=n_nodes, d_min=d_min)
        G_base, _ = builder.build_graph(regime=regime, C=c_clusters)

        # Apply 2 distinct disaster topologies per procedural graph
        chosen_disasters = [
            disasters[(g_idx + offset) % len(disasters)]
            for offset in range(2)
        ]
        for d_type in chosen_disasters:
            subs = generate_masked_scenario_subgraphs(
                G_base=G_base,
                disaster_type=d_type,
                rng=rng,
                k_hop=k_hop,
                mask_ratios=(0.45, 0.65, 0.85),
                max_targets_per_mask=36,
                max_sub_nodes=max_sub_nodes,
            )
            dataset.extend(subs)

    elapsed = time.perf_counter() - t0
    print(
        f"[{desc} Dataset] Generated {len(dataset):,} k-hop={k_hop} enclosing subgraphs "
        f"from {num_graphs} procedural graphs in {elapsed:.2f}s."
    )
    return dataset


@torch.no_grad()
def evaluate_loader(
    model: SEALModel,
    loader: DataLoader,
    device: torch.device,
    bce_weight: float = 0.35,
    mse_weight: float = 0.45,
    kl_weight: float = 0.20,
) -> Dict[str, float]:
    model.eval()
    total_bce = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_kl = 0.0
    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_probs = []
    all_bins = []
    batch_latencies_ms = []

    for batch in loader:
        batch = batch.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        logits = model.forward_logits(batch)
        preds = torch.sigmoid(logits)

        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

        y_prob = batch.y_prob.view(-1)
        y_bin = batch.y_bin.view(-1)
        bsz = y_prob.size(0)

        bce = F.binary_cross_entropy_with_logits(logits, y_bin)
        mse = F.mse_loss(preds, y_prob)
        mae = F.l1_loss(preds, y_prob)
        kl = compute_bernoulli_kl(preds, y_prob)
        loss = bce_weight * bce + mse_weight * mse + kl_weight * kl

        total_bce += float(bce.item()) * bsz
        total_mse += float(mse.item()) * bsz
        total_mae += float(mae.item()) * bsz
        total_kl += float(kl.item()) * bsz
        total_loss += float(loss.item()) * bsz
        total_samples += bsz

        all_preds.append(preds.detach().cpu().numpy())
        all_probs.append(y_prob.detach().cpu().numpy())
        all_bins.append(y_bin.detach().cpu().numpy())

    preds_np = np.concatenate(all_preds) if all_preds else np.zeros(1)
    bins_np = np.concatenate(all_bins) if all_bins else np.zeros(1)

    if len(np.unique(bins_np)) > 1:
        roc_auc = float(roc_auc_score(bins_np, preds_np))
        ap = float(average_precision_score(bins_np, preds_np))
    else:
        roc_auc = 0.5
        ap = float(np.mean(bins_np))

    denom = max(1, total_samples)
    return {
        "loss": total_loss / denom,
        "bce": total_bce / denom,
        "mse": total_mse / denom,
        "mae": total_mae / denom,
        "kl": total_kl / denom,
        "roc_auc": roc_auc,
        "ap": ap,
        "batch_latency_ms": float(np.mean(batch_latencies_ms)) if batch_latencies_ms else 0.0,
    }


def train_seal(
    epochs: int = 25,
    num_train_graphs: int = 36,
    num_val_graphs: int = 12,
    batch_size: int = 128,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 64,
    num_layers: int = 3,
    sortpool_k: int = 16,
    k_hop: int = 2,
    seed: int = 42,
    save_dir: str = None,
    checkpoint_path: str = None,
    metrics_path: str = None,
    device=None,
    max_sub_nodes: int = 150,
    return_summary: bool = False,
):
    set_all_seeds(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = checkpoint_path if checkpoint_path is not None else os.path.join(save_dir, "best_seal_model.pt")
    metrics_json_path = metrics_path if metrics_path is not None else os.path.join(save_dir, "training_metrics.json")
    os.makedirs(os.path.dirname(os.path.abspath(best_ckpt_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(metrics_json_path)), exist_ok=True)

    train_wall_start = time.perf_counter()

    print("=" * 96)
    print(f"STANDALONE INDUCTIVE SEAL SPATIAL PREDICTION MODULE — TEACHER-STUDENT TRAINING (k_hop={k_hop})")
    print("=" * 96)
    print(f"Device              : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Architecture        : Inductive GINEConv ({num_layers} layers, hidden={hidden_dim}) + SortPooling (k={sortpool_k})")
    print(f"Subgraph Config     : k_hop={k_hop} | max_nodes_per_subgraph={max_sub_nodes if max_sub_nodes else 'Unbounded'}")
    print(f"Node / Edge Dim     : X={NODE_FEATURE_DIM} (DRNL + 2D Coords + Rel. Midpoint + Cluster + Multi-Scale IDW) | E={EDGE_FEATURE_DIM}")
    print(f"Observability Mask  : 40% - 90% edges masked as Unknown; 10% - 60% revealed as Clear/Blocked")
    print(f"Supervision Targets : Continuous GP Posterior q_e^GP (MSE + KL) + Realized Binary B_e (BCE)")
    print("-" * 96)

    train_data = build_procedural_dataset(
        num_graphs=num_train_graphs, k_hop=k_hop, seed=seed, desc="Train", max_sub_nodes=max_sub_nodes
    )
    val_data = build_procedural_dataset(
        num_graphs=num_val_graphs, k_hop=k_hop, seed=seed + 1000, desc="Val", max_sub_nodes=max_sub_nodes
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = SEALModel(
        node_in_dim=NODE_FEATURE_DIM,
        edge_in_dim=EDGE_FEATURE_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        sortpool_k=sortpool_k,
        dropout=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {num_params:,} | Train Subgraphs: {len(train_data):,} | Val Subgraphs: {len(val_data):,}")
    print("-" * 96)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    bce_weight = 0.35
    mse_weight = 0.45
    kl_weight = 0.20

    best_val_mse = float("inf")
    best_epoch = 0
    history = []

    header = (
        f"{'Epoch':>5} | {'TrBCE':>7} {'TrMSE':>7} {'TrKL':>7} | "
        f"{'ValBCE':>7} {'ValMSE':>7} {'ValKL':>7} | "
        f"{'ROC-AUC':>7} {'ValAP':>7} | {'BatchLat':>9} | {'Status':>8}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(1, epochs + 1):
        model.train()
        tr_bce_sum = 0.0
        tr_mse_sum = 0.0
        tr_kl_sum = 0.0
        tr_samples = 0
        batch_times_ms = []

        for batch in train_loader:
            batch = batch.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            optimizer.zero_grad()
            logits = model.forward_logits(batch)
            preds = torch.sigmoid(logits)

            y_prob = batch.y_prob.view(-1)
            y_bin = batch.y_bin.view(-1)
            bsz = y_prob.size(0)

            # Teacher-Student Hybrid Loss:
            #   1. MSE(q_hat_e, q_e^GP) + KL(Bern(q_e^GP) || Bern(q_hat_e)) against true GP field
            #   2. BCE(q_hat_e, B_e) against binary road realizations
            loss_bce = F.binary_cross_entropy_with_logits(logits, y_bin)
            loss_mse = F.mse_loss(preds, y_prob)
            loss_kl = compute_bernoulli_kl(preds, y_prob)
            loss = bce_weight * loss_bce + mse_weight * loss_mse + kl_weight * loss_kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_times_ms.append((time.perf_counter() - t0) * 1000.0)

            tr_bce_sum += float(loss_bce.item()) * bsz
            tr_mse_sum += float(loss_mse.item()) * bsz
            tr_kl_sum += float(loss_kl.item()) * bsz
            tr_samples += bsz

        scheduler.step()

        tr_bce = tr_bce_sum / max(1, tr_samples)
        tr_mse = tr_mse_sum / max(1, tr_samples)
        tr_kl = tr_kl_sum / max(1, tr_samples)

        val_metrics = evaluate_loader(
            model, val_loader, device,
            bce_weight=bce_weight, mse_weight=mse_weight, kl_weight=kl_weight,
        )

        is_best = val_metrics["mse"] < best_val_mse
        status_str = ""
        if is_best:
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            status_str = "*BEST*"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": {
                        "node_in_dim": NODE_FEATURE_DIM,
                        "edge_in_dim": EDGE_FEATURE_DIM,
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "sortpool_k": sortpool_k,
                        "k_hop": k_hop,
                        "dropout": 0.1,
                    },
                },
                best_ckpt_path,
            )

        epoch_record = {
            "epoch": epoch,
            "train_bce": tr_bce,
            "train_mse": tr_mse,
            "train_kl": tr_kl,
            "val_bce": val_metrics["bce"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_kl": val_metrics["kl"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_ap": val_metrics["ap"],
            "val_batch_latency_ms": val_metrics["batch_latency_ms"],
            "train_batch_latency_ms": float(np.mean(batch_times_ms)),
        }
        history.append(epoch_record)

        print(
            f"{epoch:5d} | {tr_bce:7.4f} {tr_mse:7.4f} {tr_kl:7.4f} | "
            f"{val_metrics['bce']:7.4f} {val_metrics['mse']:7.4f} {val_metrics['kl']:7.4f} | "
            f"{val_metrics['roc_auc']:7.4f} {val_metrics['ap']:7.4f} | "
            f"{val_metrics['batch_latency_ms']:7.2f}ms | {status_str:>8}"
        )

    train_wall_time_sec = time.perf_counter() - train_wall_start
    summary_payload = {
        "k_hop": k_hop,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "train_wall_time_sec": train_wall_time_sec,
        "best_checkpoint_path": best_ckpt_path,
        "history": history,
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("-" * len(header))
    print(
        f"Training Complete! Best Val GP MSE: {best_val_mse:.4f} (Epoch {best_epoch}) | "
        f"Wall-Clock Time: {train_wall_time_sec:.2f}s"
    )
    print(f"Saved Best Weights : {best_ckpt_path}")
    print(f"Saved Metrics Log  : {metrics_json_path}")
    if return_summary:
        return model, best_ckpt_path, summary_payload
    return model, best_ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone SEAL Spatial Blockage Predictor Training")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--train-graphs", type=int, default=36, help="Number of procedural training graphs")
    parser.add_argument("--val-graphs", type=int, default=12, help="Number of procedural validation graphs")
    parser.add_argument("--batch-size", type=int, default=128, help="PyG subgraph batch size")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Initial learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GINEConv hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of GINEConv layers")
    parser.add_argument("--sortpool-k", type=int, default=16, help="SortPooling k nodes")
    parser.add_argument("--k-hop", type=int, default=2, help="Enclosing subgraph hop radius")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_seal(
        epochs=args.epochs,
        num_train_graphs=args.train_graphs,
        num_val_graphs=args.val_graphs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        sortpool_k=args.sortpool_k,
        k_hop=args.k_hop,
        seed=args.seed,
    )
