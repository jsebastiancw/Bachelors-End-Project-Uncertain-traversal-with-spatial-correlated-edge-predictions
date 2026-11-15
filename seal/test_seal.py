"""
seal/test_seal.py
Deliverable 2: Standalone Testing & Comprehensive Diagnostic Script for SEAL.

Evaluates the frozen inductive SEAL model strictly standalone (zero-shot, without
retraining embeddings) on:
  1. Held-out procedural GBM/spatial graphs ('sparse', 'dense', 'multidistrict')
     across 'earthquake', 'tsunami', and 'flood' GP hazard fields.
  2. Canonical transportation benchmark networks ('SiouxFalls', 'Anaheim',
     'Friedrichshain', 'Winnipeg') loaded from `marvel_modernized/Networks/`
     and `sevac/data/` under simulated GP disasters.

Outputs a comprehensive diagnostic report covering:
  - ROC-AUC, Average Precision (AP), MSE & MAE vs. true GP probabilities q_e^{GP},
    and Brier Score (probability calibration).
  - Precision / Recall / F1 across classification thresholds tau in {0.3, 0.4, 0.5, 0.6, 0.7}.
  - Performance breakdown by hop distance (h = 1 vs. h = 2).
  - Performance breakdown by observability percentage (10%, 30%, 50% map revealed).
  - Average millisecond inference speed per query and per batched graph call.
"""

import os
import sys
import json
import time
import argparse
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
)

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
    SEALPredictor,
    predict_edges,
    prepare_spatial_graph,
    load_canonical_network,
    normalize_edge_status_dict,
)


def compute_metrics_bundle(
    y_pred: np.ndarray,
    y_gp: np.ndarray,
    y_bin: np.ndarray,
) -> Dict[str, float]:
    """
    Computes core continuous (vs. GP posterior q_e^{GP}) and binary (vs. realization B_e)
    evaluation metrics.
    """
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, 1.0)
    y_gp = np.clip(np.asarray(y_gp, dtype=np.float64), 0.0, 1.0)
    y_bin = np.asarray(y_bin, dtype=np.int64)

    mse = float(np.mean((y_pred - y_gp) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_gp)))
    brier = float(brier_score_loss(y_bin, y_pred))

    if len(np.unique(y_bin)) > 1:
        roc_auc = float(roc_auc_score(y_bin, y_pred))
        ap = float(average_precision_score(y_bin, y_pred))
    else:
        roc_auc = 0.5
        ap = float(np.mean(y_bin))

    return {
        "n_edges": int(len(y_pred)),
        "roc_auc": roc_auc,
        "ap": ap,
        "mse_gp": mse,
        "mae_gp": mae,
        "brier": brier,
    }


def compute_threshold_sweep(
    y_pred: np.ndarray,
    y_bin: np.ndarray,
    thresholds: Sequence[float] = (0.30, 0.40, 0.50, 0.60, 0.70),
) -> List[Dict[str, float]]:
    results = []
    for tau in thresholds:
        pred_cls = (y_pred >= tau).astype(np.int64)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_bin, pred_cls, average="binary", zero_division=0
        )
        acc = float(np.mean(pred_cls == y_bin))
        results.append({
            "threshold": float(tau),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "accuracy": acc,
        })
    return results


def compute_hop_distance_to_observed(
    G: nx.Graph,
    observed_edges: Sequence[Tuple[int, int]],
    target_edges: Sequence[Tuple[int, int]],
) -> List[int]:
    """
    Computes the topological hop distance h in {1, 2} from each target edge e = (u, v)
    to the nearest observed edge in `observed_edges`:
      - h = 1: Target edge shares at least one endpoint with an observed edge.
      - h = 2: Target edge is >= 2 hops away from any observed edge (neither u nor v
               has an incident observed edge).
    """
    obs_nodes = set()
    for u, v in observed_edges:
        obs_nodes.add(u)
        obs_nodes.add(v)

    hops = []
    for u, v in target_edges:
        if u in obs_nodes or v in obs_nodes:
            hops.append(1)
        else:
            hops.append(2)
    return hops


def evaluate_graph_scenarios(
    predictor: SEALPredictor,
    G_base: nx.Graph,
    graph_label: str,
    observability_rates: Sequence[float] = (0.10, 0.30, 0.50),
    disaster_types: Sequence[str] = ("earthquake", "tsunami", "flood"),
    k_hop_list: Sequence[int] = (1, 2),
    max_query_edges: int = 350,
    seed: int = 101,
    batch_size: int = 256,
    max_sub_nodes: Optional[int] = 150,
) -> List[Dict[str, Any]]:
    """
    Runs simulated GP disasters on `G_base`, masks edges according to
    `observability_rates` (10%, 30%, 50% revealed -> 90%, 70%, 50% unknown),
    and evaluates `predictor.predict_edges(...)` across hop distances in `k_hop_list`.
    """
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []

    for d_idx, d_type in enumerate(disaster_types):
        G = G_base.copy()
        boundary = ["right", "left", "top", "bottom"][d_idx % 4]
        G, _ = set_disaster_type(
            G,
            disaster_type=d_type,
            num_epicenters=2 if d_type == "earthquake" else 1,
            boundary=boundary,
            safe_y=-3.0,
            k_steepness=0.6,
            y_offset=1.5,
            max_blockage_prob=0.92,
            seed=seed + d_idx * 37,
        )
        G = prepare_spatial_graph(G, copy_graph=False)
        sample_gp_disaster_realization(G, max_resamples=300, allow_relaxation=True)

        all_edges = list(G.edges())
        n_total = len(all_edges)
        if n_total < 4:
            continue

        for obs_rate in observability_rates:
            n_obs = max(1, min(n_total - 1, int(round(obs_rate * n_total))))
            perm = rng.permutation(n_total)
            obs_idx_set = set(perm[:n_obs].tolist())

            edge_status_dict: Dict[Tuple[int, int], int] = {}
            observed_edges: List[Tuple[int, int]] = []
            unobserved_edges: List[Tuple[int, int]] = []

            for idx, e in enumerate(all_edges):
                u, v = e
                b_e = int(G.edges[u, v].get("is_blocked", 0))
                if idx in obs_idx_set:
                    edge_status_dict[e] = STATUS_BLOCKED if b_e == 1 else STATUS_CLEAR
                    observed_edges.append(e)
                else:
                    edge_status_dict[e] = STATUS_UNKNOWN
                    unobserved_edges.append(e)

            if len(unobserved_edges) > max_query_edges:
                sel = rng.choice(len(unobserved_edges), size=max_query_edges, replace=False)
                target_edges = [unobserved_edges[i] for i in sel]
            else:
                target_edges = unobserved_edges

            topo_hops = compute_hop_distance_to_observed(G, observed_edges, target_edges)
            y_gp = np.array([float(G.edges[u, v]["blockage_prob"]) for u, v in target_edges], dtype=np.float64)
            y_bin = np.array([int(G.edges[u, v]["is_blocked"]) for u, v in target_edges], dtype=np.int64)

            for k_hop in k_hop_list:
                if predictor.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                pred_dict = predictor.predict_edges(
                    nx_graph=G,
                    edge_status_dict=edge_status_dict,
                    target_edges=target_edges,
                    k_hop=k_hop,
                    batch_size=batch_size,
                    max_sub_nodes=max_sub_nodes,
                )

                if predictor.device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                ms_per_query = elapsed_ms / max(1, len(target_edges))

                y_pred = np.array([pred_dict[e] for e in target_edges], dtype=np.float64)

                records.append({
                    "graph_label": graph_label,
                    "disaster_type": d_type,
                    "obs_rate": float(obs_rate),
                    "k_hop": int(k_hop),
                    "y_pred": y_pred,
                    "y_gp": y_gp,
                    "y_bin": y_bin,
                    "topo_hops": np.asarray(topo_hops, dtype=np.int64),
                    "elapsed_ms": elapsed_ms,
                    "ms_per_query": ms_per_query,
                    "n_queries": len(target_edges),
                })

    return records


def aggregate_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}
    y_pred = np.concatenate([r["y_pred"] for r in records])
    y_gp = np.concatenate([r["y_gp"] for r in records])
    y_bin = np.concatenate([r["y_bin"] for r in records])
    total_ms = sum(r["elapsed_ms"] for r in records)
    total_q = sum(r["n_queries"] for r in records)

    bundle = compute_metrics_bundle(y_pred, y_gp, y_bin)
    bundle["ms_per_query"] = float(total_ms / max(1, total_q))
    bundle["avg_batch_ms"] = float(np.mean([r["elapsed_ms"] for r in records]))
    return bundle


def verify_downstream_interfaces(predictor: SEALPredictor) -> Dict[str, Any]:
    """
    Verifies zero-friction API compatibility with:
      1. Top-level `predict_edges(nx_graph, edge_status_dict, target_edges, k_hop=2)`
      2. MARVEL dynamic adjacency matrix & graph entropy H(G) update
      3. SEVAC state-action feature vector phi(s, a) builder
    """
    builder = SpatialGraphGenerator(n_nodes=35, d_min=0.04)
    G_undir, G_dir = builder.build_graph(regime="multidistrict", C=4)
    G_undir, _ = set_disaster_type(G_undir, disaster_type="earthquake", seed=777)
    sample_gp_disaster_realization(G_undir)

    all_edges = list(G_undir.edges())
    status_dict = {
        e: (STATUS_CLEAR if idx % 2 == 0 else STATUS_UNKNOWN)
        for idx, e in enumerate(all_edges)
    }
    target_edges = [e for e, st in status_dict.items() if st == STATUS_UNKNOWN]

    # 1. Top-level functional call
    preds = predict_edges(G_undir, status_dict, target_edges, k_hop=2, model=predictor)
    assert isinstance(preds, dict) and len(preds) == len(target_edges)
    assert all(0.0 <= v <= 1.0 for v in preds.values())

    # 2. MARVEL dynamic probability & entropy H(G) update check
    node_mapping = {n: i for i, n in enumerate(G_undir.nodes())}
    adj = torch.ones((G_undir.number_of_nodes(), G_undir.number_of_nodes()), dtype=torch.float32)
    updated_adj, entropy_val, _ = predictor.update_marvel_adj_and_entropy(
        G_undir, adj, node_mapping, status_dict, target_edges=target_edges, k_hop=2
    )

    # 3. SEVAC feature vector phi(s, a) check on directed graph
    phi_vec, full_q = predictor.build_sevac_edge_features(G_dir, status_dict, time_budget=15.0, k_hop=2)

    return {
        "num_target_edges_queried": len(preds),
        "marvel_entropy_H_G": float(entropy_val.item()),
        "sevac_phi_dim": int(phi_vec.shape[0]),
        "directed_edges_scored": len(full_q),
    }


def run_diagnostics(
    checkpoint_path: Optional[str] = None,
    num_procedural_per_regime: int = 4,
    seed: int = 2026,
    eval_k_hop: int = 2,
    device: Optional[str] = None,
    batch_size: int = 256,
    max_sub_nodes: Optional[int] = 150,
    report_txt_path: Optional[str] = None,
    results_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    predictor = SEALPredictor(checkpoint_path=checkpoint_path, device=device)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    if report_txt_path is None:
        report_txt_path = os.path.join(out_dir, "diagnostic_report.txt")
    if results_json_path is None:
        results_json_path = os.path.join(out_dir, "diagnostic_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(report_txt_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(results_json_path)), exist_ok=True)

    # Evaluate both h=1 and the target eval_k_hop (default 2) so Table D always has a 1-hop vs k-hop comparison
    k_hop_list = sorted(list({1, 2, int(eval_k_hop)})) if eval_k_hop != 2 else (1, 2)

    lines: List[str] = []

    def log(msg: str = ""):
        print(msg)
        lines.append(msg)

    log("=" * 100)
    log(f"DELIVERABLE 2: STANDALONE SEAL SPATIAL PREDICTION MODULE — DIAGNOSTIC & BENCHMARK REPORT (k_hop={eval_k_hop})")
    log("=" * 100)
    log(f"Device             : {predictor.device}")
    log(f"Checkpoint Loaded  : {predictor.checkpoint_path or 'Untrained/Initialized Weights'}")
    log(f"Subgraph Config    : eval_k_hop={eval_k_hop} | max_nodes_per_subgraph={max_sub_nodes if max_sub_nodes else 'Unbounded'}")
    log(f"Design Paradigm    : Purely Inductive (DRNL + 2D Coords + Rel. Midpoint + Cluster + Edge-Aware GINEConv + SortPool)")
    log("-" * 100)

    # -------------------------------------------------------------------------
    # 1. Held-Out Procedural Graphs Evaluation
    # -------------------------------------------------------------------------
    log("\n[1] HELD-OUT PROCEDURAL GRAPHS (Geometric Block Model & Spatial Regimes)")
    procedural_records: List[Dict[str, Any]] = []
    regimes = ["sparse", "dense", "multidistrict"]

    for r_idx, regime in enumerate(regimes):
        for g_i in range(num_procedural_per_regime):
            builder = SpatialGraphGenerator(n_nodes=45, d_min=0.04)
            G_proc, _ = builder.build_graph(regime=regime, C=4)
            recs = evaluate_graph_scenarios(
                predictor=predictor,
                G_base=G_proc,
                graph_label=f"Procedural-{regime}",
                observability_rates=(0.10, 0.30, 0.50),
                disaster_types=("earthquake", "tsunami", "flood"),
                k_hop_list=k_hop_list,
                seed=seed + r_idx * 100 + g_i * 17,
                batch_size=batch_size,
                max_sub_nodes=max_sub_nodes,
            )
            procedural_records.extend(recs)

    # -------------------------------------------------------------------------
    # 2. Canonical Real-World Road Networks Evaluation (Zero-Shot Inductive)
    # -------------------------------------------------------------------------
    log("[2] ZERO-SHOT CANONICAL BENCHMARK NETWORKS (SiouxFalls, Anaheim, Friedrichshain, Winnipeg)")
    canonical_names = ["SiouxFalls", "Anaheim", "Friedrichshain", "Winnipeg"]
    canonical_records: List[Dict[str, Any]] = []
    canonical_graph_stats: Dict[str, Dict[str, int]] = {}

    for c_idx, net_name in enumerate(canonical_names):
        G_canon = load_canonical_network(net_name, workspace_root=WORKSPACE_ROOT, seed=seed + c_idx)
        canonical_graph_stats[net_name] = {
            "nodes": G_canon.number_of_nodes(),
            "edges": G_canon.number_of_edges(),
        }
        recs = evaluate_graph_scenarios(
            predictor=predictor,
            G_base=G_canon,
            graph_label=net_name,
            observability_rates=(0.10, 0.30, 0.50),
            disaster_types=("earthquake", "tsunami", "flood"),
            k_hop_list=k_hop_list,
            max_query_edges=300,
            seed=seed + 500 + c_idx * 53,
            batch_size=batch_size,
            max_sub_nodes=max_sub_nodes,
        )
        canonical_records.extend(recs)

    all_records = procedural_records + canonical_records
    all_target_k = [r for r in all_records if r["k_hop"] == eval_k_hop]

    # -------------------------------------------------------------------------
    # Table A: Performance by Network & Regime (k_hop = eval_k_hop)
    # -------------------------------------------------------------------------
    log("\n" + "=" * 100)
    log(f"TABLE A: PERFORMANCE BY NETWORK TOPOLOGY (k_hop = {eval_k_hop}, Aggregated Across Disasters & Observability)")
    log("=" * 100)
    header_a = (
        f"{'Network / Regime':<24} | {'Queries':>7} | {'ROC-AUC':>7} | {'AP':>7} | "
        f"{'MSE(GP)':>8} | {'MAE(GP)':>8} | {'Brier':>7} | {'Latency/Query':>13}"
    )
    log(header_a)
    log("-" * len(header_a))

    group_labels = [
        "Procedural-sparse",
        "Procedural-dense",
        "Procedural-multidistrict",
        "SiouxFalls",
        "Anaheim",
        "Friedrichshain",
        "Winnipeg",
    ]
    per_network_summary: Dict[str, Any] = {}
    for glabel in group_labels:
        sub_recs = [r for r in all_target_k if r["graph_label"] == glabel]
        agg = aggregate_records(sub_recs)
        per_network_summary[glabel] = agg
        log(
            f"{glabel:<24} | {agg['n_edges']:7d} | {agg['roc_auc']:7.4f} | {agg['ap']:7.4f} | "
            f"{agg['mse_gp']:8.4f} | {agg['mae_gp']:8.4f} | {agg['brier']:7.4f} | "
            f"{agg['ms_per_query']:10.3f} ms"
        )

    log("-" * len(header_a))
    proc_k_agg = aggregate_records([r for r in procedural_records if r["k_hop"] == eval_k_hop])
    canon_k_agg = aggregate_records([r for r in canonical_records if r["k_hop"] == eval_k_hop])
    overall_k_agg = aggregate_records(all_target_k)

    log(
        f"{'ALL PROCEDURAL (Held-Out)':<24} | {proc_k_agg['n_edges']:7d} | {proc_k_agg['roc_auc']:7.4f} | "
        f"{proc_k_agg['ap']:7.4f} | {proc_k_agg['mse_gp']:8.4f} | {proc_k_agg['mae_gp']:8.4f} | "
        f"{proc_k_agg['brier']:7.4f} | {proc_k_agg['ms_per_query']:10.3f} ms"
    )
    log(
        f"{'ALL CANONICAL (Zero-Shot)':<24} | {canon_k_agg['n_edges']:7d} | {canon_k_agg['roc_auc']:7.4f} | "
        f"{canon_k_agg['ap']:7.4f} | {canon_k_agg['mse_gp']:8.4f} | {canon_k_agg['mae_gp']:8.4f} | "
        f"{canon_k_agg['brier']:7.4f} | {canon_k_agg['ms_per_query']:10.3f} ms"
    )
    log(
        f"{'OVERALL COMBINED':<24} | {overall_k_agg['n_edges']:7d} | {overall_k_agg['roc_auc']:7.4f} | "
        f"{overall_k_agg['ap']:7.4f} | {overall_k_agg['mse_gp']:8.4f} | {overall_k_agg['mae_gp']:8.4f} | "
        f"{overall_k_agg['brier']:7.4f} | {overall_k_agg['ms_per_query']:10.3f} ms"
    )

    # -------------------------------------------------------------------------
    # Table B: Breakdown by Disaster Topology ('earthquake', 'tsunami', 'flood')
    # -------------------------------------------------------------------------
    log("\n" + "=" * 100)
    log(f"TABLE B: BREAKDOWN BY GP DISASTER REGIME (k_hop = {eval_k_hop})")
    log("=" * 100)
    log(header_a)
    log("-" * len(header_a))
    disaster_summary: Dict[str, Any] = {}
    for dtype in ("earthquake", "tsunami", "flood"):
        drecs = [r for r in all_target_k if r["disaster_type"] == dtype]
        agg = aggregate_records(drecs)
        disaster_summary[dtype] = agg
        log(
            f"{dtype.capitalize():<24} | {agg['n_edges']:7d} | {agg['roc_auc']:7.4f} | {agg['ap']:7.4f} | "
            f"{agg['mse_gp']:8.4f} | {agg['mae_gp']:8.4f} | {agg['brier']:7.4f} | "
            f"{agg['ms_per_query']:10.3f} ms"
        )

    # -------------------------------------------------------------------------
    # Table C: Breakdown by Observability Percentage (10%, 30%, 50% Map Revealed)
    # -------------------------------------------------------------------------
    log("\n" + "=" * 100)
    log(f"TABLE C: BREAKDOWN BY OBSERVABILITY PERCENTAGE (10%, 30%, 50% Map Revealed, k_hop = {eval_k_hop})")
    log("=" * 100)
    header_c = (
        f"{'Map Revealed (%)':<24} | {'Queries':>7} | {'ROC-AUC':>7} | {'AP':>7} | "
        f"{'MSE(GP)':>8} | {'MAE(GP)':>8} | {'Brier':>7} | {'Latency/Query':>13}"
    )
    log(header_c)
    log("-" * len(header_c))
    obs_summary: Dict[str, Any] = {}
    for obs_rate in (0.10, 0.30, 0.50):
        orecs = [r for r in all_target_k if abs(r["obs_rate"] - obs_rate) < 1e-4]
        agg = aggregate_records(orecs)
        label = f"{int(obs_rate * 100)}% Revealed ({int((1 - obs_rate) * 100)}% Unk)"
        obs_summary[f"{int(obs_rate * 100)}pct"] = agg
        log(
            f"{label:<24} | {agg['n_edges']:7d} | {agg['roc_auc']:7.4f} | {agg['ap']:7.4f} | "
            f"{agg['mse_gp']:8.4f} | {agg['mae_gp']:8.4f} | {agg['brier']:7.4f} | "
            f"{agg['ms_per_query']:10.3f} ms"
        )

    # -------------------------------------------------------------------------
    # Table D: Breakdown by Hop Distance (h = 1 vs. h = 2 / eval_k_hop)
    # -------------------------------------------------------------------------
    log("\n" + "=" * 100)
    log("TABLE D: BREAKDOWN BY HOP DISTANCE")
    log("=" * 100)
    log(header_a)
    log("-" * len(header_a))

    hop_breakdown_dict: Dict[str, Any] = {}
    for kh in k_hop_list:
        kh_agg = aggregate_records([r for r in all_records if r["k_hop"] == kh])
        hop_breakdown_dict[f"subgraph_radius_h{kh}"] = kh_agg
        log(
            f"{f'Subgraph Radius h={kh}':<24} | {kh_agg['n_edges']:7d} | {kh_agg['roc_auc']:7.4f} | {kh_agg['ap']:7.4f} | "
            f"{kh_agg['mse_gp']:8.4f} | {kh_agg['mae_gp']:8.4f} | {kh_agg['brier']:7.4f} | "
            f"{kh_agg['ms_per_query']:10.3f} ms"
        )

    # D.2: Topological Hop Distance to Nearest Observed Edge (h=1 adjacent vs h=2 multi-hop)
    y_pred_all = np.concatenate([r["y_pred"] for r in all_target_k])
    y_gp_all = np.concatenate([r["y_gp"] for r in all_target_k])
    y_bin_all = np.concatenate([r["y_bin"] for r in all_target_k])
    topo_hops_all = np.concatenate([r["topo_hops"] for r in all_target_k])

    mask_h1 = topo_hops_all == 1
    mask_h2 = topo_hops_all == 2
    topo_h1_metrics = compute_metrics_bundle(y_pred_all[mask_h1], y_gp_all[mask_h1], y_bin_all[mask_h1])
    topo_h2_metrics = compute_metrics_bundle(y_pred_all[mask_h2], y_gp_all[mask_h2], y_bin_all[mask_h2])
    hop_breakdown_dict["obs_distance_h1"] = topo_h1_metrics
    hop_breakdown_dict["obs_distance_h2"] = topo_h2_metrics

    log(
        f"{'Obs-Distance h=1 (Adj.)':<24} | {topo_h1_metrics['n_edges']:7d} | {topo_h1_metrics['roc_auc']:7.4f} | "
        f"{topo_h1_metrics['ap']:7.4f} | {topo_h1_metrics['mse_gp']:8.4f} | {topo_h1_metrics['mae_gp']:8.4f} | "
        f"{topo_h1_metrics['brier']:7.4f} | {overall_k_agg['ms_per_query']:10.3f} ms"
    )
    log(
        f"{'Obs-Distance h=2 (Far)':<24} | {topo_h2_metrics['n_edges']:7d} | {topo_h2_metrics['roc_auc']:7.4f} | "
        f"{topo_h2_metrics['ap']:7.4f} | {topo_h2_metrics['mse_gp']:8.4f} | {topo_h2_metrics['mae_gp']:8.4f} | "
        f"{topo_h2_metrics['brier']:7.4f} | {overall_k_agg['ms_per_query']:10.3f} ms"
    )

    # -------------------------------------------------------------------------
    # Table E: Precision / Recall / F1 Across Decision Thresholds
    # -------------------------------------------------------------------------
    log("\n" + "=" * 100)
    log(f"TABLE E: PRECISION / RECALL / F1 ACROSS CLASSIFICATION THRESHOLDS (k_hop = {eval_k_hop})")
    log("=" * 100)
    header_e = f"{'Threshold (tau)':<18} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Accuracy':>10}"
    log(header_e)
    log("-" * len(header_e))
    threshold_results = compute_threshold_sweep(y_pred_all, y_bin_all)
    for row in threshold_results:
        log(
            f"{row['threshold']:<18.2f} | {row['precision']:10.4f} | {row['recall']:10.4f} | "
            f"{row['f1']:10.4f} | {row['accuracy']:10.4f}"
        )

    # -------------------------------------------------------------------------
    # Downstream API Integration Sanity Check
    # -------------------------------------------------------------------------
    interface_check = verify_downstream_interfaces(predictor)
    log("\n" + "=" * 100)
    log("DOWNSTREAM INTERFACE VERIFICATION (MARVEL & SEVAC PLUG-AND-PLAY)")
    log("=" * 100)
    log(f"  - predict_edges(nx_graph, edge_status_dict, target_edges, k_hop=2) -> {interface_check['num_target_edges_queried']} edges predicted")
    log(f"  - MARVEL dynamic entropy H(G) computed on predicted adjacency      -> {interface_check['marvel_entropy_H_G']:.4f} nats")
    log(f"  - SEVAC state-action feature phi(s, a) dimension (3 * |E_dir|)     -> {interface_check['sevac_phi_dim']} dims")
    log("=" * 100)

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    results_payload = {
        "eval_k_hop": int(eval_k_hop),
        "overall": overall_k_agg,
        "procedural": proc_k_agg,
        "canonical": canon_k_agg,
        "overall_k2": overall_k_agg,
        "procedural_k2": proc_k_agg,
        "canonical_k2": canon_k_agg,
        "per_network": per_network_summary,
        "per_disaster": disaster_summary,
        "per_observability": obs_summary,
        "hop_breakdown": hop_breakdown_dict,
        "threshold_sweep": threshold_results,
        "canonical_graph_stats": canonical_graph_stats,
        "interface_verification": interface_check,
        "report_txt_path": report_txt_path,
        "results_json_path": results_json_path,
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nSaved Diagnostic Text Report : {report_txt_path}")
    print(f"Saved Diagnostic JSON Metrics: {results_json_path}")
    return results_payload


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone SEAL Diagnostic & Benchmark Script")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained SEAL checkpoint")
    parser.add_argument("--proc-per-regime", type=int, default=4, help="Held-out procedural graphs per regime")
    parser.add_argument("--k-hop", type=int, default=2, help="Primary enclosing subgraph hop radius to evaluate")
    parser.add_argument("--seed", type=int, default=2026, help="Evaluation random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diagnostics(
        checkpoint_path=args.checkpoint,
        num_procedural_per_regime=args.proc_per_regime,
        eval_k_hop=args.k_hop,
        seed=args.seed,
    )
