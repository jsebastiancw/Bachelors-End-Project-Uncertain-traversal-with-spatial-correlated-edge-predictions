"""
seal/run_seal.py
Unified Training & Evaluation Orchestration Runner with Multi-k-Hop Sweep Support.

Automates end-to-end standalone SEAL training (`seal/train_seal.py`) and zero-shot
diagnostic evaluation (`seal/test_seal.py`) across one or more enclosing subgraph
radii `k_hop` (e.g., `--k_hops 1 2 3 4`).

Features:
  - Flexible CLI (`--k_hops`, `--mode {train,eval,both}`, `--epochs`, `--batch_size`,
    `--lr`, `--num_train_graphs`, `--seed`, `--device`, `--max_nodes_per_subgraph`).
  - Isolated per-k checkpoints (`seal/checkpoints/best_seal_k{k}.pt`) and diagnostic
    reports (`seal/results/eval_report_k{k}.txt`, `seal/results/eval_results_k{k}.json`).
  - Hub-node & GPU memory safeguards for k >= 3 (subgraph node capping, automatic
    batch-size scaling, CUDA cache clearing, and graceful OOM fallback/recovery).
  - Side-by-side Cross-k Comparative Summary Table (Overall metrics, Flood vs.
    Earthquake vs. Tsunami breakdown, and Training Wall-Clock Time).
"""

import os
import sys
import csv
import json
import time
import traceback
import argparse
from typing import Any, Dict, List, Optional

import torch

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from seal.train_seal import train_seal
from seal.test_seal import run_diagnostics


def _resolve_effective_batch_size(base_batch_size: int, k_hop: int, auto_scale: bool = True) -> int:
    """
    Scales down batch size for larger k-hop radii (k >= 3) to prevent GPU/RAM spikes
    when enclosing subgraphs expand across high-degree hub intersections.
    """
    if not auto_scale or k_hop <= 2:
        return max(8, int(base_batch_size))
    scale_factor = 2 ** (k_hop - 2)  # k=3 -> /2, k=4 -> /4
    return max(16, int(base_batch_size) // scale_factor)


def _clear_Accelerator_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_cross_k_comparison_table(
    k_hops: List[int],
    sweep_records: Dict[int, Dict[str, Any]],
    summary_txt_path: Optional[str] = None,
) -> str:
    """
    Formats and prints the side-by-side cross-k comparative summary tables:
      1. Overall Combined Metrics + Latency/Query + Training Wall-Clock Time
      2. Disaster Regime Breakdown (Flood vs. Earthquake vs. Tsunami: ROC-AUC, AP, MSE)
      3. Procedural vs. Zero-Shot Canonical Network Generalization
    """
    lines: List[str] = []

    def out(line: str = ""):
        print(line)
        lines.append(line)

    out("\n" + "=" * 114)
    out("CROSS-k COMPARATIVE SUMMARY TABLE — STANDALONE SEAL SPATIAL PREDICTION SWEEP")
    out("=" * 114)

    # -------------------------------------------------------------------------
    # Table 1: Overall Combined Performance & Compute Cost per k-hop
    # -------------------------------------------------------------------------
    out("\n[A] OVERALL COMBINED PERFORMANCE & COMPUTATIONAL LATENCY ACROSS k-HOP RADII")
    header_1 = (
        f"{'k-Hop':>5} | {'Status':<8} | {'ROC-AUC':>8} | {'AP':>8} | "
        f"{'MSE(GP)':>8} | {'MAE(GP)':>8} | {'Brier':>8} | "
        f"{'Latency/Query':>14} | {'Train Time':>12}"
    )
    out(header_1)
    out("-" * len(header_1))

    for k in k_hops:
        rec = sweep_records.get(k, {})
        status = rec.get("status", "SKIPPED")
        train_sec = rec.get("train_wall_time_sec", None)
        train_str = f"{train_sec:9.2f} s" if isinstance(train_sec, (int, float)) else f"{'N/A':>11}"

        eval_res = rec.get("eval_results")
        if eval_res and "overall" in eval_res and eval_res["overall"]:
            ov = eval_res["overall"]
            out(
                f"{f'k={k}':>5} | {status:<8} | {ov['roc_auc']:8.4f} | {ov['ap']:8.4f} | "
                f"{ov['mse_gp']:8.4f} | {ov['mae_gp']:8.4f} | {ov['brier']:8.4f} | "
                f"{ov['ms_per_query']:11.3f} ms | {train_str:>12}"
            )
        else:
            err_note = rec.get("error", "No eval data")[:42]
            out(
                f"{f'k={k}':>5} | {status:<8} | {'--':>8} | {'--':>8} | "
                f"{'--':>8} | {'--':>8} | {'--':>8} | "
                f"{err_note:>14} | {train_str:>12}"
            )

    # -------------------------------------------------------------------------
    # Table 2: Disaster Breakdown (Flood vs. Earthquake vs. Tsunami)
    # -------------------------------------------------------------------------
    out("\n[B] DISASTER TOPOLOGY BREAKDOWN ACROSS k-HOP (Flood Meandering River vs. Earthquake vs. Tsunami)")
    header_2 = (
        f"{'k-Hop':>5} | {'Flood ROC-AUC':>13} {'Flood AP':>9} {'Flood MSE':>10} | "
        f"{'Quake ROC-AUC':>13} {'Quake AP':>9} {'Quake MSE':>10} | "
        f"{'Tsunami ROC-AUC':>15} {'Tsunami AP':>10} {'Tsunami MSE':>11}"
    )
    out(header_2)
    out("-" * len(header_2))

    for k in k_hops:
        rec = sweep_records.get(k, {})
        eval_res = rec.get("eval_results")
        if eval_res and "per_disaster" in eval_res:
            pd_map = eval_res["per_disaster"]
            fl = pd_map.get("flood", {})
            eq = pd_map.get("earthquake", {})
            ts = pd_map.get("tsunami", {})
            if fl and eq and ts:
                out(
                    f"{f'k={k}':>5} | "
                    f"{fl['roc_auc']:13.4f} {fl['ap']:9.4f} {fl['mse_gp']:10.4f} | "
                    f"{eq['roc_auc']:13.4f} {eq['ap']:9.4f} {eq['mse_gp']:10.4f} | "
                    f"{ts['roc_auc']:15.4f} {ts['ap']:10.4f} {ts['mse_gp']:11.4f}"
                )
            else:
                out(f"{f'k={k}':>5} | {'Incomplete disaster metrics':<90}")
        else:
            out(f"{f'k={k}':>5} | {'Not evaluated':<90}")

    # -------------------------------------------------------------------------
    # Table 3: Procedural (Held-Out) vs. Canonical (Zero-Shot) Generalization
    # -------------------------------------------------------------------------
    out("\n[C] HELD-OUT PROCEDURAL vs. ZERO-SHOT CANONICAL NETWORKS (SiouxFalls, Anaheim, Friedrichshain, Winnipeg)")
    header_3 = (
        f"{'k-Hop':>5} | {'Proc ROC-AUC':>12} {'Proc AP':>9} {'Proc MSE':>9} | "
        f"{'Canon ROC-AUC':>13} {'Canon AP':>9} {'Canon MSE':>10} | "
        f"{'Checkpoint Path':<36}"
    )
    out(header_3)
    out("-" * len(header_3))

    for k in k_hops:
        rec = sweep_records.get(k, {})
        eval_res = rec.get("eval_results")
        ckpt_rel = os.path.relpath(rec.get("checkpoint_path", ""), WORKSPACE_ROOT) if rec.get("checkpoint_path") else "N/A"
        if eval_res and "procedural" in eval_res and "canonical" in eval_res:
            pr = eval_res["procedural"]
            cn = eval_res["canonical"]
            out(
                f"{f'k={k}':>5} | "
                f"{pr['roc_auc']:12.4f} {pr['ap']:9.4f} {pr['mse_gp']:9.4f} | "
                f"{cn['roc_auc']:13.4f} {cn['ap']:9.4f} {cn['mse_gp']:10.4f} | "
                f"{ckpt_rel:<36}"
            )
        else:
            out(f"{f'k={k}':>5} | {'Not evaluated':<68} | {ckpt_rel:<36}")

    out("=" * 114)

    full_text = "\n".join(lines) + "\n"
    if summary_txt_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(summary_txt_path)), exist_ok=True)
        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)
    return full_text


def save_sweep_csv_and_json(
    k_hops: List[int],
    sweep_records: Dict[int, Dict[str, Any]],
    csv_path: str,
    json_path: str,
):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)

    fieldnames = [
        "k_hop",
        "status",
        "train_wall_time_sec",
        "best_epoch",
        "best_val_mse",
        "overall_roc_auc",
        "overall_ap",
        "overall_mse_gp",
        "overall_mae_gp",
        "overall_brier",
        "ms_per_query",
        "flood_roc_auc",
        "flood_ap",
        "flood_mse_gp",
        "earthquake_roc_auc",
        "earthquake_ap",
        "earthquake_mse_gp",
        "tsunami_roc_auc",
        "tsunami_ap",
        "tsunami_mse_gp",
        "procedural_roc_auc",
        "procedural_ap",
        "procedural_mse_gp",
        "canonical_roc_auc",
        "canonical_ap",
        "canonical_mse_gp",
        "checkpoint_path",
        "eval_report_path",
    ]

    rows = []
    for k in k_hops:
        rec = sweep_records.get(k, {})
        ev = rec.get("eval_results") or {}
        ov = ev.get("overall") or {}
        pd_map = ev.get("per_disaster") or {}
        fl = pd_map.get("flood") or {}
        eq = pd_map.get("earthquake") or {}
        ts = pd_map.get("tsunami") or {}
        pr = ev.get("procedural") or {}
        cn = ev.get("canonical") or {}

        row = {
            "k_hop": k,
            "status": rec.get("status", "UNKNOWN"),
            "train_wall_time_sec": rec.get("train_wall_time_sec"),
            "best_epoch": rec.get("best_epoch"),
            "best_val_mse": rec.get("best_val_mse"),
            "overall_roc_auc": ov.get("roc_auc"),
            "overall_ap": ov.get("ap"),
            "overall_mse_gp": ov.get("mse_gp"),
            "overall_mae_gp": ov.get("mae_gp"),
            "overall_brier": ov.get("brier"),
            "ms_per_query": ov.get("ms_per_query"),
            "flood_roc_auc": fl.get("roc_auc"),
            "flood_ap": fl.get("ap"),
            "flood_mse_gp": fl.get("mse_gp"),
            "earthquake_roc_auc": eq.get("roc_auc"),
            "earthquake_ap": eq.get("ap"),
            "earthquake_mse_gp": eq.get("mse_gp"),
            "tsunami_roc_auc": ts.get("roc_auc"),
            "tsunami_ap": ts.get("ap"),
            "tsunami_mse_gp": ts.get("mse_gp"),
            "procedural_roc_auc": pr.get("roc_auc"),
            "procedural_ap": pr.get("ap"),
            "procedural_mse_gp": pr.get("mse_gp"),
            "canonical_roc_auc": cn.get("roc_auc"),
            "canonical_ap": cn.get("ap"),
            "canonical_mse_gp": cn.get("mse_gp"),
            "checkpoint_path": rec.get("checkpoint_path"),
            "eval_report_path": rec.get("eval_report_path"),
        }
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "k_hops": k_hops,
                "runs": {str(k): sweep_records[k] for k in k_hops if k in sweep_records},
            },
            f,
            indent=2,
        )


def run_seal_sweep(
    k_hops: List[int],
    mode: str = "both",
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1.5e-3,
    weight_decay: float = 1e-4,
    num_train_graphs: int = 36,
    num_val_graphs: int = 12,
    proc_per_regime: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 3,
    sortpool_k: int = 16,
    max_nodes_per_subgraph: Optional[int] = 150,
    auto_scale_batch: bool = True,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[int, Dict[str, Any]]:
    seal_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(seal_dir, "checkpoints")
    results_dir = os.path.join(seal_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Normalize max_nodes_per_subgraph (0 or negative means None / unbounded)
    eff_max_nodes = max_nodes_per_subgraph if (max_nodes_per_subgraph is not None and max_nodes_per_subgraph > 0) else None

    sweep_records: Dict[int, Dict[str, Any]] = {}

    print("#" * 114)
    print("UNIFIED SEAL ORCHESTRATION RUNNER — MULTI-k-HOP SPATIAL PREDICTION SWEEP")
    print("#" * 114)
    print(f"Requested k_hops         : {k_hops}")
    print(f"Execution Mode           : {mode.upper()}")
    print(f"Training Hyperparameters : epochs={epochs}, base_batch_size={batch_size}, lr={lr}, train_graphs={num_train_graphs}, val_graphs={num_val_graphs}")
    print(f"Hub / Memory Safeguards  : max_nodes_per_subgraph={eff_max_nodes if eff_max_nodes else 'Unbounded'}, auto_scale_batch={auto_scale_batch}")
    print(f"Checkpoints Directory    : {ckpt_dir}")
    print(f"Diagnostic Results Dir   : {results_dir}")
    print("#" * 114)

    for idx, k in enumerate(k_hops, start=1):
        ckpt_path = os.path.join(ckpt_dir, f"best_seal_k{k}.pt")
        train_metrics_path = os.path.join(ckpt_dir, f"training_metrics_k{k}.json")
        eval_report_txt = os.path.join(results_dir, f"eval_report_k{k}.txt")
        eval_results_json = os.path.join(results_dir, f"eval_results_k{k}.json")

        eff_batch_size = _resolve_effective_batch_size(batch_size, k, auto_scale=auto_scale_batch)

        print(f"\n>>> [{idx}/{len(k_hops)}] STARTING RUN FOR k_hop = {k} (effective_batch_size={eff_batch_size}, max_sub_nodes={eff_max_nodes}) <<<")

        run_info: Dict[str, Any] = {
            "k_hop": int(k),
            "status": "OK",
            "checkpoint_path": ckpt_path,
            "train_metrics_path": train_metrics_path,
            "eval_report_path": eval_report_txt,
            "eval_json_path": eval_results_json,
            "effective_batch_size": eff_batch_size,
            "max_nodes_per_subgraph": eff_max_nodes,
            "train_wall_time_sec": None,
            "best_epoch": None,
            "best_val_mse": None,
            "eval_results": None,
            "error": None,
        }

        # ---------------------------------------------------------------------
        # Phase 1: Training (if mode in {'train', 'both'})
        # ---------------------------------------------------------------------
        if mode in ("train", "both"):
            _clear_Accelerator_memory()
            try:
                _, saved_ckpt, train_summary = train_seal(
                    epochs=epochs,
                    num_train_graphs=num_train_graphs,
                    num_val_graphs=num_val_graphs,
                    batch_size=eff_batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    sortpool_k=sortpool_k,
                    k_hop=k,
                    seed=seed,
                    save_dir=ckpt_dir,
                    checkpoint_path=ckpt_path,
                    metrics_path=train_metrics_path,
                    device=device,
                    max_sub_nodes=eff_max_nodes,
                    return_summary=True,
                )
                run_info["checkpoint_path"] = saved_ckpt
                run_info["train_wall_time_sec"] = train_summary.get("train_wall_time_sec")
                run_info["best_epoch"] = train_summary.get("best_epoch")
                run_info["best_val_mse"] = train_summary.get("best_val_mse")

            except RuntimeError as err:
                is_oom = "out of memory" in str(err).lower()
                if is_oom:
                    print(f"[WARNING] GPU Out-Of-Memory encountered during training at k={k}. Retrying with reduced batch & subgraph cap...")
                    _clear_Accelerator_memory()
                    try:
                        fallback_bs = max(8, eff_batch_size // 2)
                        fallback_nodes = min(eff_max_nodes or 100, 100)
                        _, saved_ckpt, train_summary = train_seal(
                            epochs=epochs,
                            num_train_graphs=num_train_graphs,
                            num_val_graphs=num_val_graphs,
                            batch_size=fallback_bs,
                            lr=lr,
                            weight_decay=weight_decay,
                            hidden_dim=hidden_dim,
                            num_layers=num_layers,
                            sortpool_k=sortpool_k,
                            k_hop=k,
                            seed=seed,
                            save_dir=ckpt_dir,
                            checkpoint_path=ckpt_path,
                            metrics_path=train_metrics_path,
                            device=device,
                            max_sub_nodes=fallback_nodes,
                            return_summary=True,
                        )
                        run_info["checkpoint_path"] = saved_ckpt
                        run_info["train_wall_time_sec"] = train_summary.get("train_wall_time_sec")
                        run_info["best_epoch"] = train_summary.get("best_epoch")
                        run_info["best_val_mse"] = train_summary.get("best_val_mse")
                        run_info["status"] = "OK(Retry)"
                    except Exception as retry_err:
                        run_info["status"] = "OOM_FAIL"
                        run_info["error"] = str(retry_err)
                        print(f"[ERROR] Training retry failed for k={k}: {retry_err}")
                        sweep_records[k] = run_info
                        continue
                else:
                    run_info["status"] = "TRAIN_ERR"
                    run_info["error"] = str(err)
                    print(f"[ERROR] Training failed for k={k}: {err}")
                    traceback.print_exc()
                    sweep_records[k] = run_info
                    continue
            except Exception as err:
                run_info["status"] = "TRAIN_ERR"
                run_info["error"] = str(err)
                print(f"[ERROR] Unexpected error during training for k={k}: {err}")
                traceback.print_exc()
                sweep_records[k] = run_info
                continue
        else:
            # In 'eval'-only mode, load existing training wall-clock metadata if available
            if os.path.exists(train_metrics_path):
                try:
                    with open(train_metrics_path, "r", encoding="utf-8") as f:
                        t_meta = json.load(f)
                    run_info["train_wall_time_sec"] = t_meta.get("train_wall_time_sec")
                    run_info["best_epoch"] = t_meta.get("best_epoch")
                    run_info["best_val_mse"] = t_meta.get("best_val_mse")
                except Exception:
                    pass
            if not os.path.exists(ckpt_path):
                fallback_default = os.path.join(ckpt_dir, "best_seal_model.pt")
                if os.path.exists(fallback_default):
                    print(f"[INFO] {ckpt_path} not found; falling back to {fallback_default} for k={k} evaluation.")
                    ckpt_path = fallback_default
                    run_info["checkpoint_path"] = ckpt_path
                else:
                    print(f"[WARNING] No trained checkpoint found at {ckpt_path}; evaluating with initialized weights.")

        # ---------------------------------------------------------------------
        # Phase 2: Evaluation (if mode in {'eval', 'both'})
        # ---------------------------------------------------------------------
        if mode in ("eval", "both"):
            _clear_Accelerator_memory()
            try:
                eval_batch_size = max(16, eff_batch_size * 2)
                eval_results = run_diagnostics(
                    checkpoint_path=ckpt_path,
                    num_procedural_per_regime=proc_per_regime,
                    seed=seed + 2000,
                    eval_k_hop=k,
                    device=device,
                    batch_size=eval_batch_size,
                    max_sub_nodes=eff_max_nodes,
                    report_txt_path=eval_report_txt,
                    results_json_path=eval_results_json,
                )
                run_info["eval_results"] = eval_results
            except RuntimeError as err:
                is_oom = "out of memory" in str(err).lower()
                if is_oom:
                    print(f"[WARNING] GPU Out-Of-Memory during evaluation at k={k}. Retrying with smaller batch & subgraph cap...")
                    _clear_Accelerator_memory()
                    try:
                        eval_results = run_diagnostics(
                            checkpoint_path=ckpt_path,
                            num_procedural_per_regime=proc_per_regime,
                            seed=seed + 2000,
                            eval_k_hop=k,
                            device=device,
                            batch_size=32,
                            max_sub_nodes=min(eff_max_nodes or 100, 100),
                            report_txt_path=eval_report_txt,
                            results_json_path=eval_results_json,
                        )
                        run_info["eval_results"] = eval_results
                        run_info["status"] = "OK(Retry)"
                    except Exception as retry_err:
                        run_info["status"] = "EVAL_OOM"
                        run_info["error"] = str(retry_err)
                        print(f"[ERROR] Evaluation retry failed for k={k}: {retry_err}")
                else:
                    run_info["status"] = "EVAL_ERR"
                    run_info["error"] = str(err)
                    print(f"[ERROR] Evaluation failed for k={k}: {err}")
                    traceback.print_exc()
            except Exception as err:
                run_info["status"] = "EVAL_ERR"
                run_info["error"] = str(err)
                print(f"[ERROR] Unexpected error during evaluation for k={k}: {err}")
                traceback.print_exc()

        sweep_records[k] = run_info
        _clear_Accelerator_memory()

    # -------------------------------------------------------------------------
    # Phase 3: Cross-k Comparative Summary Table & Consolidated Artifacts
    # -------------------------------------------------------------------------
    summary_txt_path = os.path.join(results_dir, "k_hop_sweep_summary.txt")
    summary_csv_path = os.path.join(results_dir, "k_hop_sweep_summary.csv")
    summary_json_path = os.path.join(results_dir, "k_hop_sweep_summary.json")

    if mode in ("eval", "both") or len(k_hops) >= 1:
        print_cross_k_comparison_table(k_hops, sweep_records, summary_txt_path=summary_txt_path)
        save_sweep_csv_and_json(k_hops, sweep_records, csv_path=summary_csv_path, json_path=summary_json_path)
        print(f"Saved Sweep Summary Table (TXT) : {summary_txt_path}")
        print(f"Saved Sweep Summary Table (CSV) : {summary_csv_path}")
        print(f"Saved Sweep Summary Data (JSON) : {summary_json_path}")

    return sweep_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified SEAL Training & Evaluation Runner with k-Hop Hyperparameter Sweep"
    )
    parser.add_argument(
        "--k_hops",
        "--k-hops",
        dest="k_hops",
        type=int,
        nargs="+",
        default=[2],
        help="One or more k-hop subgraph radii to train/evaluate (e.g. --k_hops 1 2 3 4; default: 2)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "both"],
        default="both",
        help="Execution mode: 'train', 'eval', or 'both' (default: 'both')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs per k-hop configuration (default: 25)",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=128,
        help="Base PyG subgraph batch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.5e-3,
        help="Initial learning rate (default: 1.5e-3)",
    )
    parser.add_argument(
        "--weight_decay",
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--num_train_graphs",
        "--train-graphs",
        dest="num_train_graphs",
        type=int,
        default=36,
        help="Number of procedural training graphs per k-hop run (default: 36)",
    )
    parser.add_argument(
        "--num_val_graphs",
        "--val-graphs",
        dest="num_val_graphs",
        type=int,
        default=12,
        help="Number of procedural validation graphs per k-hop run (default: 12)",
    )
    parser.add_argument(
        "--proc_per_regime",
        "--proc-per-regime",
        dest="proc_per_regime",
        type=int,
        default=4,
        help="Number of held-out procedural graphs per regime during evaluation (default: 4)",
    )
    parser.add_argument(
        "--hidden_dim",
        "--hidden-dim",
        dest="hidden_dim",
        type=int,
        default=64,
        help="GINEConv hidden dimension (default: 64)",
    )
    parser.add_argument(
        "--num_layers",
        "--num-layers",
        dest="num_layers",
        type=int,
        default=3,
        help="Number of GINEConv message-passing layers (default: 3)",
    )
    parser.add_argument(
        "--sortpool_k",
        "--sortpool-k",
        dest="sortpool_k",
        type=int,
        default=16,
        help="SortPooling node count k (default: 16)",
    )
    parser.add_argument(
        "--max_nodes_per_subgraph",
        "--max-nodes-per-subgraph",
        dest="max_nodes_per_subgraph",
        type=int,
        default=150,
        help="Maximum nodes per enclosing subgraph to prevent hub explosion at k>=3 (default: 150; <=0 for unbounded)",
    )
    parser.add_argument(
        "--no_auto_scale_batch",
        action="store_true",
        help="Disable automatic batch size scaling for k >= 3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu', or None for auto-detect)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_seal_sweep(
        k_hops=args.k_hops,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_train_graphs=args.num_train_graphs,
        num_val_graphs=args.num_val_graphs,
        proc_per_regime=args.proc_per_regime,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        sortpool_k=args.sortpool_k,
        max_nodes_per_subgraph=args.max_nodes_per_subgraph,
        auto_scale_batch=not args.no_auto_scale_batch,
        seed=args.seed,
        device=args.device,
    )
