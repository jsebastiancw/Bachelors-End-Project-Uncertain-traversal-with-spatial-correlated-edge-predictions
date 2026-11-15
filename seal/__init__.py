"""
seal/__init__.py
Standalone, Inductive SEAL Spatial Edge Blockage Prediction Package.
"""

from .model import (
    STATUS_CLEAR,
    STATUS_BLOCKED,
    STATUS_UNKNOWN,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    SEALModel,
    SEALPredictor,
    predict_edges,
    compute_drnl,
    prepare_spatial_graph,
    load_canonical_network,
    extract_enclosing_subgraph,
    extract_batch_subgraphs,
    normalize_edge_status_dict,
)

__all__ = [
    "STATUS_CLEAR",
    "STATUS_BLOCKED",
    "STATUS_UNKNOWN",
    "NODE_FEATURE_DIM",
    "EDGE_FEATURE_DIM",
    "SEALModel",
    "SEALPredictor",
    "predict_edges",
    "compute_drnl",
    "prepare_spatial_graph",
    "load_canonical_network",
    "extract_enclosing_subgraph",
    "extract_batch_subgraphs",
    "normalize_edge_status_dict",
]
