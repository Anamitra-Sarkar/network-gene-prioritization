"""Prioritization service (real RWR logic, gated behind release flag)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def load_graph_and_run_prioritization(
    seed_genes: list[str],
    hpo_terms: list[str] | None = None,
    top_k: int = 50,
    restart_prob: float = 0.3,
) -> list[dict]:
    """
    Real prioritization: would load STRING graph, run RWR, fuse channels.
    This function is only called when model release is approved and artifacts exist.
    For now, it checks for artifact path and raises if not found (abstain).
    """
    artifact_dir = os.getenv("ARTIFACT_DIR", "artifacts")
    artifact_revision = os.getenv("APPROVED_ARTIFACT_REVISION", "")

    # Check for precomputed artifacts (model weights, gene index, normalized graph)
    # If not present, we cannot serve real predictions -- abstain honestly.
    artifact_path = Path(artifact_dir) / artifact_revision if artifact_revision else None
    if artifact_path is None or not artifact_path.exists():
        raise FileNotFoundError(f"No approved artifact found at {artifact_path}")

    # Real implementation would:
    # 1. Load gene index (HGNC symbols -> idx)
    # 2. Load precomputed column-normalized W (sparse npz)
    # 3. Map seed_genes to indices, run rwr_multi_channel
    # 4. Load fusion MLP weights, compute feature matrix, predict
    # 5. Rank and return top_k

    # Placeholder for artifact-based loading (not reached until real training on Kaggle)
    raise NotImplementedError("Artifact loading not yet implemented -- run training pipeline first")
