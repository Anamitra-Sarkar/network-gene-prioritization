"""
Evaluation metrics and cross-validation splits for gene prioritization.

Honest protocol: leave-genes-out or leave-one-disease-out (not random edge split).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Recall@k: fraction of positives ranked in top k."""
    if y_true.sum() == 0:
        return 0.0
    # Get top-k indices
    k = min(k, len(y_score))
    top_k_idx = np.argsort(y_score)[::-1][:k]
    tp_in_topk = y_true[top_k_idx].sum()
    return float(tp_in_topk / y_true.sum())


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, ks: list[int] | None = None) -> dict:
    if ks is None:
        ks = [10, 25, 50, 100]
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(y_true, y_score, k)
    # AUPRC (average precision)
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        try:
            metrics["auprc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            metrics["auprc"] = 0.0
    else:
        metrics["auprc"] = 0.0
    return metrics


def leave_genes_out_split(
    n_genes: int,
    positive_indices: np.ndarray | list[int],
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Leave-genes-out CV: split positive genes into folds, each fold holds out
    a subset of positives as test, rest as train. Negatives are all non-positives.
    Returns list of (train_pos_idx, test_pos_idx) per fold.
    Ensures no leakage: a gene is either train or test per fold, never both.

    Also returns full y vectors would require caller to construct train/test masks.
    Here we just split positive indices.
    """
    rng = np.random.default_rng(seed)
    pos = np.array(positive_indices)
    rng.shuffle(pos)
    folds = np.array_split(pos, n_folds)
    splits = []
    for i in range(n_folds):
        test_pos = folds[i]
        train_pos = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        splits.append((train_pos, test_pos))
    return splits


def make_labels(n_genes: int, positive_indices: np.ndarray | list[int]) -> np.ndarray:
    y = np.zeros(n_genes, dtype=int)
    y[np.asarray(positive_indices)] = 1
    return y


def evaluate_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    Simple CV evaluation placeholder that returns per-fold metrics.
    Caller is responsible for training model per fold.
    This just provides split helper + metric computation.

    For honest evaluation without leakage, caller should:
      - For each fold, train fusion on train_pos, evaluate on test_pos vs negatives.
    """
    pos_idx = np.where(y == 1)[0]
    splits = leave_genes_out_split(len(y), pos_idx, n_folds=n_folds, seed=seed)
    return {"splits": splits, "pos_idx": pos_idx}
