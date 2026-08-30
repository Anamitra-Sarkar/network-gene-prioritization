"""
Random Walk with Restart (RWR) over STRING PPI network.

Formula: p_{t+1} = (1 - r) * W * p_t + r * p_0
Where W is column-normalized adjacency (or row, configurable, default column),
p_0 is seed indicator vector, r is restart probability.

Uses scipy.sparse for scalability to ~20k proteins / 500k+ edges.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_column_normalized_adjacency(
    n_nodes: int,
    edges: list[tuple[int, int, float]] | np.ndarray,
    add_self_loops: bool = False,
) -> sp.csr_matrix:
    """
    Build column-normalized adjacency matrix W from edge list.
    edges: iterable of (src, dst, weight). Graph is treated as undirected
           (both directions added) unless already directed.
    Column normalization: W[:, j] sums to 1 (or 0 for isolated nodes).
    """
    if len(edges) == 0:
        return sp.csr_matrix((n_nodes, n_nodes), dtype=np.float64)

    if isinstance(edges, np.ndarray):
        rows = edges[:, 0].astype(int)
        cols = edges[:, 1].astype(int)
        data = edges[:, 2].astype(float) if edges.shape[1] > 2 else np.ones(len(rows))
    else:
        rows, cols, data = [], [], []
        for e in edges:
            if len(e) == 3:
                r, c, w = e
            else:
                r, c = e[0], e[1]
                w = 1.0
            rows.append(int(r))
            cols.append(int(c))
            data.append(float(w))
        rows = np.array(rows, dtype=int)
        cols = np.array(cols, dtype=int)
        data = np.array(data, dtype=float)

    # Make undirected: add reverse edges
    # Use symmetric addition but avoid double-counting if already symmetric
    # Simple: add both directions explicitly
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    all_data = np.concatenate([data, data])

    if add_self_loops:
        all_rows = np.concatenate([all_rows, np.arange(n_nodes)])
        all_cols = np.concatenate([all_cols, np.arange(n_nodes)])
        all_data = np.concatenate([all_data, np.ones(n_nodes)])

    adj = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(n_nodes, n_nodes)).tocsr()

    # Column normalization: divide each column by its sum
    col_sums = np.array(adj.sum(axis=0)).ravel()
    # Avoid div by zero
    col_sums[col_sums == 0] = 1.0
    inv_col_sums = 1.0 / col_sums
    diag = sp.diags(inv_col_sums, format="csr")
    W = adj @ diag  # column-normalized: adj * D^{-1}

    return W.tocsr()


def random_walk_with_restart(
    W: sp.spmatrix,
    seed_indices: list[int] | np.ndarray,
    restart_prob: float = 0.3,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Run RWR: p_{t+1} = (1-r)*W*p_t + r*p0
    Returns stationary distribution p (n_nodes,) summing to 1.

    Args:
        W: column-normalized adjacency (n x n), sparse
        seed_indices: indices of seed genes
        restart_prob: r in [0,1]
        max_iter: max iterations
        tol: L1 convergence threshold
        seed_weights: optional weights for seeds (same length as seed_indices),
                     if None uniform.
    """
    n = W.shape[0]
    if len(seed_indices) == 0:
        raise ValueError("seed_indices must be non-empty")

    p0 = np.zeros(n, dtype=np.float64)
    if seed_weights is not None:
        w = np.asarray(seed_weights, dtype=np.float64)
        w = w / w.sum()
        p0[np.asarray(seed_indices)] = w
    else:
        p0[np.asarray(seed_indices)] = 1.0 / len(seed_indices)

    p = p0.copy()
    r = float(restart_prob)
    one_minus_r = 1.0 - r

    for _ in range(max_iter):
        p_new = one_minus_r * (W @ p) + r * p0
        # Renormalize to handle numerical drift
        s = p_new.sum()
        if s > 0:
            p_new = p_new / s
        delta = np.abs(p_new - p).sum()
        p = p_new
        if delta < tol:
            break

    # Final sanity: ensure non-negative and sums to 1
    p = np.maximum(p, 0)
    s = p.sum()
    if s > 0:
        p = p / s
    return p


def rwr_multi_channel(
    W: sp.spmatrix,
    ppi_seeds: list[int] | np.ndarray,
    hpo_seeds: list[int] | np.ndarray | None = None,
    hpo_weights: np.ndarray | None = None,
    restart_prob: float = 0.3,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, np.ndarray]:
    """
    Run RWR for multiple channels:
      - ppi: seeded from known disease genes
      - hpo: seeded from HPO phenotype-similarity weighted genes (if provided)
    Returns dict of channel_name -> score vector.
    """
    channels: dict[str, np.ndarray] = {}
    channels["ppi"] = random_walk_with_restart(
        W, ppi_seeds, restart_prob=restart_prob, max_iter=max_iter, tol=tol
    )
    if hpo_seeds is not None and len(hpo_seeds) > 0:
        channels["hpo"] = random_walk_with_restart(
            W, hpo_seeds, restart_prob=restart_prob, max_iter=max_iter, tol=tol,
            seed_weights=hpo_weights,
        )
    return channels


def build_feature_matrix(
    channels: dict[str, np.ndarray],
    W: sp.spmatrix | None = None,
    include_topology: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Stack propagation channels + optional topology features into feature matrix.
    Returns (n_nodes x n_features) and feature_names.
    Topology features (if W provided and include_topology): degree, log_degree.
    """
    n = next(iter(channels.values())).shape[0]
    feature_names: list[str] = []
    cols: list[np.ndarray] = []

    for name, scores in channels.items():
        cols.append(scores.reshape(-1, 1))
        feature_names.append(f"rwr_{name}")

    if include_topology and W is not None:
        # Degree from adjacency (before normalization, approximate via W nonzero)
        # Use W's column sums before norm would be degree; here use nonzero count per node
        # More accurate: compute degree as number of neighbors or weighted degree
        # We'll use W's structure: degree ~ number of nonzeros per row
        if sp.issparse(W):
            degrees = np.array((W != 0).sum(axis=1)).ravel().astype(float)
            # Weighted degree: sum of column-normalized values != true degree, so use nnz
        else:
            degrees = (W != 0).sum(axis=1).astype(float)
        cols.append(degrees.reshape(-1, 1))
        feature_names.append("degree")
        cols.append(np.log1p(degrees).reshape(-1, 1))
        feature_names.append("log_degree")

    X = np.hstack(cols) if cols else np.zeros((n, 0))
    return X, feature_names
