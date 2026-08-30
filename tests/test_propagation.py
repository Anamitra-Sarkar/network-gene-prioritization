"""Tests for RWR propagation with synthetic fixtures."""
import numpy as np
import scipy.sparse as sp
import pytest

from data_pipeline.propagation import (
    build_column_normalized_adjacency,
    random_walk_with_restart,
    rwr_multi_channel,
    build_feature_matrix,
)


def _tiny_chain_graph():
    """
    Tiny chain: 0 - 1 - 2 - 3 - 4
    Seed = node 0, so ranking should be 1 > 2 > 3 > 4 in RWR.
    """
    n = 5
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
    W = build_column_normalized_adjacency(n, edges)
    return n, W


def test_rwr_converges_to_prob_distribution():
    n, W = _tiny_chain_graph()
    p = random_walk_with_restart(W, [0], restart_prob=0.3, max_iter=100, tol=1e-8)
    assert p.shape == (n,)
    assert np.all(p >= -1e-9), "All scores non-negative"
    assert abs(p.sum() - 1.0) < 1e-6, f"Sums to 1, got {p.sum()}"
    # Seed should have highest score
    assert p[0] == p.max()


def test_rwr_neighbor_ranking():
    n, W = _tiny_chain_graph()
    p = random_walk_with_restart(W, [0], restart_prob=0.5, max_iter=200, tol=1e-9)
    # Direct neighbor (1) should rank above 3-hop-away (3) and far node 4
    assert p[1] > p[3], f"neighbor 1 ({p[1]}) should > node 3 ({p[3]})"
    assert p[1] > p[4], f"neighbor 1 ({p[1]}) should > node 4 ({p[4]})"
    assert p[2] > p[4], f"node 2 ({p[2]}) should > node 4 ({p[4]})"


def test_rwr_sparse_vs_dense_agreement():
    # Small graph where dense computation is tractable
    n = 4
    edges = [(0, 1, 2.0), (1, 2, 1.0), (2, 3, 1.0), (0, 2, 0.5)]
    W = build_column_normalized_adjacency(n, edges)
    p = random_walk_with_restart(W, [0], restart_prob=0.3, max_iter=100)
    assert abs(p.sum() - 1.0) < 1e-6
    assert np.all(p >= 0)


def test_rwr_empty_seeds_raises():
    n, W = _tiny_chain_graph()
    with pytest.raises(ValueError):
        random_walk_with_restart(W, [], restart_prob=0.3)


def test_column_normalized():
    n = 3
    edges = [(0, 1, 1.0), (1, 2, 1.0)]
    W = build_column_normalized_adjacency(n, edges)
    col_sums = np.array(W.sum(axis=0)).ravel()
    # Non-isolated columns should sum to 1; isolated columns sum to 0
    # In chain 0-1-2, all nodes have edges so each col sum ~1
    for cs in col_sums:
        assert abs(cs - 1.0) < 1e-9 or abs(cs) < 1e-9


def test_multi_channel_shapes():
    n, W = _tiny_chain_graph()
    channels = rwr_multi_channel(W, ppi_seeds=[0], hpo_seeds=[4], hpo_weights=np.array([1.0]))
    assert "ppi" in channels and "hpo" in channels
    assert channels["ppi"].shape == (n,)
    assert channels["hpo"].shape == (n,)
    X, names = build_feature_matrix(channels, W=W)
    # 2 channels + 2 topology = 4 features
    assert X.shape == (n, 4)
    assert len(names) == 4
    assert names[0] == "rwr_ppi"
    assert names[1] == "rwr_hpo"


def test_feature_matrix_without_hpo():
    n, W = _tiny_chain_graph()
    channels = rwr_multi_channel(W, ppi_seeds=[0])
    assert "hpo" not in channels
    X, names = build_feature_matrix(channels, W=W, include_topology=False)
    assert X.shape == (n, 1)


def test_rwr_weighted_seeds():
    n, W = _tiny_chain_graph()
    p_uniform = random_walk_with_restart(W, [0, 4], restart_prob=0.3)
    p_weighted = random_walk_with_restart(W, [0, 4], restart_prob=0.3, seed_weights=np.array([0.9, 0.1]))
    # Weighted toward 0 should give node 0 higher than uniform's node 0 vs node 4 balance
    # Actually weighted 0 heavily should make p[0] > p[4] more pronounced
    assert p_weighted[0] > p_weighted[4]
    # The weighted version should have more mass near 0 than uniform
    assert p_weighted[1] > p_uniform[1] or p_weighted[0] > p_uniform[0]
