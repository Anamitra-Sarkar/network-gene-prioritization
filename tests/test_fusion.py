"""Tests for fusion MLP + evaluation."""
import numpy as np
import pytest
import torch

from data_pipeline.fusion import FusionMLP, train_fusion_model, predict_scores
from data_pipeline.evaluation import recall_at_k, compute_metrics, leave_genes_out_split


def test_mlp_forward_shape():
    model = FusionMLP(n_features=4, hidden_dims=[8, 4], dropout=0.0)
    x = torch.randn(10, 4)
    out = model(x)
    assert out.shape == (10,)


def test_mlp_forward_no_nan():
    model = FusionMLP(n_features=3, hidden_dims=[16], dropout=0.0)
    x = torch.randn(5, 3)
    out = model(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_train_and_predict():
    np.random.seed(0)
    torch.manual_seed(0)
    n, d = 20, 4
    X = np.random.randn(n, d)
    # Make label correlated with first feature
    y = (X[:, 0] > 0).astype(int)
    # Ensure both classes present
    assert y.sum() > 0 and y.sum() < n
    model = train_fusion_model(X, y, epochs=10, lr=1e-2, seed=0)
    scores = predict_scores(model, X)
    assert scores.shape == (n,)
    assert np.all(scores >= 0) and np.all(scores <= 1)
    # Trained model should do better than random on training set (not strict, but check AUPRC > 0.3)
    metrics = compute_metrics(y, scores)
    assert metrics["auprc"] > 0.3


def test_recall_at_k():
    y_true = np.array([1, 0, 1, 0, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.05])
    # Top 1 has 1 positive, recall = 1/2 = 0.5
    assert abs(recall_at_k(y_true, y_score, 1) - 0.5) < 1e-9
    # Top 2 has both positives, recall = 1.0
    assert abs(recall_at_k(y_true, y_score, 2) - 1.0) < 1e-9
    # k >= n
    assert abs(recall_at_k(y_true, y_score, 100) - 1.0) < 1e-9


def test_compute_metrics():
    y_true = np.array([1, 0, 1, 0, 0, 1])
    y_score = np.array([0.9, 0.1, 0.8, 0.3, 0.2, 0.7])
    m = compute_metrics(y_true, y_score, ks=[1, 2, 3])
    assert "recall@1" in m and "recall@2" in m and "auprc" in m
    assert 0 <= m["auprc"] <= 1


def test_leave_genes_out_no_leak():
    n_genes = 20
    pos = np.array([0, 1, 2, 3, 4, 5])
    splits = leave_genes_out_split(n_genes, pos, n_folds=3, seed=42)
    assert len(splits) == 3
    all_test = set()
    for train_pos, test_pos in splits:
        # No overlap within fold
        assert len(set(train_pos) & set(test_pos)) == 0
        # Train+test covers all positives
        assert set(train_pos) | set(test_pos) == set(pos)
        all_test.update(test_pos)
    # Every positive appears in exactly one test fold
    assert all_test == set(pos)
