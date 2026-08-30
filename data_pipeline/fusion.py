"""
Learned fusion / re-ranking layer: MLP that combines multi-channel RWR scores
into a final prioritization score. Trained with BCE ranking objective.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """Small MLP scorer: n_features -> hidden -> 1 logit."""

    def __init__(self, n_features: int, hidden_dims: list[int] | None = None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, n_features) -> (n,) logits
        return self.net(x).squeeze(-1)


def train_fusion_model(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int | None = None,
    hidden_dims: list[int] | None = None,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    pos_weight: float | None = None,
    device: str = "cpu",
    seed: int = 42,
) -> FusionMLP:
    """
    Train FusionMLP with BCEWithLogitsLoss.
    X: (n, d) features, y: (n,) binary labels (1 = disease gene)
    Returns trained model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if n_features is None:
        n_features = X.shape[1]

    model = FusionMLP(n_features=n_features, hidden_dims=hidden_dims).to(device)
    Xt = torch.from_numpy(X.astype(np.float32)).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)

    # Handle class imbalance
    if pos_weight is None:
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight = n_neg / n_pos
        else:
            pos_weight = 1.0

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(Xt)
        loss = criterion(logits, yt)
        loss.backward()
        optimizer.step()

    return model


def predict_scores(model: FusionMLP, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X.astype(np.float32)).to(device)
        logits = model(Xt)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs
