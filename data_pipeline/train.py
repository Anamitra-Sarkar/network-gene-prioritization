"""
Training pipeline: builds graph, runs multi-channel RWR, trains fusion MLP,
evaluates with leave-genes-out CV and baselines.

Usage (real data on Kaggle/Modal):
  python -m data_pipeline.train --string data/raw/9606.protein.links.v12.0.txt.gz \
    --hgnc data/raw/hgnc_complete_set.txt \
    --hpo data/raw/genes_to_phenotype.txt \
    --disgenet data/raw/disgenet_curated.tsv \
    --out artifacts/rev1

Sandbox: uses synthetic fixtures if no real files provided.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from data_pipeline.parsers import parse_string_links, parse_hpo_genes_to_phenotype, parse_hgnc, parse_disgenet
from data_pipeline.propagation import build_column_normalized_adjacency, rwr_multi_channel, build_feature_matrix
from data_pipeline.fusion import FusionMLP, train_fusion_model, predict_scores
from data_pipeline.evaluation import compute_metrics, leave_genes_out_split, make_labels
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--string", type=str, default=None)
    p.add_argument("--hgnc", type=str, default=None)
    p.add_argument("--hpo", type=str, default=None)
    p.add_argument("--disgenet", type=str, default=None)
    p.add_argument("--out", type=str, default="artifacts/dev")
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()

    print("Training pipeline: checking data sources...")
    if not args.string or not Path(args.string).exists():
        print("No real STRING file provided -- this is a sandbox run, not real training.")
        print("For real training on Kaggle/Modal, provide --string with real download.")
        # Demo: create tiny synthetic run to verify pipeline wiring
        n = 20
        rng = np.random.default_rng(0)
        edges = [(i, (i + 1) % n, 1.0) for i in range(n)] + [(i, (i + 2) % n, 0.5) for i in range(n)]
        W = build_column_normalized_adjacency(n, edges)
        # Simulate disease with 3 positives
        pos = [0, 1, 2]
        splits = leave_genes_out_split(n, pos, n_folds=3)
        print(f"Demo splits: {splits}")
        # Build features from RWR
        chans = rwr_multi_channel(W, ppi_seeds=[0], hpo_seeds=[5], hpo_weights=np.array([1.0]))
        X, names = build_feature_matrix(chans, W=W)
        y = make_labels(n, pos)
        print(f"Feature matrix {X.shape}, names {names}")
        model = train_fusion_model(X, y, epochs=5)
        scores = predict_scores(model, X)
        metrics = compute_metrics(y, scores)
        print(f"Demo metrics: {metrics}")
        # Baselines
        rwr_scores = chans["ppi"]
        print(f"RWR baseline metrics: {compute_metrics(y, rwr_scores)}")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "demo_metrics.json", "w") as f:
            json.dump({"fusion": metrics, "rwr_baseline": compute_metrics(y, rwr_scores.tolist() if hasattr(rwr_scores, 'tolist') else rwr_scores)}, f, indent=2)
        print(f"Wrote demo metrics to {out}")
        return

    # Real data path would continue here
    print("Real data training not yet executed in sandbox -- see docs for Kaggle/Modal instructions.")


if __name__ == "__main__":
    main()
