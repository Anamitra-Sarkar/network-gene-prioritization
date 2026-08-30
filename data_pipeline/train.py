"""
Training pipeline: builds graph, runs multi-channel RWR, trains fusion MLP,
evaluates with leave-genes-out CV and baselines.

Usage (real data on Kaggle/Modal):
  python -m data_pipeline.train --string data/raw/9606.protein.links.v12.0.txt.gz \
    --string-info data/raw/9606.protein.info.v12.0.txt.gz \
    --hpo data/raw/genes_to_phenotype.txt \
    --genes-to-disease data/raw/genes_to_disease.txt \
    --out artifacts/rev1

Sandbox: uses synthetic fixtures if no real files provided.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from data_pipeline.parsers import (
    parse_string_links,
    parse_string_info,
    string_protein_to_gene,
    parse_hpo_genes_to_phenotype,
    parse_genes_to_disease,
    build_gene_hpo_terms,
)
from data_pipeline.propagation import (
    build_column_normalized_adjacency,
    build_hpo_similarity_adjacency,
    rwr_multi_channel,
    build_feature_matrix,
)
from data_pipeline.fusion import FusionMLP, train_fusion_model, predict_scores
from data_pipeline.evaluation import compute_metrics, leave_genes_out_split, make_labels
import torch


def _select_usable_device() -> str:
    """Real-op probe: torch.cuda.is_available() can lie on some GPU/wheel
    combinations (seen repeatedly on Kaggle P100/sm_60 this session)."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        (torch.zeros(1, device="cuda") + 1).cpu()
        return "cuda"
    except Exception as exc:  # noqa: BLE001
        print(f"CUDA reported available but a real op failed ({exc}) -- falling back to CPU")
        return "cpu"


def run_real_training(
    string_path: str,
    string_info_path: str,
    hpo_path: str,
    genes_to_disease_path: str,
    out_dir: str,
    min_string_score: int = 700,
    min_disease_genes: int = 15,
    max_disease_genes: int = 80,
    max_diseases: int = 5,
    n_folds: int = 5,
    epochs: int = 50,
    device: str | None = None,
) -> dict:
    device = device or _select_usable_device()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/6] Parsing STRING PPI links + protein.info gene-symbol map...")
    links = parse_string_links(string_path, min_score=min_string_score)
    id_to_symbol = parse_string_info(string_info_path)
    print(f"  {len(links)} STRING edges >= {min_string_score}, {len(id_to_symbol)} protein->symbol mappings")

    print("[2/6] Building gene index + column-normalized PPI adjacency...")
    symbols_seen = set()
    edge_symbol_pairs = []
    for p1, p2, score in links.itertuples(index=False):
        s1 = id_to_symbol.get(p1)
        s2 = id_to_symbol.get(p2)
        if not s1 or not s2 or s1 == s2:
            continue
        symbols_seen.add(s1)
        symbols_seen.add(s2)
        edge_symbol_pairs.append((s1, s2, float(score)))

    gene_list = sorted(symbols_seen)
    gene_index = {g: i for i, g in enumerate(gene_list)}
    n_nodes = len(gene_list)
    edges = [(gene_index[s1], gene_index[s2], w) for s1, s2, w in edge_symbol_pairs]
    W_ppi = build_column_normalized_adjacency(n_nodes, edges)
    print(f"  {n_nodes} genes, {len(edges)} edges in PPI graph")

    print("[3/6] Parsing HPO genes_to_phenotype.txt + building phenotype-similarity graph...")
    gene_hpo_terms = build_gene_hpo_terms(hpo_path)
    W_hpo = build_hpo_similarity_adjacency(gene_hpo_terms, gene_index, n_nodes, min_shared_terms=2)
    print(f"  HPO similarity graph: {W_hpo.nnz} nonzero entries")

    print("[4/6] Parsing genes_to_disease.txt, selecting evaluation diseases...")
    disease_to_genes = parse_genes_to_disease(genes_to_disease_path)
    candidates = []
    for disease_id, genes in disease_to_genes.items():
        in_graph = [g for g in genes if g in gene_index]
        if min_disease_genes <= len(in_graph) <= max_disease_genes:
            candidates.append((disease_id, in_graph))
    candidates.sort(key=lambda x: x[0])  # deterministic, no cherry-picking
    candidates = candidates[:max_diseases]
    print(f"  Selected {len(candidates)} diseases: {[c[0] for c in candidates]}")

    if not candidates:
        raise RuntimeError(
            "No diseases found with gene-count in the configured range that are "
            "present in the STRING graph -- widen --min-disease-genes/--max-disease-genes."
        )

    print("[5/6] Running leave-genes-out CV per disease (fusion vs RWR-only vs degree baselines)...")
    all_results = {}
    for disease_id, pos_genes in candidates:
        pos_idx = np.array([gene_index[g] for g in pos_genes])
        splits = leave_genes_out_split(n_nodes, pos_idx, n_folds=min(n_folds, len(pos_idx)))
        fold_metrics_fusion, fold_metrics_rwr, fold_metrics_degree = [], [], []

        for train_pos, test_pos in splits:
            if len(train_pos) == 0 or len(test_pos) == 0:
                continue
            chans = rwr_multi_channel(W_ppi, ppi_seeds=train_pos, hpo_seeds=train_pos)
            chans_hpo_only = None
            if W_hpo.nnz > 0:
                from data_pipeline.propagation import random_walk_with_restart
                try:
                    chans["hpo"] = random_walk_with_restart(W_hpo, train_pos)
                except ValueError:
                    pass

            X, feat_names = build_feature_matrix(chans, W=W_ppi)
            y = np.zeros(n_nodes, dtype=int)
            y[test_pos] = 1
            # Exclude train-positive genes from evaluation (they were the seeds, not held out)
            eval_mask = np.ones(n_nodes, dtype=bool)
            eval_mask[train_pos] = False

            model = train_fusion_model(X[eval_mask], y[eval_mask], epochs=epochs, device=device)
            scores_full = predict_scores(model, X, device=device)

            fold_metrics_fusion.append(compute_metrics(y[eval_mask], scores_full[eval_mask]))
            fold_metrics_rwr.append(compute_metrics(y[eval_mask], chans["ppi"][eval_mask]))
            degree_col = feat_names.index("degree")
            fold_metrics_degree.append(compute_metrics(y[eval_mask], X[eval_mask, degree_col]))

        def _avg(fold_list):
            if not fold_list:
                return {}
            keys = fold_list[0].keys()
            return {k: float(np.mean([f[k] for f in fold_list])) for k in keys}

        all_results[disease_id] = {
            "n_genes": len(pos_genes),
            "n_folds_run": len(fold_metrics_fusion),
            "fusion": _avg(fold_metrics_fusion),
            "rwr_baseline": _avg(fold_metrics_rwr),
            "degree_baseline": _avg(fold_metrics_degree),
        }
        print(f"  {disease_id} ({len(pos_genes)} genes): "
              f"fusion AUPRC={all_results[disease_id]['fusion'].get('auprc', 0):.4f} "
              f"vs rwr={all_results[disease_id]['rwr_baseline'].get('auprc', 0):.4f} "
              f"vs degree={all_results[disease_id]['degree_baseline'].get('auprc', 0):.4f}")

    print("[6/6] Writing metrics...")
    macro_fusion_auprc = float(np.mean([r["fusion"].get("auprc", 0) for r in all_results.values()]))
    macro_rwr_auprc = float(np.mean([r["rwr_baseline"].get("auprc", 0) for r in all_results.values()]))
    macro_degree_auprc = float(np.mean([r["degree_baseline"].get("auprc", 0) for r in all_results.values()]))

    summary = {
        "n_nodes": n_nodes,
        "n_ppi_edges": len(edges),
        "min_string_score": min_string_score,
        "per_disease": all_results,
        "macro_avg": {
            "fusion_auprc": macro_fusion_auprc,
            "rwr_baseline_auprc": macro_rwr_auprc,
            "degree_baseline_auprc": macro_degree_auprc,
        },
        "interpretation": (
            "fusion beats both baselines" if macro_fusion_auprc > max(macro_rwr_auprc, macro_degree_auprc)
            else "fusion does NOT beat the strongest baseline (honest)"
        ),
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["macro_avg"], indent=2))
    print(f"Interpretation: {summary['interpretation']}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--string", type=str, default=None)
    p.add_argument("--string-info", type=str, default=None)
    p.add_argument("--hgnc", type=str, default=None)
    p.add_argument("--hpo", type=str, default=None)
    p.add_argument("--disgenet", type=str, default=None)
    p.add_argument("--genes-to-disease", type=str, default=None)
    p.add_argument("--out", type=str, default="artifacts/dev")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--min-string-score", type=int, default=700)
    p.add_argument("--min-disease-genes", type=int, default=15)
    p.add_argument("--max-disease-genes", type=int, default=80)
    p.add_argument("--max-diseases", type=int, default=5)
    args = p.parse_args()

    print("Training pipeline: checking data sources...")
    have_real = args.string and Path(args.string).exists() and args.string_info and Path(args.string_info).exists() \
        and args.hpo and Path(args.hpo).exists() and args.genes_to_disease and Path(args.genes_to_disease).exists()

    if not have_real:
        print("No complete real data set provided -- this is a sandbox run, not real training.")
        print("For real training on Kaggle/Modal, provide --string --string-info --hpo --genes-to-disease.")
        # Demo: create tiny synthetic run to verify pipeline wiring
        n = 20
        rng = np.random.default_rng(0)
        edges = [(i, (i + 1) % n, 1.0) for i in range(n)] + [(i, (i + 2) % n, 0.5) for i in range(n)]
        W = build_column_normalized_adjacency(n, edges)
        pos = [0, 1, 2]
        splits = leave_genes_out_split(n, pos, n_folds=3)
        print(f"Demo splits: {splits}")
        chans = rwr_multi_channel(W, ppi_seeds=[0], hpo_seeds=[5], hpo_weights=np.array([1.0]))
        X, names = build_feature_matrix(chans, W=W)
        y = make_labels(n, pos)
        print(f"Feature matrix {X.shape}, names {names}")
        model = train_fusion_model(X, y, epochs=5)
        scores = predict_scores(model, X)
        metrics = compute_metrics(y, scores)
        print(f"Demo metrics: {metrics}")
        rwr_scores = chans["ppi"]
        print(f"RWR baseline metrics: {compute_metrics(y, rwr_scores)}")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "demo_metrics.json", "w") as f:
            json.dump({"fusion": metrics, "rwr_baseline": compute_metrics(y, rwr_scores.tolist() if hasattr(rwr_scores, 'tolist') else rwr_scores)}, f, indent=2)
        print(f"Wrote demo metrics to {out}")
        return

    run_real_training(
        string_path=args.string,
        string_info_path=args.string_info,
        hpo_path=args.hpo,
        genes_to_disease_path=args.genes_to_disease,
        out_dir=args.out,
        min_string_score=args.min_string_score,
        min_disease_genes=args.min_disease_genes,
        max_disease_genes=args.max_disease_genes,
        max_diseases=args.max_diseases,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
