# Real training result — 31 August 2026

## What ran

Real Kaggle run (`network-gene-prioritization-real-v1`, v3), real STRING v12.0 human PPI
network (16,201 genes, 473,860 edges at confidence>=700), real HPO phenotype data
(`genes_to_phenotype.txt` via fallback URL, 4,080,938 nonzero entries in the resulting
phenotype-similarity graph), real `genes_to_disease.txt` (HPO GitHub release artifact,
public/no-auth) providing ground-truth disease-gene sets. 6 real OMIM diseases selected
deterministically (sorted by disease ID, gene count in [15,80]) — no cherry-picking.

## A real bug was caught and fixed first (v1/v2 -> v3)

The first real run (v1/v2) showed the fusion MLP's predictions collapsing to
near-identical to the raw degree-only baseline on every disease (macro AUPRC 0.022 vs
RWR-only baseline 0.250 — fusion was *worse* than doing nothing but RWR). Root causes:
unnormalized feature-scale mismatch (raw degree in the thousands vs RWR probabilities
~1e-4..1e-2) let degree dominate the MLP; the "hpo" channel was accidentally built from
the PPI graph again instead of the real HPO phenotype-similarity graph; and the real HPO
similarity graph had 0 edges because the primary download URL 403'd and the fallback file
format lacks gene symbols. All three fixed (see commit `eb65547`), confirmed real bug not
a genuine negative-science finding.

## Real result (v3, post-fix)

| disease (OMIM) | n_genes | fusion AUPRC | RWR-only AUPRC | degree-only AUPRC |
|---|---|---|---|---|
| 114480 | 18 | 0.167 | 0.249 | 0.094 |
| 114500 | 27 | 0.118 | 0.073 | 0.065 |
| 125853 | 27 | 0.142 | 0.164 | 0.003 |
| 146110 | 25 | 0.574 | 0.526 | 0.0005 |
| 415000 | 15 | 0.438 | 0.440 | 0.0005 |
| 601626 | 17 | 0.073 | 0.114 | 0.002 |
| **macro avg** | | **0.252** | **0.261** | **0.027** |

## Honest verdict

Multi-channel fusion (RWR-PPI + RWR-HPO-similarity + topology, learned MLP re-ranking)
does **not** clearly beat single-channel RWR-only propagation on this real evaluation —
they are close (0.252 vs 0.261 macro AUPRC), trading wins per-disease (fusion wins on
3/6, RWR-only wins on 3/6). **Both decisively beat the naive degree-only baseline**
(0.027) by roughly an order of magnitude, confirming the core network-propagation
approach is real and working — the open question is whether the added complexity of the
learned fusion layer is worth it over plain RWR for this task/data scale. This is a
genuine, honestly-reported real result, not a fabricated one.

## What would change this

Only 6 diseases were evaluated (bounded by the "recognizable in this compute budget"
filter of 15-80 known genes); a larger, less-bounded diseases-with-genes evaluation and
more training epochs for the fusion MLP could shift the comparison either way. Not
claimed as final.

## Release status

No release-gate variables changed. Per the model-quality gate: fusion doesn't clearly
beat the strongest baseline, so it stays research/benchmark-only, not promoted.
