# Architecture

## Overview
Network-propagated multi-omics disease-gene prioritization.

Given seed genes known to be associated with a disease, propagate signal across the STRING PPI network (RWR), combine with HPO phenotype-similarity channel, and learn a fusion scorer.

## Components

### Data Pipeline (`data_pipeline/`)

- **parsers.py**: Real parsers for STRING (`protein.links.v12.0.txt.gz`), HGNC (`hgnc_complete_set.txt`), HPO (`genes_to_phenotype.txt` / `phenotype.hpoa`), DisGeNET (curated TSV).
- **download.py**: Helpers to fetch public files (STRING, HPO, HGNC). DisGeNET requires manual download or API key.
- **propagation.py**: Sparse RWR (`p_{t+1} = (1-r)W p_t + r p_0`), column-normalized adjacency via `scipy.sparse`, multi-channel wrapper, feature matrix builder (RWR scores + degree/log_degree).
- **fusion.py**: `FusionMLP` (PyTorch): `n_features -> [32,16] -> 1 logit`, BCEWithLogitsLoss with pos_weight for imbalance.
- **evaluation.py**: `recall@k` (k=10,25,50,100), AUPRC, `leave_genes_out_split` (no leakage), degree and raw-RWR baselines.
- **train.py**: End-to-end training with leave-genes-out CV; sandbox demo uses synthetic graph if no real files.

### Backend (`backend/app/`)

- **main.py**: FastAPI app, CORS, `/api/v1` router.
- **api/routes.py**:
  - `GET /api/v1/healthz` — public, no auth, reports `model_approved`.
  - `GET /api/v1/me` — Firebase auth required.
  - `POST /api/v1/prioritize` — service-token gated (`X-Service-Token` or `Authorization: Bearer`), fail-closed release gate.
- **core/config.py**: `is_model_release_approved()` — requires `MODEL_RELEASE_APPROVED=true` AND `APPROVED_ARTIFACT_REVISION` non-empty. Default abstains.
- **core/auth.py**: Firebase Admin SDK verification, `ALLOW_ANONYMOUS=true` bypass for local dev.
- **services/prioritization.py**: Artifact-gated inference (checks `ARTIFACT_DIR/<revision>` exists).

### Frontend (`frontend/`)

- React + Vite, Firebase JS SDK auth (`VITE_FIREBASE_*`).
- `PrioritizePage.jsx`: disease/seed-gene/HPO input, calls `/api/v1/prioritize`, honestly renders abstention state when no approved model.
- When `VITE_FIREBASE_API_KEY` not set, shows preview mode with abstention demo.

## Data Flow (real run)

```
STRING (9606.protein.links) --\
HGNC (symbol map)          ---> build graph (20k nodes, 500k+ edges) -> W (column-norm sparse)
HPO (genes_to_phenotype)   --/        |
DisGeNET (curated)  ----------------> seeds per disease -> RWR per channel -> feature matrix -> FusionMLP -> ranked genes
                                      Evaluation: leave-genes-out CV, recall@k, AUPRC vs RWR-only and degree baselines
```

## Security

- Release gate is fail-closed: both env vars required, default unapproved, no demo hallucination.
- `/prioritize` requires service token; user-facing `/me` requires Firebase ID token.
- No synthetic-data fallback by default in production code; synthetic fixtures only in tests/train demo mode.

## Deployment

- API: `uvicorn backend.app.main:app --port 8000` (Docker/Modal).
- Frontend: `npm run build` -> static `dist/`.
- Training: `python -m data_pipeline.train --string ... --hgnc ... --hpo ... --disgenet ... --out artifacts/<rev>` on Kaggle/Modal (GPU for fusion MLP, CPU sufficient for RWR).

## Honest Limitations

- DisGeNET current access: see README — we could not confirm exact current API terms without live check; pipeline accepts file or env key.
- No real training has run yet; all metrics in repo are from synthetic unit-test fixtures.
