# Architecture

## Overview
Network-propagated multi-omics disease-gene prioritization.

Given seed genes known to be associated with a disease, propagate signal across the STRING PPI network (RWR), combine with HPO phenotype-similarity channel, and learn a fusion scorer.

## Components

### Data Pipeline (`data_pipeline/`)

- **parsers.py**: Real parsers for STRING (`protein.links.v12.0.txt.gz` + `protein.info.v12.0.txt.gz`), HGNC (`hgnc_complete_set.txt`), HPO (`genes_to_phenotype.txt` / `phenotype_to_genes.txt` / `phenotype.hpoa`), DisGeNET (curated TSV), and HPO `genes_to_disease.txt`. Hardened for real file quirks: skips `#` comment lines and blank rows, handles header case/whitespace variants, extra trailing columns, `.gz` vs plain, `comment="#"` in pandas, and gracefully returns empty on header-only files; raises `FileNotFoundError` with path on missing files.
- **download.py**: Helpers to fetch public files (STRING, HPO, HGNC) with ordered fallback URLs (HPO `genes_to_phenotype` -> `phenotype_to_genes` -> `phenotype.hpoa` last-resort). DisGeNET requires manual download or API key (`DISGENET_API_KEY` / `DISGENET_FILE`).
- **propagation.py**: Sparse RWR (`p_{t+1} = (1-r)W p_t + r p_0`), column-normalized adjacency via `scipy.sparse` (handles isolated nodes, optional self-loops, undirected expansion), multi-channel wrapper, feature matrix builder (RWR scores + degree/log_degree), and `build_hpo_similarity_adjacency` (IDF-weighted gene-gene graph from shared HPO terms, `min_shared_terms`, hub down-weighting).
- **fusion.py**: `FusionMLP` (PyTorch): `n_features -> [32,16] -> 1 logit`, BCEWithLogitsLoss with pos_weight for imbalance.
- **evaluation.py**: `recall@k` (k=10,25,50,100), AUPRC (average_precision, 0 when no positives or no negatives), `leave_genes_out_split` (no leakage, deterministic via seed), degree and raw-RWR baselines, `compute_metrics` wrapper.
- **train.py**: End-to-end training with leave-genes-out CV per disease (fusion vs RWR-only vs degree, macro-averaged AUPRC, honest "does NOT beat baseline" reporting), feature standardization (RWR ~1e-4 vs degree in thousands), and `--string-info` / `--genes-to-disease` real wiring; sandbox demo uses synthetic graph if no real files.

### Backend (`backend/app/`)

- **main.py**: FastAPI app, CORS (`allow_credentials=False` with wildcard — correct for browser), `/api/v1` router, `/` root info endpoint.
- **api/routes.py**:
  - `GET /api/v1/healthz` — public, no auth, reports `model_approved` + `artifact_revision`.
  - `GET /api/v1/me` — Firebase auth required.
  - `POST /api/v1/prioritize` — service-token gated (`X-Service-Token` or `Authorization: Bearer`), fail-closed release gate, input sanitization (trims/drops empty seed genes, upper-cases HPO terms, caps list length at 500, returns clean 422 with detail instead of 500 on malformed input), and honest abstention when no artifact.
- **core/config.py**: `is_model_release_approved()` — requires `MODEL_RELEASE_APPROVED=true` AND `APPROVED_ARTIFACT_REVISION` non-empty. Default abstains.
- **core/auth.py**: Firebase Admin SDK verification, `ALLOW_ANONYMOUS=true` bypass for local dev; service-token gate returns 503 when not configured, 403 on wrong token.
- **models/schemas.py**: Pydantic validation for `PrioritizeRequest` (`top_k` 1-500, `restart_prob` 0-1, field validators that strip/filter `seed_genes`/`hpo_terms`/`disease_name`, drop inner-whitespace entries, enforce max lengths).
- **services/prioritization.py**: Artifact-gated inference (checks `ARTIFACT_DIR/<revision>` exists, raises `FileNotFoundError`/`NotImplementedError` honestly when artifact missing — never hallucinates scores).

### Frontend (`frontend/`)

- React + Vite, Firebase JS SDK auth (`VITE_FIREBASE_*`), `getApiBase()`/`apiUrl()` helper (`VITE_API_URL`/`VITE_API_BASE`, falls back to relative `/api/*` for Vite proxy).
- `PrioritizePage.jsx`: disease/seed-gene/HPO input with associated `<label htmlFor>` + `aria-describedby`, client-side validation (requires disease OR seed genes, trims empty entries), `aria-live`/`role="alert"`/`role="status"` for loading/error/result, `aria-busy` on submit, HPO upper-casing, proper `detail` parsing for 422 arrays, responsive `maxWidth`/`clamp()` header, `overflowX:auto` for results table with `scope="col"`.
- `App.jsx`: health fetch with unreachable handling, `code` word-break for JSON, abstention banner with honest messaging, `LoginForm` with labeled `type="email"`/`type="password"` + `autoComplete` + `required` + accessible error `role="alert"`, header with `clamp()` sizing for narrow widths.
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
- No real training has run yet; all metrics in repo are from synthetic unit-test fixtures (real training deferred to Kaggle/Modal).
- Parsers + API + frontend are now hardened against real-format quirks (comments, blanks, header variants, gz, 422 validation) and covered by ~60 tests, but still untested against a live multi-GB STRING download (by design — no large download in CI/sandbox).
