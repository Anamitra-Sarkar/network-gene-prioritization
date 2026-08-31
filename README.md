# network-gene-prioritization

Network-propagated multi-omics disease-gene prioritization — a classic bioinformatics task: given seed genes known to be associated with a disease, propagate that signal across a protein-protein interaction network to rank all other genes by likelihood of being disease-relevant (candidate gene discovery for rare disease diagnosis / GWAS follow-up).

> **Status: research-grade scaffold, no real training run yet.** All real data pipelines are implemented and tested against structurally-realistic fixtures, but no approved model artifact exists. The API honestly abstains (`status: "abstained"`) until `MODEL_RELEASE_APPROVED=true` and `APPROVED_ARTIFACT_REVISION` are set and an artifact is present. No synthetic demo scores are ever returned as real.

## Method (real, not stub)

1. **Random Walk with Restart (RWR)** over the STRING PPI graph: `p_{t+1} = (1-r) W p_t + r p_0` (column-normalized adjacency, sparse `scipy.sparse`, iterate to convergence). Tested on tiny graphs where neighbor ranking is hand-verifiable.
2. **Multiple propagation channels**: (a) seeded from known disease genes on the PPI network, (b) seeded from HPO phenotype-similarity-weighted genes — stacked as per-gene feature channels (+ degree/log_degree topology features).
3. **Learned fusion / re-ranking**: small PyTorch `FusionMLP` (`n_features -> 32 -> 16 -> 1 logit`) trained with `BCEWithLogitsLoss` (pos_weight for imbalance) on held-out genes; evaluated with **leave-genes-out CV** (not random split — avoids network-proximity leakage).
4. **Honest evaluation**: `recall@k` (k=10,25,50,100) + AUPRC vs baselines (raw RWR-only, degree). The question answered is: *does multi-channel fusion beat single-channel RWR?* — reported whichever way it goes.

## Real Data Sources

| Source | What | Access | Citation / URL |
|--------|------|--------|----------------|
| **STRING** v12.0 | Human PPI network (~20k proteins, 500k+ edges, `combined_score`) | **Public, no auth** — `https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz` (also REST API at `https://string-db.org/api/`) | Szklarczyk et al., Nucleic Acids Res 2023 |
| **HGNC** | Canonical gene symbols/IDs (`hgnc_complete_set.txt`) | **Public, no auth** — `https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt` (or genenames.org) | Seal et al., Nucleic Acids Res 2023 |
| **HPO** | Phenotype-to-gene associations (`genes_to_phenotype.txt` / `phenotype.hpoa`) | **Public, no auth** — `https://hpo.jax.org/app/download/annotation` (e.g. `genes_to_phenotype.txt`, `phenotype.hpoa` at `http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa`) | Köhler et al., Nucleic Acids Res 2021 |
| **DisGeNET** | Curated gene-disease associations | **Requires free account + API key for full current access** (honest: exact current terms not re-verified live in this sandbox; pipeline accepts a pre-downloaded file OR `DISGENET_API_KEY`/`DISGENET_FILE` env var — see `data_pipeline/download.py` and `docs/architecture.md`). Older versions were open; recent versions gate downloads. Verify at `https://disgenet.com/signup` and docs at `https://disgenet.com/disgenet_docs/` | Piñero et al., Nucleic Acids Res 2020 |

> **Access honesty**: STRING, HGNC, and HPO were confirmed public/no-auth from general knowledge (consistent with prior portfolio repos `dti-polypharm-ehgt`, `robustbiodiscoverer` which already use HGNC). **DisGeNET**: we **could not confirm the exact current real access terms without a live web check** (sandbox has no external network to re-verify). The code and docs therefore state the requirement plainly and do not assume anonymous access — it accepts either a user-supplied exported file (`--disgenet data/raw/disgenet_curated.tsv`) or API key env var, and fails with instructions if neither is provided. This is the correct fail-closed posture for a gated source.

## Repo Layout

```
data_pipeline/        # RWR, fusion, parsers, evaluation, download, train
backend/app/          # FastAPI: /healthz, /prioritize (service-token + release gate), Firebase auth
frontend/             # React/Vite: authenticated prioritize page with abstention state
tests/                # pytest: RWR, fusion, parsers (inline fixtures), API gate
docs/architecture.md  # detailed architecture
```

## Quickstart (local, no real data)

```bash
pip install -r requirements.txt
pytest -v  # 60+ tests, ~2-4s

# API (abstains honestly without release approval)
uvicorn backend.app.main:app --reload --port 8000
curl http://localhost:8000/api/v1/healthz
# {"status":"ok","model_approved":false,"artifact_revision":null}

curl -X POST http://localhost:8000/api/v1/prioritize \
  -H "X-Service-Token: $SERVICE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"seed_genes":["FBN1","TGFBR1"],"top_k":10}'
# {"status":"abstained","message":"Research service, no approved model yet..."}

# Frontend
cd frontend && npm install && npm run dev  # http://localhost:5173
```

Enable real inference (after Kaggle/Modal training):

```bash
export MODEL_RELEASE_APPROVED=true
export APPROVED_ARTIFACT_REVISION=rev-2026-08-31
export ARTIFACT_DIR=artifacts  # must contain artifacts/rev-2026-08-31/
export SERVICE_TOKEN=<secret>
export FIREBASE_ADMIN_JSON=<path or JSON>
```

## Real Training (Kaggle / Modal — for coordinator)

```bash
# 1. Download real data (public sources no auth; DisGeNET needs account/file)
python -m data_pipeline.download --all
# Or manually: curl -O https://stringdb-downloads.org/.../9606.protein.links.v12.0.txt.gz
# For DisGeNET: download via account at disgenet.com or set DISGENET_API_KEY

# 2. Train (GPU for fusion MLP, CPU OK for RWR)
python -m data_pipeline.train \
  --string data/raw/9606.protein.links.v12.0.txt.gz \
  --hgnc data/raw/hgnc_complete_set.txt \
  --hpo data/raw/genes_to_phenotype.txt \
  --disgenet data/raw/disgenet_curated.tsv \
  --out artifacts/rev-2026-08-31 --epochs 50

# 3. Evaluate: check artifacts/rev-*/metrics.json for recall@k, AUPRC vs baselines
# 4. Release: set MODEL_RELEASE_APPROVED=true and APPROVED_ARTIFACT_REVISION=rev-2026-08-31
```

## API

- `GET /api/v1/healthz` — public, no auth. Returns `{status, model_approved, artifact_revision}`.
- `GET /api/v1/me` — Firebase auth (`Authorization: Bearer <ID token>`).
- `POST /api/v1/prioritize` — service-token gated (`X-Service-Token` or `Authorization: Bearer <token>`, env `SERVICE_TOKEN`/`INTERNAL_SERVICE_TOKEN`). Body: `{disease_name?, seed_genes?, hpo_terms?, top_k?, restart_prob?}`. Returns `abstained` until release approved, else ranked genes.

Env vars: `MODEL_RELEASE_APPROVED`, `APPROVED_ARTIFACT_REVISION`, `SERVICE_TOKEN` / `INTERNAL_SERVICE_TOKEN`, `FIREBASE_ADMIN_JSON` / `FIREBASE_SERVICE_ACCOUNT`, `ALLOW_ANONYMOUS` (dev bypass), `ARTIFACT_DIR`.

## Tests

```bash
pytest -v --cov
# 60+ tests: RWR convergence/ranking/isolated/self-loops/restart extremes, multi-channel shapes, HPO similarity graph, fusion forward, recall@k/AUPRC edge cases, leave-genes-out no-leak/deterministic, STRING/HPO/HGNC/DisGeNET parsers (inline fixtures + comment/blank/header-variant/gz edge cases, genes_to_disease, phenotype.hpoa), API validation (422 for bad top_k/restart_prob/empty seeds, 403 vs 503 token, Bearer vs X-Service-Token), release-gate fail-closed, schema sanitization (strip/H).
```

## Honest Current Status

- [x] Sparse RWR + multi-channel + FusionMLP + evaluation + parsers + API gate + frontend abstention + tests — all implemented and passing on synthetic fixtures.
- [ ] No real STRING/HPO/HGNC/DisGeNET download or training has run in this sandbox (by design — large downloads/training deferred to Kaggle/Modal).
- [ ] No approved artifact; `MODEL_RELEASE_APPROVED` defaults to unapproved.
- Next for coordinator: run real download + `data_pipeline.train` on Kaggle/Modal, compare fusion vs RWR baseline, set release env vars if metrics justify it.

## License

MIT
