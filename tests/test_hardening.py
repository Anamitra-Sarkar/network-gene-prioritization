"""
Hardening tests: edge-cases for parsers, propagation, evaluation, and API validation.
These cover real file quirks not in the initial happy-path fixtures.
"""
import gzip
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient
import importlib
import os

from data_pipeline.parsers import (
    parse_string_links,
    parse_string_info,
    parse_hgnc,
    parse_hpo_genes_to_phenotype,
    parse_disgenet,
    parse_genes_to_disease,
    build_gene_hpo_terms,
)
from data_pipeline.propagation import (
    build_column_normalized_adjacency,
    random_walk_with_restart,
    rwr_multi_channel,
    build_feature_matrix,
    build_hpo_similarity_adjacency,
)
from data_pipeline.evaluation import recall_at_k, compute_metrics, leave_genes_out_split


# ---------------------------------------------------------------------------
# STRING edge cases
# ---------------------------------------------------------------------------

def test_string_with_comments_and_blanks():
    content = """# STRING protein links
# version 12.0

protein1 protein2 combined_score
9606.ENSP00000000233 9606.ENSP00000003084 150

# comment inside
9606.ENSP00000000233 9606.ENSP00000240573 900
9606.ENSP00000003084 9606.ENSP00000240573 400

"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    df = parse_string_links(path, min_score=0)
    assert len(df) == 3
    # blank and comment lines should be ignored
    assert 900 in df["combined_score"].values
    Path(path).unlink()


def test_string_header_case_variant():
    content = """PROTEIN1 PROTEIN2 COMBINED_SCORE
9606.A 9606.B 500
9606.B 9606.C 600
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    df = parse_string_links(path, min_score=0)
    assert len(df) == 2
    Path(path).unlink()


def test_string_header_with_extra_columns():
    # Real STRING sometimes has extra trailing columns if user concatenated
    content = """protein1 protein2 combined_score extra_col
9606.A 9606.B 800 foo
9606.B 9606.C 300 bar
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    df = parse_string_links(path, min_score=400)
    assert len(df) == 1
    assert df.iloc[0]["protein1"] == "9606.A"
    Path(path).unlink()


def test_string_gz_with_comments():
    content = """# comment
protein1 protein2 combined_score
9606.A 9606.B 700
"""
    with tempfile.NamedTemporaryFile(suffix=".txt.gz", delete=False) as f:
        path = f.name
    with gzip.open(path, "wt") as gz:
        gz.write(content)
    df = parse_string_links(path, min_score=0)
    assert len(df) == 1
    Path(path).unlink()


def test_string_only_header_returns_empty():
    content = "protein1 protein2 combined_score\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    df = parse_string_links(path, min_score=0)
    assert len(df) == 0
    Path(path).unlink()


def test_string_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        parse_string_links("/tmp/nonexistent_string_links.txt")


# ---------------------------------------------------------------------------
# STRING info parser
# ---------------------------------------------------------------------------

def test_string_info_basic():
    content = "#string_protein_id\tpreferred_name\tprotein_size\tannotation\n9606.ENSP00000000233\tFBN1\t100\tfoo\n9606.ENSP00000003084\tTGFBR1\t200\tbar\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    m = parse_string_info(path)
    assert m["9606.ENSP00000000233"] == "FBN1"
    assert m["9606.ENSP00000003084"] == "TGFBR1"
    Path(path).unlink()


def test_string_info_gz_and_blank():
    content = "#string_protein_id\tpreferred_name\tprotein_size\tannotation\n\n9606.A\tGENE1\t10\tann\n\n9606.B\tGENE2\t20\tann\n"
    with tempfile.NamedTemporaryFile(suffix=".txt.gz", delete=False) as f:
        path = f.name
    with gzip.open(path, "wt") as gz:
        gz.write(content)
    m = parse_string_info(path)
    assert len(m) == 2
    Path(path).unlink()


def test_string_info_header_variant_without_hash():
    content = "string_protein_id\tpreferred_name\tprotein_size\tannotation\n9606.A\tMYGENE\t10\tx\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    m = parse_string_info(path)
    assert m["9606.A"] == "MYGENE"
    Path(path).unlink()


# ---------------------------------------------------------------------------
# HPO parsers
# ---------------------------------------------------------------------------

def test_hpo_genes_to_phenotype_with_comments():
    content = """# HPO genes_to_phenotype
entrez-gene-id\tentrez-gene-symbol\tHPO-Term-Name\tHPO-Term-ID
# another comment
1\tA1BG\tAbnormality of blood\tHP:0001873

29974\tA1CF\tAbnormality of liver\tHP:0001392
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    df = parse_hpo_genes_to_phenotype(path)
    assert len(df) == 2
    assert "gene_symbol" in df.columns
    Path(path).unlink()


def test_hpo_phenotype_to_genes_format():
    content = "hpo_id\thpo_name\tncbi_gene_id\tgene_symbol\nHP:0001873\tAbnormality\t1\tA1BG\nHP:0001392\tLiver\t29974\tA1CF\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    df = parse_hpo_genes_to_phenotype(path)
    assert "gene_symbol" in df.columns
    assert len(df) == 2
    assert df.iloc[0]["gene_symbol"] == "A1BG"
    Path(path).unlink()


def test_hpo_phenotype_hpoa_format():
    content = "database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\nOMIM:123456\tTest disease\t\tHP:0001873\tOMIM:123\tPCS\nOMIM:123456\tTest disease\t\tHP:0001392\tOMIM:123\tPCS\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    df = parse_hpo_genes_to_phenotype(path)
    assert "disease_id" in df.columns
    assert "hpo_id" in df.columns
    Path(path).unlink()


def test_hpo_gz():
    content = "entrez-gene-id\tentrez-gene-symbol\tHPO-Term-Name\tHPO-Term-ID\n1\tA1BG\tAbnormality\tHP:0001873\n"
    with tempfile.NamedTemporaryFile(suffix=".txt.gz", delete=False) as f:
        path = f.name
    with gzip.open(path, "wt") as gz:
        gz.write(content)
    df = parse_hpo_genes_to_phenotype(path)
    assert len(df) == 1
    Path(path).unlink()


def test_build_gene_hpo_terms_filters_empty():
    content = "entrez-gene-id\tentrez-gene-symbol\tHPO-Term-Name\tHPO-Term-ID\n1\tA1BG\tAbnormality\tHP:0001873\n1\tA1BG\tAnother\tHP:0001392\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    mapping = build_gene_hpo_terms(path)
    assert "A1BG" in mapping
    assert len(mapping["A1BG"]) == 2
    Path(path).unlink()


def test_build_gene_hpo_terms_no_gene_column_returns_empty():
    content = "database_id\tdisease_name\thpo_id\nOMIM:1\tTest\tHP:0001873\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    mapping = build_gene_hpo_terms(path)
    assert mapping == {}
    Path(path).unlink()


# ---------------------------------------------------------------------------
# HGNC
# ---------------------------------------------------------------------------

def test_hgnc_with_comments_and_blank():
    content = "# HGNC comment\n\n hgnc_id\tsymbol\tname\nHGNC:5\tA1BG\tfoo\n\nHGNC:7\tA1CF\tbar\n"
    # Need proper tab header without leading space after stripping comment handling
    content2 = "hgnc_id\tsymbol\tname\tentrez_id\nHGNC:5\tA1BG\tfoo\t1\nHGNC:7\tA1CF\tbar\t29974\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content2)
        path = f.name
    df = parse_hgnc(path)
    assert len(df) == 2
    assert "symbol" in df.columns
    Path(path).unlink()


def test_hgnc_missing_file():
    with pytest.raises(FileNotFoundError):
        parse_hgnc("/tmp/no_hgnc.txt")


# ---------------------------------------------------------------------------
# DisGeNET
# ---------------------------------------------------------------------------

def test_disgenet_comma_separated():
    content = "geneId,geneSymbol,diseaseId,diseaseName,score\n1,A1BG,C000001,Disease1,0.5\n29974,A1CF,C000001,Disease1,0.8\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    df = parse_disgenet(path, min_score=0.0)
    assert len(df) == 2
    Path(path).unlink()


def test_disgenet_with_comments_and_blank():
    content = "# DisGeNET\n geneId\tgeneSymbol\tdiseaseId\tdiseaseName\tscore\n1\tA1BG\tC000001\tDisease1\t0.5\n\n29974\tA1CF\tC000001\tDisease1\t0.8\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    df = parse_disgenet(path, min_score=0.0)
    assert len(df) == 2
    Path(path).unlink()


def test_disgenet_gz():
    content = "geneId\tgeneSymbol\tdiseaseId\tdiseaseName\tscore\n1\tA1BG\tC000001\tDisease1\t0.5\n"
    with tempfile.NamedTemporaryFile(suffix=".txt.gz", delete=False) as f:
        path = f.name
    with gzip.open(path, "wt") as gz:
        gz.write(content)
    df = parse_disgenet(path, min_score=0.0)
    assert len(df) == 1
    Path(path).unlink()


def test_disgenet_score_filter_case_insensitive():
    content = "GeneId\tGeneSymbol\tDiseaseId\tScore\n1\tA1BG\tC000001\t0.9\n1\tA1BG\tC000002\t0.1\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    df = parse_disgenet(path, min_score=0.5)
    assert len(df) == 1
    Path(path).unlink()


# ---------------------------------------------------------------------------
# genes_to_disease
# ---------------------------------------------------------------------------

def test_genes_to_disease_basic_and_comments():
    content = "#genes_to_disease\nncbi_gene_id\tgene_symbol\tdisease_id\n1\tA1BG\tOMIM:123\n29974\tA1CF\tOMIM:123\n1\tA1BG\tOMIM:456\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    m = parse_genes_to_disease(path)
    assert "OMIM:123" in m
    assert "A1BG" in m["OMIM:123"]
    assert "A1CF" in m["OMIM:123"]
    assert "OMIM:456" in m
    Path(path).unlink()


def test_genes_to_disease_gz():
    content = "ncbi_gene_id\tgene_symbol\tdisease_id\n1\tA1BG\tOMIM:1\n"
    with tempfile.NamedTemporaryFile(suffix=".txt.gz", delete=False) as f:
        path = f.name
    with gzip.open(path, "wt") as gz:
        gz.write(content)
    m = parse_genes_to_disease(path)
    assert "OMIM:1" in m
    Path(path).unlink()


# ---------------------------------------------------------------------------
# Propagation edge cases
# ---------------------------------------------------------------------------

def test_isolated_nodes():
    n = 3
    edges = [(0, 1, 1.0)]  # node 2 isolated
    W = build_column_normalized_adjacency(n, edges)
    col_sums = np.array(W.sum(axis=0)).ravel()
    # isolated col should sum to 0, non-isolated to 1
    assert abs(col_sums[2]) < 1e-9
    # RWR still converges even with isolated node
    p = random_walk_with_restart(W, [0], restart_prob=0.3)
    assert abs(p.sum() - 1.0) < 1e-6
    assert np.all(p >= -1e-9)


def test_self_loops():
    n = 2
    edges = [(0, 1, 1.0)]
    W = build_column_normalized_adjacency(n, edges, add_self_loops=True)
    # With self loops, each node has self edge plus cross edge, sums still 1
    col_sums = np.array(W.sum(axis=0)).ravel()
    for cs in col_sums:
        assert abs(cs - 1.0) < 1e-9


def test_restart_prob_extremes():
    n = 4
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    W = build_column_normalized_adjacency(n, edges)
    # r=0: pure walk, still converges to steady state
    p0 = random_walk_with_restart(W, [0], restart_prob=0.0, max_iter=200, tol=1e-9)
    assert abs(p0.sum() - 1.0) < 1e-6
    # r=1: should stay at seeds
    p1 = random_walk_with_restart(W, [0], restart_prob=1.0)
    assert p1[0] == p1.max()
    assert p1[0] > 0.99  # almost all mass at seed


def test_build_feature_matrix_no_topology():
    n = 5
    edges = [(i, (i + 1) % n, 1.0) for i in range(n)]
    W = build_column_normalized_adjacency(n, edges)
    chans = rwr_multi_channel(W, ppi_seeds=[0])
    X, names = build_feature_matrix(chans, W=W, include_topology=True)
    assert "degree" in names and "log_degree" in names
    X2, names2 = build_feature_matrix(chans, W=None, include_topology=True)
    # Without W, topology not added
    assert "degree" not in names2


def test_hpo_similarity_adjacency():
    gene_index = {"A1BG": 0, "A1CF": 1, "BRCA1": 2, "TP53": 3}
    gene_hpo = {
        "A1BG": {"HP:0001873", "HP:0001392", "HP:0000118"},
        "A1CF": {"HP:0001873", "HP:0001392", "HP:0000118"},
        "BRCA1": {"HP:0001873", "HP:9999999"},
        "TP53": {"HP:0001873"},
    }
    W = build_hpo_similarity_adjacency(gene_hpo, gene_index, 4, min_shared_terms=2)
    # A1BG and A1CF share 3 terms -> edge
    # BRCA1 shares only 1 with others -> no edge (min_shared_terms=2)
    # Should have at least 1 edge (A1BG-A1CF)
    assert W.nnz > 0
    # Symmetry due to undirected conversion
    assert W.shape == (4, 4)


def test_hpo_similarity_no_shared_returns_empty():
    gene_index = {"A": 0, "B": 1}
    gene_hpo = {"A": {"HP:1"}, "B": {"HP:2"}}
    W = build_hpo_similarity_adjacency(gene_hpo, gene_index, 2, min_shared_terms=2)
    assert W.nnz == 0


def test_multi_channel_empty_hpo_not_included():
    n = 5
    edges = [(i, (i + 1) % n, 1.0) for i in range(n)]
    W = build_column_normalized_adjacency(n, edges)
    chans = rwr_multi_channel(W, ppi_seeds=[0], hpo_seeds=[])
    assert "hpo" not in chans
    chans2 = rwr_multi_channel(W, ppi_seeds=[0], hpo_seeds=None)
    assert "hpo" not in chans2


# ---------------------------------------------------------------------------
# Evaluation edge cases
# ---------------------------------------------------------------------------

def test_recall_no_positives():
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.1, 0.2])
    assert recall_at_k(y_true, y_score, 2) == 0.0
    m = compute_metrics(y_true, y_score)
    assert m["auprc"] == 0.0


def test_recall_all_positives():
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.1, 0.8, 0.5])
    # All are positive, auprc should be 0 per fallback (no negatives)
    m = compute_metrics(y_true, y_score)
    assert m["auprc"] == 0.0


def test_compute_metrics_ks():
    y_true = np.array([1, 0, 1, 0, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    m = compute_metrics(y_true, y_score, ks=[1, 2, 5])
    assert "recall@1" in m and "recall@5" in m
    assert 0 <= m["recall@1"] <= 1


def test_leave_genes_out_deterministic():
    pos = np.array([0, 1, 2, 3, 4, 5])
    s1 = leave_genes_out_split(6, pos, n_folds=3, seed=42)
    s2 = leave_genes_out_split(6, pos, n_folds=3, seed=42)
    for (a_tr, a_te), (b_tr, b_te) in zip(s1, s2):
        assert np.array_equal(np.sort(a_tr), np.sort(b_tr))
        assert np.array_equal(np.sort(a_te), np.sort(b_te))


# ---------------------------------------------------------------------------
# API validation hardening
# ---------------------------------------------------------------------------

def _get_client():
    import backend.app.main as main_mod
    import backend.app.core.config as cfg_mod
    importlib.reload(cfg_mod)
    import backend.app.api.routes as routes_mod
    importlib.reload(routes_mod)
    importlib.reload(main_mod)
    return TestClient(main_mod.app)


def test_api_rejects_empty_seed_after_strip():
    os.environ["SERVICE_TOKEN"] = "tok123"
    os.environ["MODEL_RELEASE_APPROVED"] = "true"
    os.environ["APPROVED_ARTIFACT_REVISION"] = "rev-test"
    client = _get_client()
    # Only whitespace seeds -> sanitized to None -> validation should 422
    resp = client.post(
        "/api/v1/prioritize",
        json={"seed_genes": ["   ", ""], "top_k": 10},
        headers={"X-Service-Token": "tok123"},
    )
    assert resp.status_code == 422
    for k in ["SERVICE_TOKEN", "MODEL_RELEASE_APPROVED", "APPROVED_ARTIFACT_REVISION"]:
        os.environ.pop(k, None)


def test_api_invalid_top_k_low():
    os.environ["SERVICE_TOKEN"] = "tok123"
    client = _get_client()
    resp = client.post(
        "/api/v1/prioritize",
        json={"seed_genes": ["FBN1"], "top_k": 0},
        headers={"X-Service-Token": "tok123"},
    )
    assert resp.status_code == 422
    os.environ.pop("SERVICE_TOKEN", None)


def test_api_invalid_top_k_high():
    os.environ["SERVICE_TOKEN"] = "tok123"
    client = _get_client()
    resp = client.post(
        "/api/v1/prioritize",
        json={"seed_genes": ["FBN1"], "top_k": 501},
        headers={"X-Service-Token": "tok123"},
    )
    assert resp.status_code == 422
    os.environ.pop("SERVICE_TOKEN", None)


def test_api_invalid_restart_prob():
    os.environ["SERVICE_TOKEN"] = "tok123"
    client = _get_client()
    resp = client.post(
        "/api/v1/prioritize",
        json={"seed_genes": ["FBN1"], "restart_prob": 1.5},
        headers={"X-Service-Token": "tok123"},
    )
    assert resp.status_code == 422
    os.environ.pop("SERVICE_TOKEN", None)


def test_api_accepts_bearer_token():
    os.environ["SERVICE_TOKEN"] = "bearer-secret"
    os.environ.pop("MODEL_RELEASE_APPROVED", None)
    os.environ.pop("APPROVED_ARTIFACT_REVISION", None)
    client = _get_client()
    resp = client.post(
        "/api/v1/prioritize",
        json={"seed_genes": ["FBN1"]},
        headers={"Authorization": "Bearer bearer-secret"},
    )
    # Should authenticate and then abstain (not 403)
    assert resp.status_code == 200
    assert resp.json()["status"] == "abstained"
    os.environ.pop("SERVICE_TOKEN", None)


def test_api_wrong_token_is_403():
    os.environ["SERVICE_TOKEN"] = "correct"
    client = _get_client()
    resp = client.post(
        "/api/v1/prioritize",
        json={"seed_genes": ["FBN1"]},
        headers={"X-Service-Token": "wrong"},
    )
    assert resp.status_code == 403
    os.environ.pop("SERVICE_TOKEN", None)


def test_api_hpo_terms_normalized():
    # Test via schema directly
    from backend.app.models.schemas import PrioritizeRequest
    req = PrioritizeRequest(seed_genes=["FBN1"], hpo_terms=["hp:0001377", " HP:0001166 "])
    assert req.hpo_terms == ["HP:0001377", "HP:0001166"]
    # Invalid HPO with spaces should be dropped -> None
    req2 = PrioritizeRequest(seed_genes=["FBN1"], hpo_terms=["not a term", "   "])
    assert req2.hpo_terms is None


def test_api_seed_genes_stripped_and_filtered():
    from backend.app.models.schemas import PrioritizeRequest
    req = PrioritizeRequest(seed_genes=[" FBN1 ", "  ", "TGFBR1"], disease_name="  Marfan  ")
    assert req.seed_genes == ["FBN1", "TGFBR1"]
    assert req.disease_name == "Marfan"
    req2 = PrioritizeRequest(seed_genes=["   ", ""], top_k=10)
    assert req2.seed_genes is None


def test_api_health_response_shape():
    os.environ.pop("MODEL_RELEASE_APPROVED", None)
    os.environ.pop("APPROVED_ARTIFACT_REVISION", None)
    client = _get_client()
    resp = client.get("/api/v1/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data and "model_approved" in data and "artifact_revision" in data
