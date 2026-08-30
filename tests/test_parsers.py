"""
Tests for STRING/HPO/HGNC/DisGeNET parsers against small inline fixtures
matching real file formats (structurally realistic, not claimed real data).
"""
import gzip
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.parsers import (
    parse_string_links,
    parse_hgnc,
    parse_hpo_genes_to_phenotype,
    parse_disgenet,
)


# ---- STRING fixture -------------------------------------------------------
STRING_FIXTURE = """protein1 protein2 combined_score
9606.ENSP00000000233 9606.ENSP00000003084 150
9606.ENSP00000000233 9606.ENSP00000240573 900
9606.ENSP00000003084 9606.ENSP00000240573 400
9606.ENSP00000003084 9606.ENSP00000312345 700
"""

def test_parse_string_links_plain():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(STRING_FIXTURE)
        path = f.name
    # threshold 400 should exclude score 150 line
    df = parse_string_links(path, min_score=400)
    assert len(df) == 3
    assert set(df.columns) == {"protein1", "protein2", "combined_score"}
    assert 150 not in df["combined_score"].values
    Path(path).unlink()

def test_parse_string_links_gz():
    with tempfile.NamedTemporaryFile(suffix=".txt.gz", delete=False) as f:
        path = f.name
    with gzip.open(path, "wt") as gz:
        gz.write(STRING_FIXTURE)
    df = parse_string_links(path, min_score=0)
    assert len(df) == 4
    Path(path).unlink()

def test_parse_string_min_score_filter():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(STRING_FIXTURE)
        path = f.name
    df = parse_string_links(path, min_score=700)
    assert len(df) == 2  # 900 and 700
    Path(path).unlink()


# ---- HGNC fixture ---------------------------------------------------------
HGNC_FIXTURE = "hgnc_id\tsymbol\tname\tentrez_id\tensembl_gene_id\nHGNC:5\tA1BG\talpha-1-B glycoprotein\t1\tENSG00000121410\nHGNC:7\tA1CF\tAPOBEC1 complementation factor\t29974\tENSG00000148584\n"

def test_parse_hgnc():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(HGNC_FIXTURE)
        path = f.name
    df = parse_hgnc(path)
    assert len(df) == 2
    assert "symbol" in df.columns
    assert df.iloc[0]["symbol"] == "A1BG"
    Path(path).unlink()


# ---- HPO fixture (genes_to_phenotype) ------------------------------------
HPO_FIXTURE = "entrez-gene-id\tentrez-gene-symbol\tHPO-Term-Name\tHPO-Term-ID\n1\tA1BG\tAbnormality of blood\tHP:0001873\n29974\tA1CF\tAbnormality of liver\tHP:0001392\n"

def test_parse_hpo_genes_to_phenotype():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(HPO_FIXTURE)
        path = f.name
    df = parse_hpo_genes_to_phenotype(path)
    assert "gene_symbol" in df.columns
    assert "hpo_id" in df.columns
    assert len(df) == 2
    assert df.iloc[0]["gene_symbol"] == "A1BG"
    assert df.iloc[0]["hpo_id"] == "HP:0001873"
    Path(path).unlink()


# ---- DisGeNET fixture -----------------------------------------------------
DISGENET_FIXTURE = "geneId\tgeneSymbol\tdiseaseId\tdiseaseName\tscore\n1\tA1BG\tC0000001\tTest disease 1\t0.5\n29974\tA1CF\tC0000001\tTest disease 1\t0.8\n1\tA1BG\tC0000002\tTest disease 2\t0.3\n"

def test_parse_disgenet():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(DISGENET_FIXTURE)
        path = f.name
    df = parse_disgenet(path, min_score=0.0)
    assert len(df) == 3
    assert "geneSymbol" in df.columns or "genesymbol" in [c.lower() for c in df.columns]
    Path(path).unlink()

def test_parse_disgenet_score_filter():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(DISGENET_FIXTURE)
        path = f.name
    df = parse_disgenet(path, min_score=0.5)
    assert len(df) == 2
    Path(path).unlink()
