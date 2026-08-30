"""
Parsers for real data sources: STRING, HPO, HGNC, DisGeNET.

All parsers operate on file paths (downloaded real files). They do NOT
assume anonymous network access for DisGeNET (which requires account/API key).

File format references:
- STRING: protein.links.v12.0.txt.gz -> header: protein1 protein2 combined_score
  Values like "9606.ENSP00000000233 9606.ENSP00000003084 150"
- HPO: genes_to_phenotype.txt or phenotype.hpoa -> HPO term <-> gene mappings
- HGNC: hgnc_complete_set.txt -> TSV with hgnc_id, symbol, entrez_id, ensembl_gene_id
- DisGeNET: curated gene-disease associations -> TSV with geneId, geneSymbol, diseaseId, diseaseName, score
"""
from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# STRING
# ---------------------------------------------------------------------------

def parse_string_links(
    path: str | Path,
    min_score: int = 400,
    max_edges: int | None = None,
) -> pd.DataFrame:
    """
    Parse STRING protein.links file. Handles .gz or plain text.
    Filters by combined_score >= min_score.
    Returns DataFrame with columns [protein1, protein2, combined_score]
    """
    path = Path(path)
    opener = gzip.open if str(path).endswith(".gz") else open

    rows = []
    with opener(path, "rt") as f:  # type: ignore
        header = f.readline().strip().split()
        # Expect protein1 protein2 combined_score (may have extra whitespace)
        # Normalize header
        col_idx = {h: i for i, h in enumerate(header)}
        p1_i = col_idx.get("protein1", 0)
        p2_i = col_idx.get("protein2", 1)
        sc_i = col_idx.get("combined_score", 2)

        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) <= max(p1_i, p2_i, sc_i):
                continue
            try:
                score = int(parts[sc_i])
            except ValueError:
                continue
            if score < min_score:
                continue
            rows.append((parts[p1_i], parts[p2_i], score))
            if max_edges and len(rows) >= max_edges:
                break

    df = pd.DataFrame(rows, columns=["protein1", "protein2", "combined_score"])
    return df


def string_protein_to_gene(protein_id: str) -> str:
    """Extract ENSP part from STRING ID like '9606.ENSP00000000233'."""
    if "." in protein_id:
        return protein_id.split(".", 1)[1]
    return protein_id


def parse_string_info(path: str | Path) -> dict[str, str]:
    """
    Parse STRING protein.info file (protein_id  preferred_name  protein_size  annotation).
    Returns dict mapping STRING protein_id (e.g. '9606.ENSP00000000233') -> gene symbol
    (preferred_name). This is the correct, direct STRING-native id->gene-symbol mapping
    (avoids fragile cross-referencing through HGNC/Ensembl gene ids).
    """
    path = Path(path)
    opener = gzip.open if str(path).endswith(".gz") else open
    mapping: dict[str, str] = {}
    with opener(path, "rt") as f:  # type: ignore
        header = f.readline().strip().split("\t")
        col_idx = {h: i for i, h in enumerate(header)}
        pid_i = col_idx.get("#string_protein_id", col_idx.get("string_protein_id", 0))
        name_i = col_idx.get("preferred_name", 1)
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(pid_i, name_i):
                continue
            mapping[parts[pid_i]] = parts[name_i]
    return mapping


def parse_genes_to_disease(path: str | Path) -> dict[str, set[str]]:
    """
    Parse HPO genes_to_disease.txt (real, public, no-auth annotation file).
    Real columns (ncbi_gene_id, gene_symbol, association_type, disease_id, source, ...).
    Returns dict mapping disease_id (e.g. 'OMIM:154700') -> set of gene symbols.
    """
    path = Path(path)
    df = pd.read_csv(path, sep="\t", dtype=str, comment="#", low_memory=False)
    cols_lower = {c.lower().lstrip("#"): c for c in df.columns}
    sym_col = cols_lower.get("gene_symbol") or cols_lower.get("gene-symbol") or cols_lower.get("gene_symbol.1")
    dis_col = cols_lower.get("disease_id") or cols_lower.get("disease-id")
    if sym_col is None or dis_col is None:
        raise ValueError(f"genes_to_disease.txt: could not find gene/disease columns in {list(df.columns)}")
    out: dict[str, set[str]] = {}
    for sym, dis in zip(df[sym_col], df[dis_col]):
        if pd.isna(sym) or pd.isna(dis):
            continue
        out.setdefault(dis, set()).add(sym)
    return out


def build_gene_hpo_terms(path: str | Path) -> dict[str, set[str]]:
    """
    Parse HPO genes_to_phenotype.txt into gene_symbol -> set of HPO term IDs.
    Used to build a phenotype-similarity gene-gene graph (independent evidence
    channel from the PPI network).
    """
    df = parse_hpo_genes_to_phenotype(path)
    out: dict[str, set[str]] = {}
    if "gene_symbol" not in df.columns or "hpo_id" not in df.columns:
        return out
    for sym, hpo in zip(df["gene_symbol"], df["hpo_id"]):
        if pd.isna(sym) or pd.isna(hpo):
            continue
        out.setdefault(sym, set()).add(hpo)
    return out


# ---------------------------------------------------------------------------
# HGNC
# ---------------------------------------------------------------------------

def parse_hgnc(path: str | Path) -> pd.DataFrame:
    """
    Parse HGNC complete set TSV.
    Returns DataFrame (pass-through with at least hgnc_id, symbol, entrez_id, ensembl_gene_id).
    """
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    # Normalize column names to lowercase for robustness
    # Keep original but ensure key columns exist
    return df


# ---------------------------------------------------------------------------
# HPO
# ---------------------------------------------------------------------------

def parse_hpo_genes_to_phenotype(path: str | Path) -> pd.DataFrame:
    """
    Parse HPO genes_to_phenotype.txt or phenotype.hpoa.
    Real format (genes_to_phenotype.txt):
      entrez-gene-id  entrez-gene-symbol  HPO-Term-Name  HPO-Term-ID ...
    phenotype.hpoa format:
      database_id  disease_name  qualifier  hpo_id  reference  evidence  onset  frequency  sex  modifier  aspect  biocuration
    This parser handles both by checking header.
    Returns DataFrame with columns [gene_symbol, hpo_id, hpo_name] (best effort).
    """
    path = Path(path)
    # Read a few lines to detect format
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
        # Peek header
        header = first_line.strip().split("\t") if "\t" in first_line else first_line.strip().split()

    # Use pandas with header detection
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, comment="#")

    # Normalize to standard columns
    cols_lower = {c.lower(): c for c in df.columns}

    # genes_to_phenotype.txt typical columns
    if "entrez-gene-symbol" in cols_lower:
        sym_col = cols_lower["entrez-gene-symbol"]
        hpo_id_col = cols_lower.get("hpo-term-id", cols_lower.get("hpo_term_id", None))
        hpo_name_col = cols_lower.get("hpo-term-name", None)
        out = pd.DataFrame()
        out["gene_symbol"] = df[sym_col]
        if hpo_id_col:
            out["hpo_id"] = df[hpo_id_col]
        if hpo_name_col:
            out["hpo_name"] = df[hpo_name_col]
        return out

    # phenotype.hpoa format
    if "database_id" in cols_lower:
        # database_id is like OMIM:123456
        # hpo_id column
        hpo_col = cols_lower.get("hpo_id", cols_lower.get("hpo_term_id"))
        out = pd.DataFrame()
        out["disease_id"] = df[cols_lower["database_id"]]
        if hpo_col:
            out["hpo_id"] = df[hpo_col]
        # No direct gene_symbol in phenotype.hpoa; return as is
        # Caller should join via other mapping
        return out

    # Fallback: return raw
    return df


# ---------------------------------------------------------------------------
# DisGeNET
# ---------------------------------------------------------------------------

def parse_disgenet(path: str | Path, min_score: float = 0.0) -> pd.DataFrame:
    """
    Parse DisGeNET curated gene-disease association file (TSV/CSV).
    Expected columns include: geneId, geneSymbol, diseaseId, diseaseName, score
    (Actual column names vary by export version; we handle case-insensitive.)

    min_score filters by DisGeNET score column if present.
    """
    path = Path(path)
    # Detect separator: try tab first
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    if df.shape[1] == 1:
        # Try comma
        df = pd.read_csv(path, sep=",", dtype=str, low_memory=False)

    cols_lower = {c.lower(): c for c in df.columns}

    # Try to filter by score
    score_col = None
    for cand in ["score", "scoregda", "gda_score", "dsi", "dpi"]:
        if cand in cols_lower:
            score_col = cols_lower[cand]
            break

    if score_col and min_score > 0:
        # Convert to numeric, filter
        df["_score_num"] = pd.to_numeric(df[score_col], errors="coerce")
        df = df[df["_score_num"] >= min_score].drop(columns=["_score_num"])

    return df


# ---------------------------------------------------------------------------
# Utility: build mapping from gene symbols to indices
# ---------------------------------------------------------------------------

def build_gene_index(symbols: list[str]) -> dict[str, int]:
    """Map gene symbols to integer indices."""
    return {s: i for i, s in enumerate(symbols)}
