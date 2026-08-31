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

    Hardened for real file quirks: skips comment lines (#), blank lines,
    handles header variants (case/whitespace), extra trailing columns, and
    gracefully handles missing optional whitespace.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"STRING file not found: {path}")
    opener = gzip.open if str(path).endswith(".gz") else open

    rows = []
    with opener(path, "rt") as f:  # type: ignore
        # Skip leading comment/blank lines to find real header
        header = None
        header_raw = None
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            header_raw = s
            header = s.split()
            break
        if header is None:
            return pd.DataFrame(columns=["protein1", "protein2", "combined_score"])
        # Normalize header case-insensitively
        col_idx_lower = {h.lower(): i for i, h in enumerate(header)}
        # Prefer lower-case lookup but fall back to positional
        p1_i = col_idx_lower.get("protein1", 0)
        p2_i = col_idx_lower.get("protein2", 1)
        sc_i = col_idx_lower.get("combined_score", 2)

        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
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

    Handles .gz, tab-separated with possible extra columns, comment/blank lines,
    and header variants (#string_protein_id vs string_protein_id, case-insensitive).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"STRING protein.info file not found: {path}")
    opener = gzip.open if str(path).endswith(".gz") else open
    mapping: dict[str, str] = {}
    with opener(path, "rt") as f:  # type: ignore
        # Find header skipping comments/blanks
        header = None
        col_idx: dict[str, int] = {}
        pid_i = 0
        name_i = 1
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") and "\t" not in line:
                # Be careful: real header is "#string_protein_id\tpreferred_name..."
                # which starts with # but IS the header. So detect tab presence.
                if "\t" not in line:
                    continue
            if header is None:
                # This is the header line
                header = line.rstrip("\n").split("\t")
                # Normalize: strip leading # and lower
                col_idx = {h.lstrip("#").lower(): i for i, h in enumerate(header)}
                # Also keep original for exact match
                for i, h in enumerate(header):
                    col_idx[h] = i
                    col_idx[h.lower()] = i
                pid_i = col_idx.get("string_protein_id", col_idx.get("#string_protein_id", 0))
                name_i = col_idx.get("preferred_name", 1)
                break
        if header is None:
            return mapping
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#") and "\t" not in line:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(pid_i, name_i):
                continue
            pid = parts[pid_i].strip()
            name = parts[name_i].strip()
            if not pid or not name:
                continue
            mapping[pid] = name
    return mapping


def parse_genes_to_disease(path: str | Path) -> dict[str, set[str]]:
    """
    Parse HPO genes_to_disease.txt (real, public, no-auth annotation file).
    Real columns (ncbi_gene_id, gene_symbol, association_type, disease_id, source, ...).
    Returns dict mapping disease_id (e.g. 'OMIM:154700') -> set of gene symbols.
    Handles comment lines, blank rows, header case variants, and gzipped files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"genes_to_disease file not found: {path}")
    # Handle gz transparently: pandas can read gz if needed, but we try both
    try:
        if str(path).endswith(".gz"):
            df = pd.read_csv(path, sep="\t", dtype=str, comment="#", low_memory=False, compression="gzip")
        else:
            df = pd.read_csv(path, sep="\t", dtype=str, comment="#", low_memory=False)
    except Exception:
        # Fallback without comment handling for unusual quoting
        df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, comment="#")
    if df.empty or df.shape[1] == 0:
        return {}
    # Drop fully-NA rows that might come from blank lines
    df = df.dropna(how="all")
    cols_lower = {c.lower().lstrip("#").strip(): c for c in df.columns}
    sym_col = cols_lower.get("gene_symbol") or cols_lower.get("gene-symbol") or cols_lower.get("gene_symbol.1") or cols_lower.get("gene symbol")
    dis_col = cols_lower.get("disease_id") or cols_lower.get("disease-id") or cols_lower.get("disease id")
    if sym_col is None or dis_col is None:
        raise ValueError(f"genes_to_disease.txt: could not find gene/disease columns in {list(df.columns)}")
    out: dict[str, set[str]] = {}
    for sym, dis in zip(df[sym_col], df[dis_col]):
        if pd.isna(sym) or pd.isna(dis):
            continue
        s = str(sym).strip()
        d = str(dis).strip()
        if not s or not d:
            continue
        out.setdefault(d, set()).add(s)
    return out


def build_gene_hpo_terms(path: str | Path) -> dict[str, set[str]]:
    """
    Parse HPO genes_to_phenotype.txt into gene_symbol -> set of HPO term IDs.
    Used to build a phenotype-similarity gene-gene graph (independent evidence
    channel from the PPI network). Handles missing columns gracefully (returns empty).
    """
    df = parse_hpo_genes_to_phenotype(path)
    out: dict[str, set[str]] = {}
    if "gene_symbol" not in df.columns or "hpo_id" not in df.columns:
        return out
    for sym, hpo in zip(df["gene_symbol"], df["hpo_id"]):
        if pd.isna(sym) or pd.isna(hpo):
            continue
        s = str(sym).strip()
        h = str(hpo).strip()
        if not s or not h or h.lower() == "nan":
            continue
        out.setdefault(s, set()).add(h)
    return out


# ---------------------------------------------------------------------------
# HGNC
# ---------------------------------------------------------------------------

def parse_hgnc(path: str | Path) -> pd.DataFrame:
    """
    Parse HGNC complete set TSV.
    Returns DataFrame (pass-through with at least hgnc_id, symbol, entrez_id, ensembl_gene_id).
    Handles comment lines (#), blank rows, and gzipped files. Preserves original columns
    but ensures header is read correctly even with leading comment block.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HGNC file not found: {path}")
    # HGNC file may start with comment? Real file doesn't, but be defensive.
    if str(path).endswith(".gz"):
        df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, comment="#", compression="gzip")
    else:
        df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, comment="#")
    if df.empty:
        return df
    df = df.dropna(how="all")
    # Strip BOM if present
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# HPO
# ---------------------------------------------------------------------------

def parse_hpo_genes_to_phenotype(path: str | Path) -> pd.DataFrame:
    """
    Parse HPO genes_to_phenotype.txt or phenotype.hpoa / phenotype_to_genes.txt.
    Real format (genes_to_phenotype.txt):
      entrez-gene-id  entrez-gene-symbol  HPO-Term-Name  HPO-Term-ID ...
    phenotype.hpoa format:
      database_id  disease_name  qualifier  hpo_id  reference  evidence  onset  frequency  sex  modifier  aspect  biocuration
    This parser handles all by checking header (case-insensitive, handles .gz,
    comment lines, blank lines, and header variants).
    Returns DataFrame with columns [gene_symbol, hpo_id, hpo_name] (best effort).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HPO file not found: {path}")

    # Handle gz or plain transparently
    import gzip as _gzip
    opener = _gzip.open if str(path).endswith(".gz") else open
    # Peek header skipping comments/blanks
    header_line = None
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:  # type: ignore
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            header_line = s
            break
    if header_line is None:
        return pd.DataFrame(columns=["gene_symbol", "hpo_id"])

    # Use pandas with header detection (handle gz)
    try:
        if str(path).endswith(".gz"):
            df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, comment="#", compression="gzip")
        else:
            df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, comment="#")
    except Exception:
        df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, comment="#")

    if df.empty:
        return pd.DataFrame(columns=["gene_symbol", "hpo_id"])

    df = df.dropna(how="all")
    # Normalize column names: strip BOM, whitespace, lower
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    cols_lower = {c.lower().strip(): c for c in df.columns}

    # genes_to_phenotype.txt typical columns (handle header variants with spaces/underscores/dashes)
    # Real header: "entrez-gene-id", "entrez-gene-symbol", "HPO-Term-Name", "HPO-Term-ID"
    # Also seen: entrez_gene_symbol, hpo_term_id etc.
    def _find_col(*cands):
        for cand in cands:
            if cand in cols_lower:
                return cols_lower[cand]
            # also try normalized without dashes/underscores
            norm_cand = cand.replace("-", "_")
            if norm_cand in cols_lower:
                return cols_lower[norm_cand]
            norm2 = cand.replace("_", "-")
            if norm2 in cols_lower:
                return cols_lower[norm2]
        return None

    entrez_sym = _find_col("entrez-gene-symbol", "entrez_gene_symbol", "gene_symbol", "gene-symbol")
    # If entrez-gene-symbol exists, this is genes_to_phenotype.txt
    if "entrez-gene-symbol" in cols_lower or "entrez_gene_symbol" in cols_lower:
        sym_col = cols_lower.get("entrez-gene-symbol") or cols_lower.get("entrez_gene_symbol")
        hpo_id_col = _find_col("hpo-term-id", "hpo_term_id", "hpo_id", "hpo-term_id")
        hpo_name_col = _find_col("hpo-term-name", "hpo_term_name", "hpo_name")
        out = pd.DataFrame()
        out["gene_symbol"] = df[sym_col].astype(str).str.strip()
        if hpo_id_col:
            out["hpo_id"] = df[hpo_id_col].astype(str).str.strip()
        else:
            out["hpo_id"] = ""
        if hpo_name_col:
            out["hpo_name"] = df[hpo_name_col].astype(str).str.strip()
        # Drop rows where both key cols are empty/NA
        out = out[~(out["gene_symbol"].isna() | (out["gene_symbol"] == ""))]
        return out

    # phenotype_to_genes.txt format (also a valid real gene<->HPO source, columns:
    # hpo_id, hpo_name, ncbi_gene_id, gene_symbol, ...)
    if "gene_symbol" in cols_lower and "hpo_id" in cols_lower:
        out = pd.DataFrame()
        out["gene_symbol"] = df[cols_lower["gene_symbol"]].astype(str).str.strip()
        out["hpo_id"] = df[cols_lower["hpo_id"]].astype(str).str.strip()
        if "hpo_name" in cols_lower:
            out["hpo_name"] = df[cols_lower["hpo_name"]].astype(str).str.strip()
        out = out[~(out["gene_symbol"].isna() | (out["gene_symbol"] == ""))]
        return out

    # phenotype.hpoa format
    if "database_id" in cols_lower:
        hpo_col = cols_lower.get("hpo_id", cols_lower.get("hpo_term_id", cols_lower.get("hpo-term-id")))
        out = pd.DataFrame()
        out["disease_id"] = df[cols_lower["database_id"]].astype(str).str.strip()
        if hpo_col:
            out["hpo_id"] = df[hpo_col].astype(str).str.strip()
        return out

    # Fallback: return raw (but still drop empty rows and strip)
    return df


# ---------------------------------------------------------------------------
# DisGeNET
# ---------------------------------------------------------------------------

def parse_disgenet(path: str | Path, min_score: float = 0.0) -> pd.DataFrame:
    """
    Parse DisGeNET curated gene-disease association file (TSV/CSV).
    Expected columns include: geneId, geneSymbol, diseaseId, diseaseName, score
    (Actual column names vary by export version; we handle case-insensitive,
    comments, blank lines, and .gz.)

    min_score filters by DisGeNET score column if present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DisGeNET file not found: {path}")
    # Detect separator: try tab first, handle gz and comments
    read_kwargs = dict(dtype=str, low_memory=False, comment="#")
    compression = "gzip" if str(path).endswith(".gz") else None
    if compression:
        read_kwargs["compression"] = compression  # type: ignore
    try:
        df = pd.read_csv(path, sep="\t", **read_kwargs)  # type: ignore
    except Exception:
        df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, comment="#")
    if df.shape[1] == 1:
        # Try comma (also handle single-column due to wrong sep)
        try:
            df = pd.read_csv(path, sep=",", **read_kwargs)  # type: ignore
        except Exception:
            df = pd.read_csv(path, sep=",", dtype=str, low_memory=False, comment="#")
    if df.empty:
        return df
    df = df.dropna(how="all")
    # Strip column name whitespace/BOM
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    cols_lower = {c.lower().strip(): c for c in df.columns}

    # Try to filter by score
    score_col = None
    for cand in ["score", "scoregda", "gda_score", "dsi", "dpi"]:
        if cand in cols_lower:
            score_col = cols_lower[cand]
            break
        # also try without underscore
        cand2 = cand.replace("_", "")
        if cand2 in cols_lower:
            score_col = cols_lower[cand2]
            break

    if score_col and min_score > 0:
        df["_score_num"] = pd.to_numeric(df[score_col], errors="coerce")
        df = df[df["_score_num"] >= min_score].drop(columns=["_score_num"])
    # Strip string cell whitespace
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace({"nan": pd.NA, "None": pd.NA})
    return df


# ---------------------------------------------------------------------------
# Utility: build mapping from gene symbols to indices
# ---------------------------------------------------------------------------

def build_gene_index(symbols: list[str]) -> dict[str, int]:
    """Map gene symbols to integer indices."""
    return {s: i for i, s in enumerate(symbols)}
