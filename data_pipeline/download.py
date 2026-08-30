"""
Download helpers for real data sources.

- STRING: public, no auth
- HPO: public, no auth
- HGNC: public, no auth
- DisGeNET: requires account/API key or manual download; this script accepts
  a pre-downloaded file OR DISGENET_API_KEY env var (if API access is configured).

No large downloads are attempted in CI/sandbox; these are for Kaggle/Modal runs.
"""
from __future__ import annotations

import os
from pathlib import Path
import urllib.request

STRING_URL = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
STRING_INFO_URL = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
HPO_GENES_TO_PHENOTYPE_URL = "https://ci.monarchinitiative.org/view/hpo/job/hpo.annotations/lastSuccessfulBuild/artifact/genes_to_phenotype.txt"
HPO_PHENOTYPE_HPOA_URL = "http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa"
HPO_GENES_TO_DISEASE_URLS = [
    "https://ci.monarchinitiative.org/view/hpo/job/hpo.annotations/lastSuccessfulBuild/artifact/genes_to_disease.txt",
    "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/genes_to_disease.txt",
]
HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"

# DisGeNET: recent versions require authentication. Documented here honestly.
# See https://disgenet.com/disgenet_docs/ for current access.
DISGENET_INFO = (
    "DisGeNET: recent versions require a free account + API key for full access. "
    "Obtain at https://disgenet.com/signup and set DISGENET_API_KEY or "
    "DISGENET_EMAIL/DISGENET_PASSWORD, or provide a pre-downloaded file via "
    "--disgenet-file / DISGENET_FILE."
)


def download_file(url: str, dest: str | Path, chunk_size: int = 8192) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def download_string(dest: str | Path = "data/raw/9606.protein.links.v12.0.txt.gz") -> Path:
    return download_file(STRING_URL, dest)


def download_hpo(dest: str | Path = "data/raw/genes_to_phenotype.txt") -> Path:
    # Try primary, fallback to phenotype.hpoa
    try:
        return download_file(HPO_GENES_TO_PHENOTYPE_URL, dest)
    except Exception as e:
        print(f"Primary HPO download failed ({e}), trying phenotype.hpoa")
        return download_file(HPO_PHENOTYPE_HPOA_URL, dest)


def download_hgnc(dest: str | Path = "data/raw/hgnc_complete_set.txt") -> Path:
    return download_file(HGNC_URL, dest)


def download_string_info(dest: str | Path = "data/raw/9606.protein.info.v12.0.txt.gz") -> Path:
    return download_file(STRING_INFO_URL, dest)


def download_genes_to_disease(dest: str | Path = "data/raw/genes_to_disease.txt") -> Path:
    last_exc: Exception | None = None
    for url in HPO_GENES_TO_DISEASE_URLS:
        try:
            return download_file(url, dest)
        except Exception as e:
            print(f"genes_to_disease download from {url} failed ({e}), trying next")
            last_exc = e
    raise RuntimeError(f"All genes_to_disease.txt sources failed: {last_exc}")


def disgenet_instructions() -> str:
    return DISGENET_INFO


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Download real data sources")
    p.add_argument("--all", action="store_true", help="Download all public sources (STRING, HPO, HGNC)")
    p.add_argument("--string", action="store_true")
    p.add_argument("--hpo", action="store_true")
    p.add_argument("--hgnc", action="store_true")
    p.add_argument("--string-info", action="store_true")
    p.add_argument("--genes-to-disease", action="store_true")
    p.add_argument("--data-dir", default="data/raw")
    args = p.parse_args()

    if args.all or args.string:
        download_string(Path(args.data_dir) / "9606.protein.links.v12.0.txt.gz")
    if args.all or args.hpo:
        download_hpo(Path(args.data_dir) / "genes_to_phenotype.txt")
    if args.all or args.hgnc:
        download_hgnc(Path(args.data_dir) / "hgnc_complete_set.txt")
    if args.all or args.string_info:
        download_string_info(Path(args.data_dir) / "9606.protein.info.v12.0.txt.gz")
    if args.all or args.genes_to_disease:
        download_genes_to_disease(Path(args.data_dir) / "genes_to_disease.txt")

    if not (args.all or args.string or args.hpo or args.hgnc or args.string_info or args.genes_to_disease):
        print(DISGENET_INFO)
        p.print_help()
