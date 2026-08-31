"""Pydantic schemas for API request/response."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str = Field(default="ok", description="Service health status")
    model_approved: bool = Field(description="Whether release gate is approved (fail-closed)")
    artifact_revision: str | None = Field(default=None, description="Approved artifact revision if set")


class PrioritizeRequest(BaseModel):
    disease_name: str | None = Field(default=None, description="Disease name (e.g. Marfan syndrome)")
    seed_genes: list[str] | None = Field(default=None, description="List of HGNC gene symbols known for disease")
    hpo_terms: list[str] | None = Field(default=None, description="List of HPO term IDs (e.g. HP:0001377)")
    top_k: int = Field(default=50, ge=1, le=500, description="Number of top genes to return")
    restart_prob: float = Field(default=0.3, ge=0.0, le=1.0, description="RWR restart probability")

    @field_validator("disease_name", mode="before")
    @classmethod
    def _strip_disease(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            return v
        s = v.strip()
        return s if s else None

    @field_validator("seed_genes", mode="before")
    @classmethod
    def _clean_seed_genes(cls, v):
        if v is None:
            return None
        if not isinstance(v, list):
            return v
        cleaned: list[str] = []
        for item in v:
            if not isinstance(item, str):
                continue
            s = item.strip()
            if not s:
                continue
            # HGNC symbols are typically 1-15 chars alphanumeric with -/_ .
            # Reject obviously malformed entries (e.g. strings >30 chars, containing spaces inside)
            if len(s) > 30:
                continue
            if re.search(r"\s", s):
                continue
            cleaned.append(s)
        return cleaned if cleaned else None

    @field_validator("hpo_terms", mode="before")
    @classmethod
    def _clean_hpo_terms(cls, v):
        if v is None:
            return None
        if not isinstance(v, list):
            return v
        cleaned: list[str] = []
        for item in v:
            if not isinstance(item, str):
                continue
            s = item.strip().upper()
            if not s:
                continue
            # Real HPO terms are like HP:0001377
            if not re.match(r"^HP:\d{7}$", s):
                # Keep but normalize: if it looks like HP:xxx with wrong padding, still allow for validation downstream
                # But reject entries with spaces or extreme length
                if len(s) > 20 or re.search(r"\s", s):
                    continue
            cleaned.append(s)
        return cleaned if cleaned else None


class GeneScore(BaseModel):
    rank: int = Field(description="1-based rank")
    gene_symbol: str = Field(description="HGNC gene symbol")
    score: float = Field(description="Prioritization score (higher = more likely)")


class PrioritizeResponse(BaseModel):
    status: Literal["ok", "abstained"] = Field(description="Result status; abstained when no approved model")
    message: str | None = Field(default=None, description="Human-readable message (especially for abstention)")
    artifact_revision: str | None = Field(default=None, description="Artifact revision used for inference")
    results: list[GeneScore] | None = Field(default=None, description="Ranked gene list when status=ok")
    method: str | None = Field(default=None, description="Method name (e.g. rwr_multi_channel+mlp_fusion)")
