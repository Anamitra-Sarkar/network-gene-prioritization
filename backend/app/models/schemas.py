"""Pydantic schemas for API request/response."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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
