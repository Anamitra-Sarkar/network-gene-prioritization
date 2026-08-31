"""API routes: /healthz (public), /prioritize (service-token gated + release gate), /me (firebase)."""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.config import get_artifact_revision, is_model_release_approved
from backend.app.core.auth import require_firebase_user, require_service_token
from backend.app.models.schemas import (
    HealthResponse,
    PrioritizeRequest,
    PrioritizeResponse,
)

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse, tags=["public"])
async def healthz():
    approved = is_model_release_approved()
    return HealthResponse(
        status="ok",
        model_approved=approved,
        artifact_revision=get_artifact_revision(),
    )


@router.get("/me", tags=["user"])
async def me(user=Depends(require_firebase_user)):
    return {"uid": user.get("uid"), "email": user.get("email")}


@router.post("/prioritize", response_model=PrioritizeResponse, tags=["internal"])
async def prioritize(
    body: PrioritizeRequest,
    _svc=Depends(require_service_token),
):
    # Fail-closed release gate
    if not is_model_release_approved():
        return PrioritizeResponse(
            status="abstained",
            message="Research service, no approved model yet. Set MODEL_RELEASE_APPROVED=true and APPROVED_ARTIFACT_REVISION to enable.",
            artifact_revision=None,
            results=None,
            method=None,
        )

    # Validate input (after pydantic sanitization empty strings/lists become None)
    if not body.disease_name and not body.seed_genes:
        raise HTTPException(status_code=422, detail="Either disease_name or seed_genes must be provided (after trimming empty entries)")
    # Additional sanity: if seed_genes provided, ensure at least one non-empty after cleaning
    if body.seed_genes is not None and len(body.seed_genes) == 0:
        raise HTTPException(status_code=422, detail="seed_genes must contain at least one non-empty gene symbol")
    # Cap seed_genes length to avoid abuse (HGNC symbols list shouldn't be huge)
    if body.seed_genes is not None and len(body.seed_genes) > 500:
        raise HTTPException(status_code=422, detail="seed_genes list too large (max 500)")
    if body.hpo_terms is not None and len(body.hpo_terms) > 500:
        raise HTTPException(status_code=422, detail="hpo_terms list too large (max 500)")

    # If approved but no artifact yet, still abstain honestly
    artifact_dir = os.getenv("ARTIFACT_DIR", "artifacts")
    rev = get_artifact_revision()
    from pathlib import Path

    artifact_path = Path(artifact_dir) / rev if rev else None
    if artifact_path is None or not artifact_path.exists():
        return PrioritizeResponse(
            status="abstained",
            message=f"Model approved (revision={rev}) but no artifact found at {artifact_path}. Run training pipeline first.",
            artifact_revision=rev,
            results=None,
            method=None,
        )

    # Real inference would happen here via services/prioritization.py
    # For now return abstained with clear message
    return PrioritizeResponse(
        status="abstained",
        message="Artifact present but inference not yet wired in this build. See services/prioritization.py.",
        artifact_revision=rev,
        results=None,
        method="rwr_multi_channel+mlp_fusion",
    )
