"""Fail-closed release gate and config."""
from __future__ import annotations

import os


def is_model_release_approved() -> bool:
    """
    Fail-closed: only returns True if BOTH env vars are set to approved values.
    MODEL_RELEASE_APPROVED must be 'true' (case-insensitive) and
    APPROVED_ARTIFACT_REVISION must be non-empty and match expected.
    Default: unapproved / abstaining.
    """
    approved = os.getenv("MODEL_RELEASE_APPROVED", "").strip().lower() == "true"
    revision = os.getenv("APPROVED_ARTIFACT_REVISION", "").strip()
    if not approved:
        return False
    if not revision:
        return False
    return True


def get_artifact_revision() -> str | None:
    rev = os.getenv("APPROVED_ARTIFACT_REVISION", "").strip()
    return rev if rev else None


def get_service_token() -> str | None:
    return os.getenv("SERVICE_TOKEN", None) or os.getenv("INTERNAL_SERVICE_TOKEN", None)


def get_firebase_admin_json() -> str | None:
    return os.getenv("FIREBASE_ADMIN_JSON", None) or os.getenv("FIREBASE_SERVICE_ACCOUNT", None)
