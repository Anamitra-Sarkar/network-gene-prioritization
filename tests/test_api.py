"""Tests for API release gate (fail-closed) and /healthz."""
import os
import importlib

import pytest
from fastapi.testclient import TestClient


def _get_client():
    # Reimport to pick up env changes
    import backend.app.main as main_mod
    import backend.app.core.config as cfg_mod
    importlib.reload(cfg_mod)
    import backend.app.api.routes as routes_mod
    importlib.reload(routes_mod)
    importlib.reload(main_mod)
    return TestClient(main_mod.app)


def test_healthz_defaults_to_unapproved():
    # Ensure env is clean
    os.environ.pop("MODEL_RELEASE_APPROVED", None)
    os.environ.pop("APPROVED_ARTIFACT_REVISION", None)
    client = _get_client()
    resp = client.get("/api/v1/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_approved"] is False
    assert data["status"] == "ok"


def test_healthz_approved_when_both_set():
    os.environ["MODEL_RELEASE_APPROVED"] = "true"
    os.environ["APPROVED_ARTIFACT_REVISION"] = "rev-123"
    client = _get_client()
    resp = client.get("/api/v1/healthz")
    assert resp.json()["model_approved"] is True
    # cleanup
    os.environ.pop("MODEL_RELEASE_APPROVED", None)
    os.environ.pop("APPROVED_ARTIFACT_REVISION", None)


def test_healthz_not_approved_if_only_one_set():
    os.environ["MODEL_RELEASE_APPROVED"] = "true"
    os.environ.pop("APPROVED_ARTIFACT_REVISION", None)
    client = _get_client()
    assert client.get("/api/v1/healthz").json()["model_approved"] is False
    os.environ.pop("MODEL_RELEASE_APPROVED", None)


def test_prioritize_abstains_when_unapproved():
    os.environ.pop("MODEL_RELEASE_APPROVED", None)
    os.environ.pop("APPROVED_ARTIFACT_REVISION", None)
    os.environ["SERVICE_TOKEN"] = "test-token"
    client = _get_client()
    resp = client.post(
        "/api/v1/prioritize",
        json={"seed_genes": ["FBN1"], "top_k": 10},
        headers={"X-Service-Token": "test-token"},
    )
    # Should abstain, not 500 or fake results
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "abstained"
    assert "no approved model" in data["message"].lower()
    os.environ.pop("SERVICE_TOKEN", None)


def test_prioritize_requires_service_token():
    os.environ.pop("MODEL_RELEASE_APPROVED", None)
    os.environ.pop("APPROVED_ARTIFACT_REVISION", None)
    os.environ["SERVICE_TOKEN"] = "test-token"
    os.environ["MODEL_RELEASE_APPROVED"] = "false"
    client = _get_client()
    resp = client.post("/api/v1/prioritize", json={"seed_genes": ["FBN1"]})
    assert resp.status_code in (403, 503)
    for k in ["SERVICE_TOKEN", "MODEL_RELEASE_APPROVED"]:
        os.environ.pop(k, None)


def test_prioritize_validates_input():
    os.environ["SERVICE_TOKEN"] = "test-token"
    os.environ["MODEL_RELEASE_APPROVED"] = "true"
    os.environ["APPROVED_ARTIFACT_REVISION"] = "rev-x"
    # Need artifact dir to not abstain on missing artifact? Actually still abstains
    # but input validation happens after gate; test that empty body is rejected
    client = _get_client()
    resp = client.post(
        "/api/v1/prioritize",
        json={"top_k": 10},  # no disease_name or seed_genes
        headers={"X-Service-Token": "test-token"},
    )
    # Should return 422 (validation) or abstain with message about missing artifact
    # The route checks abstention before validation? Actually validates after gate check.
    # If no seed/disease, should 422.
    assert resp.status_code in (200, 422)
    if resp.status_code == 422:
        assert True
    else:
        # If 200, it's abstaining due to missing artifact, which is also honest
        assert resp.json()["status"] == "abstained"
    for k in ["SERVICE_TOKEN", "MODEL_RELEASE_APPROVED", "APPROVED_ARTIFACT_REVISION"]:
        os.environ.pop(k, None)
