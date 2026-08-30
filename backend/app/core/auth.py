"""Firebase auth dependency (fail-closed, reuse standard env var names)."""
from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import Header, HTTPException, Depends

# Lazy import firebase_admin so tests don't require it


def _get_firebase_app():
    try:
        import firebase_admin
        from firebase_admin import credentials
    except ImportError:
        return None

    if firebase_admin._apps:
        return firebase_admin._apps[0]

    cred_json = os.getenv("FIREBASE_ADMIN_JSON") or os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not cred_json:
        return None
    try:
        # cred_json may be a file path or raw JSON string
        if os.path.exists(cred_json):
            cred = credentials.Certificate(cred_json)
        else:
            info = json.loads(cred_json)
            cred = credentials.Certificate(info)
        app = firebase_admin.initialize_app(cred)
        return app
    except Exception:
        return None


async def require_firebase_user(
    authorization: Optional[str] = Header(default=None),
) -> dict:
    """
    Verify Firebase ID token from Authorization: Bearer <token>.
    If Firebase not configured, raises 503 (not 401) to indicate service misconfig.
    For local dev without Firebase, set ALLOW_ANONYMOUS=true to bypass.
    """
    if os.getenv("ALLOW_ANONYMOUS", "").lower() == "true":
        return {"uid": "anonymous", "email": "anonymous@example.com"}

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty token")

    app = _get_firebase_app()
    if app is None:
        raise HTTPException(status_code=503, detail="Firebase not configured on server")

    try:
        from firebase_admin import auth as fb_auth

        decoded = fb_auth.verify_id_token(token)
        return decoded
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


async def require_service_token(
    x_service_token: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
) -> None:
    """
    Internal service-token gate for /prioritize.
    Accepts either X-Service-Token header or Authorization: Bearer <service_token>.
    Env var: SERVICE_TOKEN or INTERNAL_SERVICE_TOKEN.
    """
    expected = os.getenv("SERVICE_TOKEN") or os.getenv("INTERNAL_SERVICE_TOKEN")
    if not expected:
        # If no token configured, deny all (fail-closed)
        raise HTTPException(status_code=503, detail="Service token not configured")

    provided = None
    if x_service_token:
        provided = x_service_token
    elif authorization and authorization.startswith("Bearer "):
        provided = authorization.split(" ", 1)[1].strip()

    if provided != expected:
        raise HTTPException(status_code=403, detail="Invalid service token")
