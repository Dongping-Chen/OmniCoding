"""Coordinator authentication and request-policy validation."""

from __future__ import annotations

import hmac
import os
from urllib.parse import urlsplit

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from omnicoding.rl.schemas import RolloutRequest
from omnicoding.rl.secrets import load_coordinator_token

_bearer = HTTPBearer(auto_error=False)


def coordinator_token() -> str:
    return load_coordinator_token()


def require_coordinator_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    expected = coordinator_token()
    supplied = credentials.credentials if credentials else ""
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid coordinator token")


def _csv_env(name: str) -> set[str]:
    return {value.strip() for value in os.environ.get(name, "").split(",") if value.strip()}


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("sglang_base_url must be HTTP(S)")
    default_port = 443 if parsed.scheme == "https" else 80
    try:
        port = parsed.port or default_port
    except ValueError as exc:
        raise ValueError("sglang_base_url has an invalid port") from exc
    return f"{parsed.scheme}://{parsed.hostname.lower()}:{port}"


def validate_policy_config() -> tuple[set[str], set[str]]:
    allowed_origins = _csv_env("ROLLOUT_ALLOWED_SGLANG_ORIGINS")
    allowed_models = _csv_env("ROLLOUT_ALLOWED_MODELS")
    if not allowed_origins or not allowed_models:
        raise RuntimeError(
            "ROLLOUT_ALLOWED_SGLANG_ORIGINS and ROLLOUT_ALLOWED_MODELS must be configured"
        )
    try:
        normalized_origins = {_origin(value) for value in allowed_origins}
    except ValueError as exc:
        raise RuntimeError(f"invalid ROLLOUT_ALLOWED_SGLANG_ORIGINS: {exc}") from exc
    return normalized_origins, allowed_models


def enforce_request_policy(request: RolloutRequest) -> None:
    try:
        allowed_origins, allowed_models = validate_policy_config()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        request_origin = _origin(request.sglang_base_url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if request_origin not in allowed_origins:
        raise HTTPException(status_code=403, detail="sglang_base_url is not allowed")
    if request.sglang_model_name not in allowed_models:
        raise HTTPException(status_code=403, detail="sglang_model_name is not allowed")
