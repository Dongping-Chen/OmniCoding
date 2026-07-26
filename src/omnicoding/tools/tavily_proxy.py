"""Search-only proxy that keeps upstream Tavily keys out of agent sandboxes."""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from omnicoding.tools import tavily_search


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)
    search_depth: str = Field(default="basic", pattern=r"^(basic|advanced)$")


def _proxy_token() -> str:
    token_file = os.environ.get(
        "OMNICODING_WEB_SEARCH_PROXY_TOKEN_FILE", ""
    ).strip()
    if not token_file:
        raise RuntimeError(
            "OMNICODING_WEB_SEARCH_PROXY_TOKEN_FILE is required"
        )
    token = Path(token_file).read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise RuntimeError("web-search proxy token must be at least 32 chars")
    return token


def _authorize(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    supplied = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied, _proxy_token()):
        raise HTTPException(status_code=403, detail="invalid bearer token")


app = FastAPI(title="OmniCoding scoped web-search proxy")


@app.get("/health")
def health(authorization: str | None = Header(default=None)) -> dict[str, str]:
    _authorize(authorization)
    # Force key validation without returning or logging secret material.
    tavily_search._load_keys()
    return {"status": "ok"}


@app.post("/search")
def search(
    request: SearchRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    _authorize(authorization)
    try:
        return tavily_search.search(
            request.query,
            max_results=request.max_results,
            search_depth=request.search_depth,
        )
    except tavily_search.TavilyError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
