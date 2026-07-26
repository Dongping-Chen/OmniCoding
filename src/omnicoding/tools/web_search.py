#!/usr/bin/env python3
"""Shell-callable web_search wrapper for the bash-only mini-swe agent.

Usage:
  web_search "query"
  web_search "query" --max 8
  web_search "query" --json     # raw Tavily JSON instead of markdown
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

from omnicoding.tools import tavily_search


def _search(
    query: str,
    *,
    max_results: int,
    search_depth: str,
) -> dict:
    """Use the scoped local proxy when configured, else direct Tavily."""

    proxy_url = os.environ.get("OMNICODING_WEB_SEARCH_PROXY_URL", "").strip()
    if not proxy_url:
        return tavily_search.search(
            query,
            max_results=max_results,
            search_depth=search_depth,
        )
    token = os.environ.get(
        "OMNICODING_WEB_SEARCH_PROXY_TOKEN", ""
    ).strip()
    if not token:
        raise tavily_search.TavilyError(
            "web-search proxy URL is configured but its scoped token is missing"
        )
    try:
        response = requests.post(
            proxy_url.rstrip("/") + "/search",
            json={
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
            },
            headers={"Authorization": "Bearer " + token},
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise tavily_search.TavilyError(
            f"local web-search proxy failed: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise tavily_search.TavilyError(
            "local web-search proxy returned a non-object response"
        )
    return payload


def main() -> int:
    p = argparse.ArgumentParser(prog="web_search")
    p.add_argument("query", help="search query")
    p.add_argument("--max", "--max-results", dest="max_results", type=int, default=5)
    p.add_argument("--depth", choices=["basic", "advanced"], default="basic")
    p.add_argument("--json", action="store_true", help="emit raw Tavily JSON")
    args = p.parse_args()
    try:
        payload = _search(
            args.query,
            max_results=args.max_results,
            search_depth=args.depth,
        )
    except tavily_search.TavilyError as exc:
        print(f"web_search error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(tavily_search.format_markdown(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
