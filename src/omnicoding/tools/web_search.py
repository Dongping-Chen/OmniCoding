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
import sys

from omnicoding.tools import tavily_search


def main() -> int:
    p = argparse.ArgumentParser(prog="web_search")
    p.add_argument("query", help="search query")
    p.add_argument("--max", "--max-results", dest="max_results", type=int, default=5)
    p.add_argument("--depth", choices=["basic", "advanced"], default="basic")
    p.add_argument("--json", action="store_true", help="emit raw Tavily JSON")
    args = p.parse_args()
    try:
        payload = tavily_search.search(
            args.query, max_results=args.max_results, search_depth=args.depth,
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
