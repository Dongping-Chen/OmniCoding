#!/usr/bin/env python3
"""Stdio MCP server exposing a single ``web_search`` tool backed by Tavily.

Hand-rolled minimal JSON-RPC 2.0 (newline-delimited, per MCP stdio
transport spec). One tool, no resources, no prompts. Used by both
opencode and claude code via per-workspace MCP config.

Logging goes to stderr — stdout is the protocol channel, mixing logs
in there desyncs the client.
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from omnicoding.tools import tavily_search

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "tavily-web-search", "version": "0.1.0"}

TOOL_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the public web via Tavily. Returns top result titles, URLs, "
        "and short snippets, plus an AI-generated direct answer when "
        "available. Use for current events, niche facts, citations, or any "
        "knowledge not already grounded in the staged inputs."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string.",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return.",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
            "search_depth": {
                "type": "string",
                "description": "Tavily search depth.",
                "enum": ["basic", "advanced"],
                "default": "basic",
            },
        },
        "required": ["query"],
    },
}


def _log(msg: str) -> None:
    sys.stderr.write(f"[tavily_mcp] {msg}\n")
    sys.stderr.flush()


def _send(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _result(req_id: Any, result: Any) -> None:
    _send({"jsonrpc": "2.0", "id": req_id, "result": result})


def _error(req_id: Any, code: int, message: str, data: Any = None) -> None:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    _send({"jsonrpc": "2.0", "id": req_id, "error": err})


def _do_search(args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return {
            "content": [{"type": "text", "text": "error: 'query' is required and must be a non-empty string."}],
            "isError": True,
        }
    try:
        payload = tavily_search.search(
            query.strip(),
            max_results=int(args.get("max_results", 5)),
            search_depth=str(args.get("search_depth", "basic")),
        )
    except tavily_search.TavilyExhausted as exc:
        return {
            "content": [{"type": "text", "text": f"web_search exhausted all keys: {exc}"}],
            "isError": True,
        }
    except tavily_search.TavilyError as exc:
        return {
            "content": [{"type": "text", "text": f"web_search error: {exc}"}],
            "isError": True,
        }
    text = tavily_search.format_markdown(payload)
    return {"content": [{"type": "text", "text": text}], "isError": False}


def handle(message: dict[str, Any]) -> None:
    method = message.get("method")
    req_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        _result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        })
        return

    if method == "notifications/initialized":
        # notification, no response
        return

    if method == "tools/list":
        _result(req_id, {"tools": [TOOL_SCHEMA]})
        return

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name != "web_search":
            _error(req_id, -32602, f"unknown tool: {name}")
            return
        try:
            _result(req_id, _do_search(args))
        except Exception as exc:  # surface unexpected failures as JSON-RPC errors
            _log(f"unhandled exception in tools/call: {exc}\n{traceback.format_exc()}")
            _error(req_id, -32000, f"internal error: {exc}")
        return

    if method == "ping":
        _result(req_id, {})
        return

    if req_id is not None:
        _error(req_id, -32601, f"method not found: {method}")


def main() -> int:
    _log("server starting")
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _log(f"non-JSON input dropped: {exc}: {line[:200]!r}")
            continue
        try:
            handle(msg)
        except Exception as exc:
            _log(f"handler crash: {exc}\n{traceback.format_exc()}")
    _log("server stdin closed; exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
