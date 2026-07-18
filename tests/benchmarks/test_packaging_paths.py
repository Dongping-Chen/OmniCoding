from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from omnicoding.benchmarks.common import claude_runner
from omnicoding.paths import runtime_root


def test_runtime_root_defaults_to_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OMNICODING_RUNTIME_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert runtime_root() == tmp_path.resolve()


def test_runtime_root_accepts_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured = tmp_path / "runtime"
    monkeypatch.setenv("OMNICODING_RUNTIME_ROOT", str(configured))
    assert runtime_root() == configured.resolve()


def test_claude_mcp_uses_packaged_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(claude_runner, "TAVILY_MCP_BIN", "")
    config_path = claude_runner._write_mcp_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    server = config["mcpServers"]["tavily"]
    assert server["command"] == sys.executable
    assert server["args"] == ["-m", "omnicoding.tools.tavily_mcp"]
