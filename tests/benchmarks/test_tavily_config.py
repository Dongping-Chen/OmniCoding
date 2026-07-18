from __future__ import annotations

import pytest

from omnicoding.tools import tavily_search


def test_tavily_api_key_does_not_depend_on_package_location(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("TAVILY_KEYS_FILE", raising=False)

    assert tavily_search._load_keys() == ["test-key"]


def test_tavily_requires_explicit_configuration(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_KEYS_FILE", raising=False)

    with pytest.raises(tavily_search.TavilyError, match="set TAVILY_API_KEY"):
        tavily_search._load_keys()
