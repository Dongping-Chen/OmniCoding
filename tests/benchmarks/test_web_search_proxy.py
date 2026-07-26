from __future__ import annotations

from omnicoding.tools import web_search


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"results": [{"title": "ok", "url": "https://example.com"}]}


def test_web_search_uses_scoped_proxy_without_upstream_key(
    monkeypatch,
) -> None:
    captured = {}

    def fake_post(url, *, json, headers, timeout):
        captured.update(
            url=url, json=json, headers=headers, timeout=timeout
        )
        return _Response()

    monkeypatch.setenv(
        "OMNICODING_WEB_SEARCH_PROXY_URL", "http://127.0.0.1:19090/"
    )
    monkeypatch.setenv(
        "OMNICODING_WEB_SEARCH_PROXY_TOKEN", "scoped-token"
    )
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(web_search.requests, "post", fake_post)

    result = web_search._search(
        "test query", max_results=3, search_depth="basic"
    )

    assert result["results"][0]["title"] == "ok"
    assert captured["url"] == "http://127.0.0.1:19090/search"
    assert captured["headers"]["Authorization"] == "Bearer scoped-token"
