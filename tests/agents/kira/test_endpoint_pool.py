"""Tests for kira.endpoint_pool: sticky weighted routing + failover."""
from __future__ import annotations

import pytest

from omnicoding.agents.kira.endpoint_pool import (
    Endpoint,
    EndpointPool,
    EndpointSession,
    parse_endpoints,
)


def test_parse_single_endpoint_default_weight():
    pool = parse_endpoints("http://a:8080/v1")
    assert pool.endpoints == [Endpoint("http://a:8080/v1", 1)]
    assert pool.total_weight == 1
    assert pool.pick_for_index(0) == "http://a:8080/v1"
    assert pool.pick_for_index(99) == "http://a:8080/v1"


def test_parse_two_endpoints_equal_weight():
    pool = parse_endpoints("http://a:8080/v1=2,http://b:8080/v1=2")
    assert {ep.weight for ep in pool.endpoints} == {2}
    # Equal weights → strict alternation.
    seq = [pool.pick_for_index(i) for i in range(pool.total_weight)]
    assert seq.count("http://a:8080/v1") == 2
    assert seq.count("http://b:8080/v1") == 2
    # Sticky: same idx → same url.
    assert pool.pick_for_index(7) == pool.pick_for_index(7 + pool.total_weight)


def test_parse_unequal_weights_interleaved():
    """4:2 split should interleave (not 4-then-2)."""
    pool = parse_endpoints("http://a:8080/v1=4,http://b:8080/v1=2")
    seq = [pool.pick_for_index(i) for i in range(pool.total_weight)]
    # 4 A's + 2 B's
    assert seq.count("http://a:8080/v1") == 4
    assert seq.count("http://b:8080/v1") == 2
    # Interleaved, not [A,A,A,A,B,B].
    assert seq != ["http://a:8080/v1"] * 4 + ["http://b:8080/v1"] * 2
    # First slot is A (highest weight).
    assert seq[0] == "http://a:8080/v1"


def test_pick_for_index_is_deterministic():
    pool = parse_endpoints("a=3,b=1")
    seq1 = [pool.pick_for_index(i) for i in range(20)]
    seq2 = [pool.pick_for_index(i) for i in range(20)]
    assert seq1 == seq2


def test_invalid_weight_raises():
    with pytest.raises(ValueError):
        parse_endpoints("http://a:8080/v1=0")
    with pytest.raises(ValueError):
        parse_endpoints("http://a:8080/v1=-1")
    with pytest.raises(ValueError):
        parse_endpoints("http://a:8080/v1=abc")


def test_empty_spec_raises():
    with pytest.raises(ValueError):
        parse_endpoints("")
    with pytest.raises(ValueError):
        parse_endpoints("  , ,")


def test_describe_round_trip():
    spec = "http://x:1/v1=3,http://y:2/v1=2"
    pool = parse_endpoints(spec)
    assert pool.describe() == spec


def test_three_endpoints_total_weight():
    pool = parse_endpoints("a=2,b=3,c=1")
    assert pool.total_weight == 6
    seq = [pool.pick_for_index(i) for i in range(6)]
    assert seq.count("a") == 2
    assert seq.count("b") == 3
    assert seq.count("c") == 1


def test_strip_whitespace():
    pool = parse_endpoints("  http://a/v1  =  4 , http://b/v1 = 2 ")
    assert pool.endpoints == [
        Endpoint("http://a/v1", 4),
        Endpoint("http://b/v1", 2),
    ]


def test_pipe_delimiter_for_sbatch_export_safety():
    """SLURM `--export=A=1,B=2` splits on commas, breaking comma-delimited
    endpoint specs. The `|` alternative (and mixed) round-trips cleanly."""
    pool = parse_endpoints("http://a:8080/v1=4|http://b:8080/v1=2")
    assert pool.endpoints == [
        Endpoint("http://a:8080/v1", 4),
        Endpoint("http://b:8080/v1", 2),
    ]
    # Mixed delimiters still work.
    pool2 = parse_endpoints("a=1|b=1,c=1")
    assert [ep.url for ep in pool2.endpoints] == ["a", "b", "c"]


# ----- EndpointSession (failover) -----------------------------------------


def test_session_starts_on_sticky_url():
    pool = parse_endpoints("a=1,b=1")
    s0 = EndpointSession(pool, idx=0)
    s1 = EndpointSession(pool, idx=1)
    # Different idx → different starting URL (with [a, b] interleave).
    assert s0.current_url != s1.current_url
    # Stickiness: same idx → same URL.
    assert EndpointSession(pool, idx=0).current_url == s0.current_url
    assert s0.history() == []


def test_session_failover_rotates_to_other_url():
    pool = parse_endpoints("a=1,b=1")
    s = EndpointSession(pool, idx=0)
    start = s.current_url
    assert s.failover("BlockTimeoutError") is True
    assert s.current_url != start
    h = s.history()
    assert len(h) == 1
    assert h[0]["from"] == start
    assert h[0]["to"] == s.current_url
    assert h[0]["reason"] == "BlockTimeoutError"


def test_session_per_url_cap_blocks_repeat_failover():
    """With 2 URLs and per_url=1, the second failover (back to start) is
    refused since both URLs would exceed the cap."""
    pool = parse_endpoints("a=1,b=1")
    s = EndpointSession(pool, idx=0, max_per_url=1)
    assert s.failover("err1") is True   # a → b
    # b's try-count is 0; a's is 1. Next failover from b would either go
    # back to a (a.tries=1, equals cap → blocked) or stay (b is current).
    # Because the loop only considers OTHER urls, only candidate is `a`,
    # which is at cap → returns False.
    assert s.failover("err2") is False


def test_session_record_success_resets_current_url_count():
    pool = parse_endpoints("a=1,b=1")
    s = EndpointSession(pool, idx=0, max_per_url=2)
    assert s.failover("e1") is True   # a → b, a.tries=1
    s.record_success()                  # b just succeeded → b.tries=0 (no-op since already 0)
    assert s.failover("e2") is True   # b → a, b.tries=1
    s.record_success()                  # a.tries=0
    assert s.failover("e3") is True   # a → b, a.tries=1
    # By now a tries=1, b tries=1. Cap=2; both still eligible.


def test_session_max_failovers_hard_cap():
    pool = parse_endpoints("a=1,b=1,c=1")
    s = EndpointSession(pool, idx=0, max_failovers=2, max_per_url=10)
    assert s.failover("e1") is True
    assert s.failover("e2") is True
    # Hard cap hit, even though per-url cap not exhausted.
    assert s.failover("e3") is False


def test_session_three_endpoints_rotate_in_order():
    pool = parse_endpoints("a=1,b=1,c=1")
    s = EndpointSession(pool, idx=0)
    visited = [s.current_url]
    for _ in range(2):
        ok = s.failover("test")
        assert ok
        visited.append(s.current_url)
    # All 3 URLs visited.
    assert set(visited) == {"a", "b", "c"}


def test_session_history_records_all_rotations():
    pool = parse_endpoints("a=1,b=1")
    s = EndpointSession(pool, idx=0)
    s.failover("BlockTimeoutError", step_hint=5)
    s.failover("APIConnectionError", step_hint=12)
    h = s.history()
    assert len(h) == 2
    assert h[0]["reason"] == "BlockTimeoutError"
    assert h[0]["step"] == 5
    assert h[1]["reason"] == "APIConnectionError"
    assert h[1]["step"] == 12
    # Round-trip: sliced history items are JSON-serialisable.
    import json
    json.dumps(h)


def test_session_single_endpoint_cannot_failover():
    pool = parse_endpoints("only")
    s = EndpointSession(pool, idx=0)
    assert s.failover("err") is False
    assert s.current_url == "only"
    assert s.history() == []
