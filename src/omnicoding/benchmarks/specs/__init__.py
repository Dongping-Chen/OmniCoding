"""Registry of `BenchSpec`s exposed via the harness `--bench` flag."""

from __future__ import annotations

from omnicoding.benchmarks.common.spec import BenchSpec

from . import lvomnibench, omnigaia, socialomni_l1, socialomni_l2, videozerobench

REGISTRY: dict[str, BenchSpec] = {
    lvomnibench.SPEC.name: lvomnibench.SPEC,
    omnigaia.SPEC.name: omnigaia.SPEC,
    socialomni_l1.SPEC.name: socialomni_l1.SPEC,
    socialomni_l2.SPEC.name: socialomni_l2.SPEC,
    videozerobench.SPEC.name: videozerobench.SPEC,
}


def get(name: str) -> BenchSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown bench {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


def names() -> list[str]:
    return sorted(REGISTRY)
