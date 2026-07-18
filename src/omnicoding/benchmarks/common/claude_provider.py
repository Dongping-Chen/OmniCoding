"""Provider helpers shared by Claude Code benchmark runners."""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import urlparse


LOCAL_PROFILE_NAMES = {"qwen_local", "local_qwen", "sglang_qwen", "local"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _lower_text(value: object) -> str:
    return str(value or "").strip().lower()


def _anthropic_base_url(env_overrides: Mapping[str, str] | None) -> str:
    if env_overrides and env_overrides.get("ANTHROPIC_BASE_URL"):
        return str(env_overrides["ANTHROPIC_BASE_URL"]).strip()
    return os.environ.get("ANTHROPIC_BASE_URL", "").strip()


def _base_url_uses_local_host(base_url: str) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url if "://" in base_url else f"http://{base_url}")
    host = (parsed.hostname or "").lower()
    configured_hosts = {
        item.strip().lower()
        for item in os.environ.get("OMNICODING_LOCAL_API_HOSTS", "").split(",")
        if item.strip()
    }
    return host in LOCAL_HOSTS | configured_hosts


def is_local_claude_provider(
    *,
    config_profile: str | None = None,
    provider_name: str | None = None,
    model_name: str | None = None,
    env_overrides: Mapping[str, str] | None = None,
) -> bool:
    profile = _lower_text(config_profile)
    provider = _lower_text(provider_name)
    model = _lower_text(model_name)
    if profile in LOCAL_PROFILE_NAMES or provider in LOCAL_PROFILE_NAMES:
        return True
    if _base_url_uses_local_host(_anthropic_base_url(env_overrides)):
        return True
    return model.startswith("qwen") and os.environ.get("ANTHROPIC_API_KEY") == "local"


def should_use_usage_limit_gate(
    mode: str,
    *,
    config_profile: str | None = None,
    provider_name: str | None = None,
    model_name: str | None = None,
    env_overrides: Mapping[str, str] | None = None,
) -> bool:
    normalized = _lower_text(mode) or "auto"
    if normalized == "on":
        return True
    if normalized == "off":
        return False
    if normalized != "auto":
        raise ValueError(f"Unknown usage-limit gate mode: {mode}")
    return not is_local_claude_provider(
        config_profile=config_profile,
        provider_name=provider_name,
        model_name=model_name,
        env_overrides=env_overrides,
    )
