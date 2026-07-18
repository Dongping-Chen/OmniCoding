"""KIRA harness — native-tool-calling coding agent.

See ``omnicoding.agents.kira.loop`` for ``KiraAgent`` and
``omnicoding.harnesses.kira`` for the benchmark driver.
"""

from omnicoding.agents.kira.endpoint_pool import (
    Endpoint,
    EndpointPool,
    parse_endpoints,
)
from omnicoding.agents.kira.llm import (
    BlockTimeoutError,
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
    call_llm_for_image,
    call_llm_with_tools,
)
from omnicoding.agents.kira.loop import (
    AgentResult,
    KiraAgent,
    StepRecord,
    messages_preview,
    trajectory_to_dicts,
)
from omnicoding.agents.kira.parser import (
    Command,
    ImageReadRequest,
    ParsedToolCalls,
    parse_tool_calls,
)
from omnicoding.agents.kira.provider import (
    default_api_base,
    default_max_tool_reminders,
    detect_provider,
    detect_routed_provider,
    provider_kwargs,
    resolve_provider,
    resolve_routed_provider,
)
from omnicoding.agents.kira.recovery import recover_tool_calls
from omnicoding.agents.kira.shell import PersistentShell
from omnicoding.agents.kira.summarize import summarize_conversation
from omnicoding.agents.kira.tools import SYSTEM_PROMPT, TOOLS

__all__ = [
    "AgentResult",
    "BlockTimeoutError",
    "Command",
    "ContextLengthExceededError",
    "Endpoint",
    "EndpointPool",
    "ImageReadRequest",
    "KiraAgent",
    "LLMResponse",
    "OutputLengthExceededError",
    "ParsedToolCalls",
    "PersistentShell",
    "StepRecord",
    "SYSTEM_PROMPT",
    "TOOLS",
    "call_llm_for_image",
    "call_llm_with_tools",
    "default_api_base",
    "default_max_tool_reminders",
    "detect_provider",
    "detect_routed_provider",
    "messages_preview",
    "parse_endpoints",
    "parse_tool_calls",
    "provider_kwargs",
    "resolve_provider",
    "resolve_routed_provider",
    "recover_tool_calls",
    "summarize_conversation",
    "trajectory_to_dicts",
]
