"""KIRA tool definitions and system prompt.

Three tools, OpenAI-shape function specs (the chat template-friendly form
Qwen3.6 + sglang's ``--tool-call-parser qwen3_coder`` understands):

  - ``execute_commands(analysis, plan, keystrokes, duration)``
  - ``task_complete()``
  - ``image_read(file_path, image_read_instruction)``

Round-12: the schema is intentionally **flat** — no nested
``commands: [{keystrokes, duration}]`` array. Qwen3.6 + qwen3_coder
serializes nested-JSON parameter values inconsistently when the inner
strings contain shell quoting (``\\"``, ``\\'``), and any malformed
``commands`` JSON would silently drop the entire call. With one
command per ``execute_commands`` turn the parameter body is just a
plain shell string — qwen3_coder hands it back verbatim, no inner
JSON parsing required. Multi-step commands chain via shell ``&&`` /
``;`` / ``\\n`` newlines instead of array entries.
"""

from __future__ import annotations

from typing import Any

_EXECUTE_DESC = (
    "Run shell commands in the persistent terminal with your analysis "
    "and plan. State (cwd, env vars, shell functions) persists across "
    "calls."
)

_ANALYSIS_DESC = (
    "Analyze the current state based on the terminal output you have so "
    "far. What do you see? What has been accomplished? What still needs "
    "to be done?"
)

_PLAN_DESC = (
    "Describe your plan for the next batch of commands. What will you "
    "run and why? Be specific about what each command should accomplish."
)

_KEYSTROKES_DESC = (
    "Exact shell text to send to bash. ONE command per call — chain "
    "multi-step work via shell ``&&`` / ``;`` / newlines. Multi-line is "
    "fine; a heredoc works. Do not append a trailing newline — the "
    "harness adds one."
)

_DURATION_DESC = (
    "Per-command timeout in seconds (default 1.0). Fast commands "
    "(cd, ls, echo): 0.1. Compile/render commands: 5-30. Long-running "
    "(make, large script): up to 60. Cap is 60s — call execute_commands "
    "again with empty keystrokes if you need to wait longer."
)

_TASK_COMPLETE_DESC = (
    "Call this when the task is complete and you have committed your "
    "final answer (typically by emitting `<answer>...</answer>` either "
    "in your last assistant content or in a prior `execute_commands` "
    "echo). One call ends the task — there is no confirmation round."
)

_IMAGE_READ_DESC = (
    "Read and visually inspect an image file. Use ONLY for image files "
    "you need to look at (PNG, JPG, JPEG, GIF, WEBP). Do NOT use this "
    "for text files — use shell tools (cat, head, etc.) instead. The "
    "image will be loaded and shown to you in the next user message; "
    "your job is then to look at it and reason from what you see. "
    "Provide a short instruction describing what you want to learn — "
    "this gets echoed back alongside the image so you stay focused."
)

_FILE_PATH_DESC = (
    "Path to the image file. Relative paths resolve against the "
    "workspace cwd; absolute paths are honored as-is."
)

_INSTRUCTION_DESC = (
    "What you want to learn from the image. Be specific about what "
    "information to extract."
)


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_commands",
            "description": _EXECUTE_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string", "description": _ANALYSIS_DESC},
                    "plan": {"type": "string", "description": _PLAN_DESC},
                    "keystrokes": {"type": "string", "description": _KEYSTROKES_DESC},
                    "duration": {"type": "number", "description": _DURATION_DESC},
                },
                "required": ["analysis", "plan", "keystrokes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": _TASK_COMPLETE_DESC,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_read",
            "description": _IMAGE_READ_DESC,
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": _FILE_PATH_DESC},
                    "image_read_instruction": {"type": "string", "description": _INSTRUCTION_DESC},
                },
                "required": ["file_path", "image_read_instruction"],
            },
        },
    },
]


SYSTEM_PROMPT = (
    "You are an autonomous coding agent solving a benchmark task in a "
    "Linux workspace. You have three tools:\n"
    "  - execute_commands: run ONE shell command in a persistent bash. "
    "Always include analysis + plan + keystrokes. Chain multi-step "
    "work via shell ``&&`` / ``;`` / newlines / heredocs (NOT by "
    "calling execute_commands multiple times in one turn).\n"
    "  - image_read: visually inspect an image (PNG / JPG / JPEG / "
    "GIF / WEBP). The image is loaded and shown to you in the next "
    "user message — look at it directly and reason from what you see.\n"
    "  - task_complete: call when finished. ONE call ends the run.\n\n"
    "Tool-use rule: while exploring, every assistant turn must contain "
    "exactly one tool call. The ONE exception is the final-answer turn: "
    "see the benchmark's Final-answer protocol below — that protocol "
    "lets you emit a plain-text `<answer>...</answer>` message with no "
    "tool call, then call task_complete in the following turn.\n\n"
    "Constraints: terminal output is capped at 30 KB. The shell is "
    "non-interactive — do not run commands that need a TTY (vim, "
    "less, etc.). No human is available to answer prompts; pass "
    "flags or use heredocs."
)


