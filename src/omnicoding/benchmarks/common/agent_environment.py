"""Shared coding-agent environment prompt blocks for local benchmark runners."""

from __future__ import annotations


def workspace_instructions(
    *,
    benchmark_name: str,
    staged_file_description: str,
    scope: str = "item",
    original_data_description: str = "original benchmark data directory",
) -> str:
    return "\n".join(
        [
            f"You are evaluating {benchmark_name} in an isolated local workspace.",
            "Work only inside the current workspace.",
            "Your current working directory is the workspace root for this run.",
            "Do not access, read, inspect, rely on, or search for files, directories, or other local content outside the current workspace.",
            "Never open absolute paths or parent-directory paths outside the workspace, and never use symlinks, environment discovery, shell expansion, or search commands to reach files beyond the workspace root.",
            "Treat any host file, cache, home-directory content, sibling directory, mounted dataset source, or path outside this workspace as strictly forbidden, even if it appears readable from the shell.",
            f"Use only the staged {staged_file_description}; do not assume hidden benchmark files exist.",
            f"The {original_data_description} is intentionally unavailable.",
            "Commands already start in the workspace root; do not `cd` to an absolute workspace path.",
            "Use relative paths exactly as listed below, such as `inputs/...` and `artifacts/...`, in shell commands.",
            "If a command fails because a path is missing, run `pwd && find . -maxdepth 2 -type f` and retry with a relative path.",
            "You may install dependencies without sudo when necessary, but keep them inside the current workspace.",
            "User-level install prefixes and caches are redirected into the workspace for this run.",
            f"Use at most one heavyweight local inference job at a time within this {scope}.",
        ]
    )


def native_vision_restriction() -> str:
    return "\n".join(
        [
            "Native vision restriction:",
            "- This run forbids direct use of the model's built-in image or video understanding ability.",
            "- Treat all images and videos as opaque files unless you inspect them through Python code or external tools.",
            "- If visual information is needed, use Python libraries or tools such as PIL, OpenCV, ffmpeg, ffprobe, OCR, or frame extraction first.",
            "- You may reason only over textual outputs, metadata, transcripts, OCR results, extracted frame descriptions, or other tool-generated text.",
            "- Do NOT directly look at, interpret, or answer from image/video inputs using the model's own native visual perception.",
        ]
    )


def network_instructions(*, allow_shell_network: bool, sandbox: str, forbidden_target: str = "benchmark answers") -> str:
    if allow_shell_network:
        return "\n".join(
            [
                "Network access may be available for shell commands in this run.",
                "You may use the network only when it materially helps process the current sample, including downloading tools or packages into the current workspace only.",
                f"Do not use the network to search for {forbidden_target}, leaked annotations, dataset-specific solutions, or existing evaluation outputs.",
            ]
        )
    if sandbox == "danger-full-access":
        return (
            "Networked shell commands may be available in this mode, but you still must not access files outside the current workspace."
        )
    return "If networked shell commands are blocked in this mode, answer using staged files and local tools only."


def gpu_instructions() -> str:
    return "\n".join(
        [
            "GPU access may be available for shell commands in this run.",
            "Before using GPU acceleration, check availability with `nvidia-smi` or a short `torch.cuda.is_available()` probe.",
            "Respect CUDA_VISIBLE_DEVICES/NVIDIA_VISIBLE_DEVICES and avoid unnecessary heavyweight downloads.",
            "When CUDA is available, prefer GPU for ASR/OCR/vision inference that materially improves the analysis.",
            "If no CUDA device is available, use CPU-friendly targeted clips instead of whole-media heavyweight inference.",
        ]
    )


def shared_python_env_instructions(*, scope: str = "item", env_path: str | None = None) -> str:
    lines = []
    if env_path:
        lines.extend(
            [
                f"The shared Python environment for this run is `{env_path}`.",
                "Commands inherit this environment through VIRTUAL_ENV and PATH; `python`, `pip`, and `whisper` should resolve from that environment.",
            ]
        )
    else:
        lines.append("A shared Python environment with preinstalled packages is available in PATH for this run.")

    lines.extend(
        [
            "Prefer reusing the shared environment over creating a fresh virtualenv or reinstalling large packages unless necessary.",
            "For speech transcription, prefer the `whisper` CLI on PATH over importing Python `whisper`.",
            f"Use at most one Whisper process per {scope}, wait for it to finish before continuing, and include `--threads 1`.",
            "When CUDA is available, prefer a stronger GPU ASR path: `whisper artifacts/audio.wav --model turbo --device cuda --fp16 True --threads 1 --output_format txt --output_dir artifacts/transcript`.",
            "Use `--model large-v3 --device cuda --fp16 True` for short or difficult clips when highest transcription accuracy is more important than speed.",
            "When CUDA is not available, use targeted clips with `--model base --device cpu --fp16 False --threads 1`; avoid whole-video CPU transcription unless necessary.",
            "Prefer extracting question-relevant audio clips with ffmpeg over whole-video transcription when the question names a time span.",
        ]
    )
    return "\n".join(lines)


def tool_workflow_instructions(*, scope: str = "item", max_commands: int = 8) -> str:
    # Note: `max_commands` is retained as a soft hint inside per-spec
    # callers but no longer surfaced to the model — the previous text
    # ("aim to answer after no more than N commands" + "stop using
    # tools and answer" + "do not write interim prose") demonstrably
    # capped Qwen3.6 thinking prematurely. Keep encouragement to think
    # and verify; let the harness's step_limit / retry-gate budget
    # bound runtime instead.
    _ = max_commands  # noqa: F841 — kept for backwards-compat with callers
    return "\n".join(
        [
            "You are operating in a terminal-capable coding-agent harness, not a direct-answer chat.",
            "Use shell, Python, ffmpeg/ffprobe, OpenCV/PIL, OCR, transcription, and other local tools to inspect staged media before deciding.",
            "It is expected to run commands, read generated text outputs, and iterate from the evidence you collect.",
            "Take the time you need to think through the problem, gather evidence, and verify your reasoning before committing to a final answer.",
            "Final-answer formatting applies only to the last response after exploration; it does not prohibit tool calls or intermediate reasoning.",
            "Use as many tool calls and intermediate reasoning steps as the task warrants — there is no hard cap.",
            "When the question is ambiguous or evidence is partial, prefer one more targeted tool call over guessing.",
        ]
    )
