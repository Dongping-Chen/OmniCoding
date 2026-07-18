"""Async subprocess lifecycle helpers used by benchmark harness runners."""

from __future__ import annotations

import asyncio
import os
import signal
from typing import Any, Callable


async def terminate_process_group(process: asyncio.subprocess.Process, *, grace_seconds: float = 5.0) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        return
    except asyncio.TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    await process.wait()


async def read_stream_lines(
    stream: asyncio.StreamReader,
    sink: list[str],
    prefix: str,
    enabled: bool,
    summarizer: Callable[[str], str] | None = None,
) -> None:
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace")
        sink.append(text)
        if enabled:
            rendered = text.rstrip()
            if summarizer is not None:
                rendered = summarizer(rendered)
            print(f"{prefix}{rendered}", flush=True)


async def execute_prompt_command(
    *,
    command: list[str],
    prompt: str,
    timeout: int,
    env: dict[str, str],
    cwd: str | None = None,
    write_stdin: bool = True,
    live_prefix: str = "",
    live_output: bool = False,
    stdout_summarizer: Callable[[str], str] | None = None,
) -> tuple[str, str, int | None, bool]:
    stdout_text = ""
    stderr_text = ""
    process_return_code: int | None = None
    timed_out = False
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_task: asyncio.Task[Any] | None = None
    stderr_task: asyncio.Task[Any] | None = None

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE if write_stdin else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )
        try:
            assert process.stdout is not None
            assert process.stderr is not None
            if write_stdin:
                assert process.stdin is not None
                process.stdin.write(prompt.encode("utf-8"))
                await process.stdin.drain()
                process.stdin.close()

            stdout_task = asyncio.create_task(
                read_stream_lines(
                    process.stdout,
                    stdout_chunks,
                    live_prefix,
                    live_output,
                    summarizer=stdout_summarizer,
                )
            )
            stderr_task = asyncio.create_task(
                read_stream_lines(
                    process.stderr,
                    stderr_chunks,
                    f"{live_prefix}stderr: ",
                    live_output,
                )
            )
            wait_task = asyncio.create_task(process.wait())
            await asyncio.wait_for(wait_task, timeout=timeout)
        except asyncio.TimeoutError:
            timed_out = True
            await terminate_process_group(process)
        finally:
            pending_tasks = [task for task in (stdout_task, stderr_task) if task is not None]
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            stdout_text = "".join(stdout_chunks)
            stderr_text = "".join(stderr_chunks)
            process_return_code = process.returncode
    except Exception as exc:
        stderr_text = f"Failed to launch command: {exc}"

    return stdout_text, stderr_text, process_return_code, timed_out
