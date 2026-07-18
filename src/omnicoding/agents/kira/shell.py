"""Persistent bash subprocess for KIRA.

State (cwd, env vars, shell functions) persists across ``run`` calls
because we keep one ``bash`` open and write commands sequentially to its
stdin. Each command is followed by a unique echo marker; we read stdout
until the marker arrives or ``duration`` elapses.

Trade-offs vs harbor's tmux-backed terminal:
  - simpler; no tmux dependency, no pseudo-terminal
  - no ANSI / interactive program support — fine for benchmark agents
  - on duration timeout we send SIGINT to bash's process group so the
    foreground command dies, then resync via a fresh marker

The 30 KB output cap is applied after marker filtering so the model
never sees marker noise.

Round-17: user commands are run via ``eval "$(<file)" </dev/null`` from
a per-call temp file rather than written directly to bash's stdin.
WHY: ffmpeg / whisper / python-with-no-`-i` default to reading stdin
(waiting for keyboard control); when the user command is on the same
stdin pipe as the marker echo line, the foreground program eats bytes
off the marker before bash gets to it — producing
``bash: line N: cho: command not found`` from a chopped ``echo``. The
``</dev/null`` redirect on ``eval`` gives every child of the user
command an empty stdin, so no marker bytes are ever consumed.

Round-17.5: switched from ``. 'file'`` to ``eval "$(<file)"`` because
``source`` leaks the temp file path into bash error messages on
syntax-broken user input (``/workspace/tmp/kira_shell_X/cmd_N.sh:
line 1: unexpected EOF...``). The model would then see and learn that
internal path. ``eval`` errors say ``bash: eval: line N`` instead, no
path leak. Verified by a 3-case reproducer (unbalanced quote, control
char, etc).
"""

from __future__ import annotations

import logging
import os
import select
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("kira.shell")

DEFAULT_MAX_OUTPUT_BYTES = 30_000
_MARKER_PREFIX = "__KIRA_MARK_"
_PROMPT_INIT = (
    # Suppress prompts and echo so bash output is clean.
    "export PS1=''; export PS2=''; export PS4=''; "
    "set +o history 2>/dev/null || true; "
    "stty -echo 2>/dev/null || true\n"
)


class ShellError(RuntimeError):
    pass


class PersistentShell:
    """One persistent ``bash --noprofile --norc`` per KIRA run."""

    def __init__(
        self,
        cwd: Path,
        env: Optional[dict[str, str]] = None,
        *,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        self.cwd = Path(cwd).resolve()
        self.cwd.mkdir(parents=True, exist_ok=True)
        # Stash for transparent restart on dead-bash recovery (round-12
        # Bug 3). cwd persists; shell-local state (functions, vars) is
        # lost — the model is told via the warning string in ``run``.
        self._env_overlay = dict(env) if env else None
        self._max_output_bytes = max_output_bytes
        self._closed = False
        self._restart_count = 0
        self._proc: subprocess.Popen | None = None
        self._seq = 0
        # Per-shell temp dir holds the per-call ``cmd_<seq>.sh`` files
        # we source. Lives outside the workspace so the model never sees
        # them and ``ls`` etc. stay clean.
        self._cmd_dir = Path(tempfile.mkdtemp(prefix="kira_shell_"))
        self._spawn_bash()

    def _spawn_bash(self) -> None:
        full_env = os.environ.copy()
        if self._env_overlay:
            full_env.update(self._env_overlay)
        self._proc = subprocess.Popen(
            ["bash", "--noprofile", "--norc"],
            cwd=str(self.cwd),
            env=full_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
        )
        self._send_raw(_PROMPT_INIT)
        # Discard whatever the init line produced. Marker sync proves the
        # shell is alive and ready.
        self._sync_init()

    def _bash_alive(self) -> bool:
        """True if the bash subprocess is still running. Use before any
        write so we don't get a BrokenPipeError that aborts the whole
        agent run (round-12 Bug 3)."""
        return self._proc is not None and self._proc.poll() is None

    def _restart_bash(self, reason: str) -> None:
        """Replace the dead bash with a fresh one. cwd preserved via
        the env overlay; shell-local state is gone (the model is told
        via the warning string in ``run``)."""
        self._restart_count += 1
        LOGGER.warning(
            "shell restarting bash (reason=%s, restart #%d)",
            reason, self._restart_count,
        )
        # Best-effort cleanup of the previous proc.
        if self._proc is not None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        self._spawn_bash()

    def _send_raw(self, text: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise ShellError("shell stdin is closed")
        try:
            self._proc.stdin.write(text.encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ShellError(f"shell stdin write failed: {exc}") from exc

    def _sync_init(self) -> None:
        self._seq += 1
        marker = f"{_MARKER_PREFIX}init_{self._seq}__"
        self._send_raw(f"echo '{marker}'\n")
        self._read_until_marker(marker, timeout_s=5.0)

    def run(self, keystrokes: str, duration: float) -> str:
        """Run ``keystrokes`` and return the (truncated) output up to
        the marker or ``duration`` seconds, whichever comes first.

        Round-12 Bug 3: if bash died between commands (long-running
        web_search etc. SIGTERM'd by ``_interrupt_and_resync`` and the
        kill propagated back up, or the model itself ran ``exit``),
        restart bash transparently and continue. The replacement loses
        shell-local state (functions, local vars) but cwd is preserved.
        We tell the model via the prepended warning so it can re-export
        anything important.
        """
        if self._closed:
            raise ShellError("shell already closed")
        restart_warning = ""
        if not self._bash_alive():
            self._restart_bash(reason="dead_before_run")
            restart_warning = (
                "[!] shell process died and was restarted. cwd is "
                "preserved; shell functions / local variables / "
                "background jobs from prior commands are lost. "
                "Re-export anything important.\n"
            )
        self._seq += 1
        marker = f"{_MARKER_PREFIX}{self._seq}__"
        body = keystrokes if keystrokes.endswith("\n") else keystrokes + "\n"
        cmd_file = self._cmd_dir / f"cmd_{self._seq}.sh"
        cmd_file.write_text(body, encoding="utf-8")
        # Read the file contents into ``$( <file )`` and eval them with
        # ``</dev/null`` redirect. Eval errors surface as
        # ``bash: eval: line N`` rather than ``<full file path>: line N``
        # — keeping the harness's internal temp dir out of the model's
        # observation stream. Children of the user command still inherit
        # fd 0 = /dev/null so they cannot consume marker bytes.
        payload = (
            f'eval "$(<"{cmd_file}")" </dev/null\n'
            f"echo '{marker}'\n"
        )
        start = time.monotonic()
        try:
            self._send_raw(payload)
        except ShellError:
            # Race: bash died between the alive check and the write.
            # Restart and retry once. Second failure is real.
            self._restart_bash(reason="dead_during_write")
            restart_warning = (
                "[!] shell process died mid-write and was restarted. "
                "cwd preserved; shell-local state lost.\n"
            )
            self._send_raw(payload)
        output = self._read_until_marker(marker, timeout_s=max(duration, 0.5))
        elapsed = time.monotonic() - start
        LOGGER.debug("shell.run seq=%d elapsed=%.2fs marker_seen=%s out=%d", self._seq, elapsed, output["marker_seen"], len(output["text"]))
        text = self._filter_markers(output["text"])
        if not output["marker_seen"]:
            if not self._bash_alive():
                # The command (or its child) killed bash. Surface that to
                # the model and restart so the next run works.
                text += (
                    "\n[!] command killed the shell process; restarting. "
                    "cwd preserved; shell-local state lost.\n"
                )
                self._restart_bash(reason="dead_after_run")
            else:
                text += (
                    f"\n[!] command exceeded duration={duration}s; "
                    "harness sent SIGINT and resynced. Subsequent commands "
                    "still run in this shell.\n"
                )
                self._interrupt_and_resync()
        return self._cap_output(restart_warning + text)

    def _read_until_marker(self, marker: str, timeout_s: float) -> dict:
        if self._proc.stdout is None:
            raise ShellError("shell stdout is closed")
        fd = self._proc.stdout.fileno()
        chunks: list[bytes] = []
        deadline = time.monotonic() + timeout_s
        marker_bytes = marker.encode("utf-8")
        marker_seen = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            ready, _, _ = select.select([fd], [], [], min(remaining, 0.1))
            if not ready:
                continue
            try:
                buf = os.read(fd, 4096)
            except OSError as exc:
                LOGGER.warning("shell.read errored: %s", exc)
                break
            if not buf:
                LOGGER.warning("shell stdout EOF")
                break
            chunks.append(buf)
            joined = b"".join(chunks)
            if marker_bytes in joined:
                marker_seen = True
                break
        text = b"".join(chunks).decode("utf-8", errors="replace")
        return {"text": text, "marker_seen": marker_seen}

    def _interrupt_and_resync(self) -> None:
        """Kill bash's direct children (the running foreground command)
        without killing bash itself. Non-interactive bash's default
        SIGINT handler exits, so we cannot signal bash directly. After
        the child dies, bash continues with the next line of stdin,
        which is the marker echo, and we resync via the marker."""
        killed = self._kill_direct_children()
        if not killed:
            LOGGER.debug("shell._interrupt found no children to kill")
        # The marker echo we already sent should arrive once the child dies
        # and bash advances. Give it a brief extra deadline.
        marker = f"{_MARKER_PREFIX}{self._seq}__"
        self._read_until_marker(marker, timeout_s=2.0)

    def _kill_direct_children(self) -> int:
        """Send SIGTERM to all direct children of the bash process. The
        bash process keeps running so we can keep using the shell."""
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(self._proc.pid)],
                capture_output=True, text=True, timeout=2,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            LOGGER.warning("shell pgrep -P failed: %s", exc)
            return 0
        killed = 0
        for line in result.stdout.split():
            try:
                pid = int(line)
            except ValueError:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
            except ProcessLookupError:
                pass
        return killed

    @staticmethod
    def _filter_markers(text: str) -> str:
        return "\n".join(
            line for line in text.split("\n") if _MARKER_PREFIX not in line
        )

    def _cap_output(self, text: str) -> str:
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= self._max_output_bytes:
            return text
        head_n = self._max_output_bytes // 2
        tail_n = self._max_output_bytes - head_n
        head = encoded[:head_n].decode("utf-8", errors="replace")
        tail = encoded[-tail_n:].decode("utf-8", errors="replace")
        elided = len(encoded) - head_n - tail_n
        return (
            f"{head}\n"
            f"... [{elided} bytes elided; output capped at "
            f"{self._max_output_bytes} bytes] ...\n{tail}"
        )

    def close(self, timeout_s: float = 2.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        try:
            self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self._proc.wait(timeout=1.0)
        shutil.rmtree(self._cmd_dir, ignore_errors=True)

    def __enter__(self) -> "PersistentShell":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
