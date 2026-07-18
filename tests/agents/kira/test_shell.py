"""Persistent-shell behaviour pinned: cwd persistence, env persistence,
30 KB cap, marker filtering, SIGINT-on-timeout resync.

Live tests — they spawn a real bash subprocess. They are fast (<5 s
total) so they are part of the default suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from omnicoding.agents.kira.shell import PersistentShell, _MARKER_PREFIX


@pytest.fixture
def tmpws(tmp_path: Path) -> Path:
    return tmp_path


def test_cwd_persists_across_calls(tmpws: Path):
    sub = tmpws / "deep" / "place"
    sub.mkdir(parents=True)
    with PersistentShell(cwd=tmpws) as sh:
        sh.run("cd deep/place", duration=1.0)
        out = sh.run("pwd", duration=1.0).strip()
    assert out.endswith("deep/place"), f"pwd output was {out!r}"


def test_env_var_persists(tmpws: Path):
    with PersistentShell(cwd=tmpws) as sh:
        sh.run("export KIRA_TEST_VAR=hello-world", duration=1.0)
        out = sh.run("echo $KIRA_TEST_VAR", duration=1.0).strip()
    assert out == "hello-world"


def test_marker_lines_filtered_from_output(tmpws: Path):
    with PersistentShell(cwd=tmpws) as sh:
        out = sh.run("echo visible-line", duration=1.0)
    assert _MARKER_PREFIX not in out
    assert "visible-line" in out


def test_output_cap_truncates_huge_outputs(tmpws: Path):
    with PersistentShell(cwd=tmpws, max_output_bytes=2_000) as sh:
        # 'yes' floods stdout; head it to 100 KB so we cap predictably.
        out = sh.run("yes A | head -c 100000", duration=5.0)
    assert "bytes elided" in out
    assert len(out.encode("utf-8")) <= 4_000  # ~2 KB head + ~2 KB tail + elision text


def test_timeout_sigints_and_resyncs(tmpws: Path):
    """A command that exceeds duration must be SIGINT'd; the next
    command in the same shell must still run."""
    with PersistentShell(cwd=tmpws) as sh:
        slow_out = sh.run("sleep 5", duration=0.5)
        next_out = sh.run("echo after-resync", duration=2.0).strip()
    assert "exceeded duration=0.5s" in slow_out
    assert "after-resync" in next_out


def test_runs_in_workspace_cwd(tmpws: Path):
    (tmpws / "marker-file.txt").write_text("ok")
    with PersistentShell(cwd=tmpws) as sh:
        out = sh.run("ls marker-file.txt", duration=1.0)
    assert "marker-file.txt" in out


def test_extra_env_passed_through(tmpws: Path):
    """The harness uses this to inject web_search on PATH."""
    bin_dir = tmpws / "fakebin"
    bin_dir.mkdir()
    fake = bin_dir / "kira_fake_tool"
    fake.write_text("#!/bin/sh\necho fake-tool-ran\n")
    fake.chmod(0o755)
    extra_env = {"PATH": f"{bin_dir}:{os.environ['PATH']}"}
    with PersistentShell(cwd=tmpws, env=extra_env) as sh:
        out = sh.run("kira_fake_tool", duration=1.0).strip()
    assert "fake-tool-ran" in out


# ---------- Round-17 regression: stdin-eating foreground -------------------


def test_stdin_eating_command_does_not_chop_marker(tmpws: Path):
    """``head -c N`` reads N bytes off stdin then exits. Pre-fix: those
    N bytes came off the queued marker echo line, producing
    ``bash: line N: <chopped>: command not found``. Post-fix: user
    command's children inherit fd 0 = /dev/null so they cannot eat the
    marker. Reproduces the item_0000 / item_0007 ffmpeg bug class."""
    with PersistentShell(cwd=tmpws) as sh:
        # head reads 5 bytes from stdin and discards them; without the
        # fix, those bytes would come from the next bash input line.
        out = sh.run("head -c 5 >/dev/null", duration=2.0)
        # If the marker leaked, the next call would see "command not found".
        next_out = sh.run("echo recovery-ok", duration=1.0).strip()
    assert "command not found" not in out, f"marker bytes were eaten: {out!r}"
    assert "recovery-ok" in next_out


def test_long_running_stdin_reader_with_timeout(tmpws: Path):
    """A simulation of the real ffmpeg case: a slow foreground command
    that ALSO would read stdin if it could. We pair a stdin-read with a
    sleep so duration fires while the child is alive. Post-fix: child
    reads /dev/null, sleep drives the timeout, marker line survives
    intact, next command runs cleanly."""
    with PersistentShell(cwd=tmpws) as sh:
        # ``head -c 5`` reads 5 bytes from stdin (gets /dev/null EOF
        # immediately), then sleep keeps the foreground alive past
        # duration so SIGINT fires in the same control path that broke
        # before. The marker still has to survive.
        slow = sh.run("head -c 5 >/dev/null && sleep 5", duration=0.5)
        next_out = sh.run("echo after-resync", duration=1.0).strip()
    assert "exceeded duration=0.5s" in slow
    assert "command not found" not in slow
    assert "after-resync" in next_out


def test_inline_python_heredoc_does_not_break_marker(tmpws: Path):
    """Item_0001 case: inline python heredoc with ``import torch.cuda``.
    Heredoc content must reach python intact (not be eaten by stdin
    consumption), and the marker must follow."""
    cmd = (
        "python3 - <<'EOF'\n"
        "import sys\n"
        "print('python-heredoc-ran')\n"
        "EOF"
    )
    with PersistentShell(cwd=tmpws) as sh:
        out = sh.run(cmd, duration=5.0)
        next_out = sh.run("echo done", duration=1.0).strip()
    assert "python-heredoc-ran" in out
    assert "command not found" not in out
    assert "done" in next_out


def test_stdin_redirect_does_not_break_user_explicit_stdin(tmpws: Path):
    """User keystrokes that explicitly feed stdin via ``<<<`` heredoc-string
    or pipeline must still work — our ``</dev/null`` only applies to
    children that DON'T set their own stdin."""
    with PersistentShell(cwd=tmpws) as sh:
        # Here-string: bash passes "hello\n" on stdin to cat.
        out1 = sh.run("cat <<< hello", duration=1.0).strip()
        # Pipeline: echo to grep via pipe.
        out2 = sh.run("echo apple banana | grep banana", duration=1.0).strip()
    assert "hello" in out1
    assert "banana" in out2


def test_temp_cmd_files_cleaned_up(tmpws: Path):
    """The per-call ``cmd_<seq>.sh`` files are sourced then become stale.
    Confirm the temp dir is wiped on close() so we don't leak under
    long-running benchmark runs."""
    sh = PersistentShell(cwd=tmpws)
    cmd_dir = sh._cmd_dir
    sh.run("echo a", duration=1.0)
    sh.run("echo b", duration=1.0)
    assert cmd_dir.exists()
    assert any(cmd_dir.iterdir())
    sh.close()
    assert not cmd_dir.exists(), f"temp dir leaked: {cmd_dir}"


def test_syntax_error_does_not_leak_temp_path(tmpws: Path):
    """Round-17.5 follow-up: when the model writes broken bash, the
    error message must NOT contain the harness's per-shell temp path
    (``/workspace/tmp/kira_shell_XXX/cmd_NN.sh``). A soak test found
    3 trajectories getting the path leaked back to them via stdout —
    which would teach the SFT model an internal path it should never
    reference. Switching from ``source`` to ``eval "$(<file)"`` makes
    bash's error blame ``eval`` instead.
    """
    with PersistentShell(cwd=tmpws) as sh:
        cmd_dir = sh._cmd_dir
        # Three syntax-broken inputs the r17.5 soak actually saw.
        for broken in [
            'echo "unclosed',         # unbalanced double quote
            "echo 'unclosed",         # unbalanced single quote
            "$'\\032' echo hi",       # literal control char as command
        ]:
            out = sh.run(broken, duration=2.0)
            assert str(cmd_dir) not in out, (
                f"temp path leaked into output for input {broken!r}: {out!r}"
            )
            # Sanity: shell still works after the syntax error.
            ok = sh.run("echo recovery-ok", duration=1.0).strip()
            assert "recovery-ok" in ok


def test_close_is_idempotent(tmpws: Path):
    sh = PersistentShell(cwd=tmpws)
    sh.run("echo hi", duration=1.0)
    sh.close()
    sh.close()  # should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
