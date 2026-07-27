# TerminalBench-O evaluator

This integration releases the evaluator for the 50-task TerminalBench-O
snapshot. Public task metadata is hosted at
[`shuaishuaicdp/TerminalBench-O`](https://huggingface.co/datasets/shuaishuaicdp/TerminalBench-O).

The release deliberately separates three things:

- Hugging Face contains public task metadata and the agent-facing prompts.
- This directory contains grader source and evaluation orchestration.
- Benchmark fixtures, hidden references, model outputs, and credentials remain
  outside Git and Hugging Face.

The evaluator does not load `.env` files. Secrets must be supplied explicitly
through the process environment. Copy `evaluator.env.example` to an ignored
file outside this repository, fill only the values needed by the selected
graders, and source it before evaluation.

## Layout

```text
integrations/terminalbench_o/
├── _lib/                    shared deterministic, ASR, LLM, and VLM judges
├── tasks/<task_id>/grader.py
├── evaluate.py              resolve and run one task grader
├── summarize.py             aggregate versioned grade.json files
├── export_metadata.py       build the metadata-only Hugging Face artifact
├── eval_array.sbatch        portable 50-task Slurm array template
├── evaluator.env.example   non-secret runtime environment template
├── requirements.txt         Python dependencies used across all graders
└── task_ids.txt             canonical task order
```

## Install

Python 3.11 is recommended. The graders also call system tools including
`ffmpeg`, `ffprobe`, ImageMagick, and SoX.

```bash
python -m venv .venv-terminalbench-o
source .venv-terminalbench-o/bin/activate
python -m pip install -r integrations/terminalbench_o/requirements.txt
```

Some audio graders use packages with platform-specific CUDA or compiler
requirements. Install a CPU/GPU-compatible PyTorch stack separately when ASR
or speaker-embedding dimensions are enabled.

## Evaluate one existing case

`BENCHMARK_ROOT` is the private extracted benchmark directory containing
`tasks/<task_id>/{fixtures,reference}`. A case directory must contain the
agent output under `workspace/output`.

```bash
source /path/to/private-terminalbench-o.env

python integrations/terminalbench_o/evaluate.py \
  --task T01a_soccer_highlights \
  --benchmark-root /path/to/private/terminalbench-o \
  --case-dir /path/to/newbench-codex-T01a_soccer_highlights-run \
  --grade-name terminalbench_o_v1
```

The command writes:

```text
<case-dir>/grades/<grade-name>/grade.json
<case-dir>/grades/<grade-name>/grade.log
```

Use `--dry-run` to verify argument resolution without running a grader.
Proxy variables are cleared by default; pass `--allow-proxy` only when the
configured judge endpoint requires the local proxy.

## Evaluate all 50 tasks on Slurm

```bash
export BENCHMARK_ROOT=/path/to/private/terminalbench-o
export RUNS_DIR=/path/to/existing/case-runs
export GRADER_PYTHON=/path/to/python
export GRADE_NAME=terminalbench_o_v1
sbatch integrations/terminalbench_o/eval_array.sbatch
```

Then summarize:

```bash
python integrations/terminalbench_o/summarize.py \
  --runs-dir "$RUNS_DIR" \
  --grade-name "$GRADE_NAME" \
  --out /path/to/terminalbench_o_summary.json
```

## Release validation

The sanitized integration was replayed on all 50 archived GPT-5.5 xHigh case
outputs on 2026-07-27:

- 50/50 grader commands resolved every required path argument;
- all 50 grader entry points imported with the released dependency list;
- 50/50 cases produced parseable grades with zero evaluator errors;
- all 50 pass/fail labels matched the archived 2026-05-07 evaluation;
- 35/50 continuous scores matched exactly. The mean absolute score delta was
  `0.014754`, consistent with nondeterministic ASR/LLM/VLM judge dimensions.

The validation mean was `0.711694` with 12 passes and 38 failures, versus the
archived mean of `0.7075` with the same 12/38 split. Case outputs, detailed
grades, judge credentials, and logs remain private.

## Export public metadata

The exporter copies only the YAML task definitions and produces a normalized
50-row JSONL. It records aggregate fixture/reference counts and hashes but
never copies their paths or contents. Legacy, shortened, missing, or stale
YAML `id` values are normalized to the canonical task-directory name and kept
for audit in the JSONL `source_declared_id` field.

```bash
python integrations/terminalbench_o/export_metadata.py \
  --source-root /path/to/private/terminalbench-o \
  --metadata-root /path/to/syntax-repaired/terminalbench-o \
  --output-dir /path/to/TerminalBench-O-staging
```

The exporter fails closed on credential-shaped values, private absolute paths,
symlinks in copied metadata, a nonempty output directory, a task-count
mismatch, or a missing grader.
`--metadata-root` is optional; use it when the immutable raw snapshot contains
the three known malformed YAML files. Fixture/reference inventory still comes
from `--source-root`, and neither tree is modified.

## Security and release boundary

- Never commit or upload `.env`, API keys, cookies, proxy credentials, raw
  benchmark media, hidden references, or generated case runs.
- `CLAW_BENCH_SKIP_VLM=1` and `CLAW_BENCH_SKIP_ASR=1` are diagnostic modes;
  they change scores and must not be used for official results.
- LLM/VLM judge calls are nondeterministic external dependencies. Preserve the
  judge model, endpoint, grader commit, and per-task `grade.json` when
  reporting a run.
- Dataset source terms vary. The metadata release does not grant permission to
  redistribute the underlying fixtures.
