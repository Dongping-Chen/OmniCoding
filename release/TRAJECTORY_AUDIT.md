# Trajectory release audit

This is a location and completeness audit, not a trajectory-data release. The
original output trees were read in place and left unchanged. This Git repository
contains selection manifests only; raw conversations, tool outputs, benchmark
media, and processed images have not been copied.

## Release-candidate matrix

| Harness | Model | Benchmark | Audited items / retained result rows | Correct complete trajectories selected | Retained trajectory form |
|---|---|---|---:|---:|---|
| Kira | Qwen3.6-27B SFT | OmniGAIA full 360 | 360 / 941 | 156 | Full `messages.json` + step summary; 130 have image subcalls/images |
| Kira | Qwen3.6-27B SFT | LVOmniBench first 100 | 100 / 186 | 60 | Full messages; 57 have image subcalls/images |
| Kira | Qwen3.6-27B SFT | SocialOmni L1 first 100 | 100 / 100 | 64 | Full messages; 63 have image subcalls/images |
| Kira | Qwen3.6-27B SFT | SocialOmni L2 first 100 | 100 / 100 | 67 | Full messages; 67 have image subcalls/images |
| Kira | Qwen3.6-27B SFT | VideoZeroBench official 500 | 499 / 948 | 92 | Full messages; 91 have image subcalls/images; benchmark item 182 was not observed |
| Kira | Qwen3.6-27B base | OmniGAIA full 360 | 360 / 360 | 0 currently selectable | Full messages exist, but no run-specific external judge artifact was located, so all 360 remain unscored |
| Kira | Qwen3.6-27B base | LVOmniBench first 100 | 100 / 160 | 61 | Full messages; 52 have image subcalls/images |
| Kira | Qwen3.6-27B base | SocialOmni L1 first 100 | 100 / 100 | 46 | Full messages; 5 have image subcalls/images |
| Kira | Qwen3.6-27B base | SocialOmni L2 first 100 | 100 / 100 | 55 | Full messages; 2 have image subcalls/images |
| Kira | Qwen3.6-27B base | VideoZeroBench official 500 | 500 / 930 | 97 | Correct item 94 lacks a complete saved trajectory; 95 selected items have image subcalls/images |
| Codex CLI | gpt-5.5 xhigh | OmniGAIA full 360 | 360 | N/A | Only `codex_stdout_tail` was retained; mark partial |
| Codex CLI | gpt-5.5 xhigh | LVOmniBench first 100 | 100 | N/A | Full JSONL stdout; item 0094 is in the late-rerun sibling rather than the 99-row merged file |
| Codex CLI | gpt-5.5 xhigh | SocialOmni L1 first 100 | 100 | N/A | Full JSONL stdout |
| Codex CLI | gpt-5.5 xhigh | SocialOmni L2 first 100 | 100 | N/A | Full Q1 JSONL stdout; Q2 stdout exists for 36 items |
| Codex CLI | gpt-5.4 low/medium/xhigh + human skill | OmniGAIA | 360 / 360 / 332 | N/A | Only stdout tails; xhigh retained merge is explicitly partial |

Some result rows are retained aggregate/recovery copies, so the right-hand
count is not guaranteed to equal the number of independent model calls. The
Kira counts are best-of-retained-attempt inventory counts, **not Pass@1
benchmark scores**. Repeated wrong attempts remain unselected; a later judged-
correct attempt becomes that question's release candidate. The exact selected
item directories and artifact-presence flags are in
`trajectory_manifests/qwen_27b_kira.json`. Codex run-level locations and
completeness are in `trajectory_manifests/codex_runs.json`.

## Qwen retry selection policy

The inventory joins attempts by the benchmark's stable ID, not by
`source_index`, because one-item retry jobs reset `source_index` to zero. It
uses `id` for OmniGAIA and `question_id` for the other four adapters. A selected
attempt must:

1. be correct according to its in-row deterministic grader, or a run-scoped
   external judge record matched by both stable ID and normalized prediction;
2. have no recorded execution error;
3. retain a valid, non-empty `messages.json` object/list and a non-empty
   `final_text.txt` file.

When several attempts satisfy those rules, the selector prefers the attempt
with `trajectory.json`, image-subcall metadata, processed images, the larger
attempt number, and finally the stable relative result path. Filesystem mtimes
are deliberately ignored so copying or extracting an archive cannot change
the selected attempt. `completed=False` and
`exit_reason=no_tool_calls` are recorded but do not make a saved trajectory
incomplete: in older Kira runs they often mean the model omitted the
`task_complete` protocol call even though a complete, correct conversation and
final answer were saved.

External judge files are scoped to a named run. This prevents a Qwen3.6-27B
SFT judgement from being accidentally reused for a base-model prediction with
the same question ID and answer text.

## Other harness outputs found

The retained `wide_smoke_20260426_001054` tree contains Qwen3.6-27B harness
smokes, not full benchmark evaluations:

| Harness | Retained rows | Coverage | Decision |
|---|---:|---|---|
| Claude Code | 17 | OmniGAIA 5, LVO 1, Social L1 5, Social L2 5, VideoZeroBench 1 | Smoke/debug only |
| mini-swe-agent | 25 | Five items on each of the five adapters | Smoke/debug only |
| OpenCode | 11 | LVO 1, Social L1 5, Social L2 5; no retained rows for OmniGAIA/VideoZeroBench | Smoke/debug only |

The associated development log records harness bugs fixed after these smokes,
including answer extraction, context trimming, output limits, token accounting,
and resume handling. They are useful regression evidence but should not be
presented as benchmark results. Earlier `cml34_kira_full` and `dual_kira_*`
Qwen trees are likewise superseded by the two final 27B roots used in the
selection manifest.

The 9B runs are intentionally not audited or processed in this release, per
the project owner's instruction. No local game-benchmark trajectory tree was
located under the audited OmniCoding roots; those runs must be inventoried from
the separate environment where they were executed.

## What can be published next

Use separate payload groups even if they live in one Hugging Face dataset
repository:

```text
manifests/                    immutable item/run/checksum records
kira/<model>/<benchmark>/     messages, step summaries, final text
codex/<model>/<benchmark>/    full JSONL event streams only
```

Do not put partial OmniGAIA stdout tails in the same `complete` split. Keep
processed images out of the first text-trajectory upload: they are derived from
benchmark media and require per-benchmark redistribution review, particularly
for CC BY-NC-ND or unspecified upstream terms. Before any payload upload, scan
messages and tool outputs for credentials, private endpoints, absolute cluster
paths, personal data, and benchmark-media copies.

## Re-run the local inventory

The inventory command is intentionally read-only with respect to run roots. It
writes only the requested manifest:

```bash
omnicoding-data-trajectory-inventory \
  --run 27b-sft=/path/to/final_27b_sft_run \
  --run 27b-base=/path/to/final_27b_base_run \
  --judge 27b-sft:omnigaia:/path/to/initial_judge.json:id:predicted_answer:llm_equal \
  --judge 27b-sft:omnigaia:/path/to/redo_judge.json:id:pred:judge_correct \
  --output trajectory_inventory.json
```
