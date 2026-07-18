# Open-source monorepo migration plan

## Goal

Build one public GitHub codebase that reproduces the complete OmniCoding
workflow: benchmark execution across multiple agent harnesses, verified
synthetic trajectory generation, ms-swift SFT, SGLang inference, evaluation,
and Relax-based RL. Large datasets and weights remain in versioned Hugging
Face repositories.

## Design decisions

1. The local experiment workspaces are immutable migration sources.
2. Files enter this repo only through the allowlist in `release/SOURCE_MAP.yaml`.
3. Benchmark-specific behavior implements the existing `BenchSpec` contract.
4. Harness, benchmark, model, and executor remain orthogonal configuration axes.
5. ms-swift and Relax are pinned upstream integrations, not copied wholesale.
6. Slurm and Modal launchers are canonical templates with no cluster identity,
   user paths, secrets, live endpoints, or job-specific recovery logic.
7. Dataset/media/model artifacts are referenced by repository and immutable
   revision; restricted assets use reconstruction manifests.

## Migration phases

### Phase 0: staging and provenance

Create the repository skeleton, ignore rules, source map, exclusions, recorded
source revisions, and release blockers. No runtime behavior changes.

### Phase 1: agent and benchmark core

Copy Kira, common benchmark runtime, five benchmark specs/prompts, supported
harness runners, and their unit tests. Rewrite only import paths needed for the
`omnicoding` package namespace. Verify collection and unit tests before moving
on.

### Phase 2: synthetic data and SFT

Copy the data preparation, trajectory generation, filtering, ms-swift
conversion, dataset upload, SFT, and LoRA merge code. Replace absolute paths
with CLI arguments or config fields. Add an eight-row fixture-based smoke path.

### Phase 3: inference and evaluation

Copy canonical SGLang serving and evaluation aggregation code. Define a run
manifest that pins model, dataset, benchmark, prompt, harness, tool, and source
revisions. Do not migrate retry/redo directories as canonical results.

### Phase 4: RL integration

Copy `relax-router` and coordinator code. Pin the audited Relax fork revision
and encode local core changes as reviewable patches. Before public deployment,
add authentication, request bounds, URL/model allowlists, media path
containment, and an explicit allowlist of environment variables passed to agent
jobs.

### Phase 5: release verification

Run unit tests, import checks, secret scanning, absolute-path scanning, license
inventory, dataset overlap analysis, benchmark smoke tests, SFT smoke, and one
RL rollout smoke from a fresh clone.

## Acceptance criteria

A release candidate is complete only when:

- a fresh clone installs without access to the original `/fs` workspace;
- all committed tests pass without private credentials;
- one command selects any supported benchmark/harness/model/executor tuple;
- an eight-row synthetic-to-SFT smoke is reproducible;
- a local-only RL coordinator smoke is authenticated and cannot expose secrets;
- every dataset/model/result references an immutable Hugging Face revision;
- train/eval overlap is reported rather than silently ignored;
- secret and large-file scans return no release blockers.

