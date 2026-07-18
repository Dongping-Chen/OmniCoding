# Data and benchmark release layout

GitHub contains code, schemas, recipes, tiny synthetic fixtures, and checksums.
Large examples, media, trajectories, and model weights do not belong in Git.

Use separate Hugging Face dataset repositories under one organization:

- one repository for the released synthetic SFT corpus;
- one repository per redistributable benchmark;
- reconstruction manifests for benchmarks whose source media cannot be
  redistributed;
- optional result repositories for immutable public evaluation artifacts.

Every public recipe must identify a Hugging Face repository and immutable
revision. Do not make `main` the reproducibility contract.

## Synthetic SFT repository

Recommended files:

```text
README.md                 dataset card, collection model, filters, licenses
data/train-*.jsonl        ms-swift Agent-format trajectories
media/...                 only redistributable referenced assets
manifests/source.json     collection prompt/model/harness revisions
manifests/filter.json     filter policy and row counts
checksums.sha256
```

Document collection model terms, benchmark contamination checks, tool traces,
PII filtering, removed-row reasons, and the exact converter version.

## Benchmark repositories

Each benchmark repository should expose a normalized item schema while
retaining the upstream identifier and license metadata. Suggested fields are
`id`, `question`, `options`, `answer`, `media`, `split`, `source`, and
`license`. Keep private test labels in a gated or evaluator-only split when
public labels would invalidate the benchmark.

The adapter code stays in this GitHub repository. Dataset cards link back to
the adapter and pinned code release; code recipes link to pinned dataset
revisions.

See `release/DATA_LICENSE_REVIEW.md` for the current per-source blockers. In
particular, a single combined upload must not erase non-commercial,
no-derivatives, unspecified, or media-level restrictions.
