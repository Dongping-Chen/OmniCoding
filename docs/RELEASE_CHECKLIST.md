# Release verification

This checklist records the checks applied to the initial public OmniCoding
code release. It separates tests that run on a CPU login node from workflows
that require the authors' Slurm GPU environment.

## CPU release gate

Run from a clean checkout with Python 3.10 or 3.11:

```bash
python -m pip install ".[dev,benchmark,rl,rl-data]"
find infra integrations recipes -type f \
  \( -name '*.sh' -o -name '*.sbatch' \) \
  -print0 | xargs -0 -n1 bash -n
python -m compileall -q src tests
pytest
python -m build
```

The 2026-07-18 release candidate passed 365 tests on both Python 3.10 and
Python 3.11, with 14 optional tokenizer/processor integration tests skipped by
default. Those 14 tests were also run separately against the released
Qwen3.6-27B tokenizer/processor; the complete model-backed integration
selection passed 22 tests. The built wheel was installed into a new virtual
environment, where package imports, console entry points, and a harness dry
run were checked.

The release gate also checks that:

- committed JSON files parse;
- committed Python files compile;
- shell and Slurm scripts pass `bash -n`;
- the Relax patches apply in order to the pinned fork baseline;
- no credential values, private workspace paths, symlinks, model weights, or
  generated training outputs are committed;
- the wheel contains the project license and bundled Apache-2.0 license copy.

GitHub Actions repeats the install, shell-syntax, and pytest checks on Python
3.10 and 3.11.

## GPU and cluster validation

The SFT, SGLang serving, and Relax RL entry points are designed for the Slurm
environment used for the paper. The checked-in scripts have been normalized
into templates and syntax-tested, but an end-to-end GPU training or RL run is
not part of the public CPU CI gate. Before using a new cluster, provide its
account, partition, QoS, storage, container, and networking configuration
through local environment variables or an ignored profile, then run a small
smoke job before scaling.

Large datasets, benchmark media, and model checkpoints are released outside
Git and must be pinned to immutable revisions in reproducible experiment
manifests.
