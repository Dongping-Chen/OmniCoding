# Relax coding-agent RL integration

This directory documents the pinned Relax dependency and the monorepo's custom
coding-agent rollout implementation. The runtime modules live under
`omnicoding.rl`; Relax's standard SGLang rollout loads the per-sample generator
with:

```text
--custom-generate-function-path omnicoding.rl.rollout.generate
```

The coordinator is authenticated and defaults to localhost-only deployment.
It refuses unlisted inference origins/models, bounds request sizes and global
capacity, copies only task media into isolated workspaces, keeps ground-truth
answers and dataset-root paths out of Slurm payloads, grades returned
trajectories in the coordinator, and exports a small non-secret environment
allowlist to Slurm workers.

On clusters with Slurm Pyxis/Enroot, prefer
`ROLLOUT_EXECUTION_BACKEND=slurm_step`: Relax, SGLang, and the coordinator
share one GPU allocation, while every trajectory runs as its own bounded
container job step. The container receives only its private workspace,
gold-free request, result file, and a read-only shared harness environment.
This provides real filesystem/process isolation without submitting dozens of
child `sbatch` jobs. The model can install extra packages only into its
trajectory-private workspace venv. Build the versioned worker squashfs with
`infra/container/build_rl_worker_image.sh`.

The Kira SFT environment is reproduced separately from the Relax training
environment:

```bash
uv venv --python /usr/bin/python3 .venv-harness
uv pip install --python .venv-harness/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match \
  -r requirement-harness.txt
```

The launcher mounts `.venv-harness` read-only at the same absolute path and
adds it behind each workspace's writable `.venv`. This keeps the SFT tool
stack (including PyTorch, Whisper, OCR, and media libraries) available while
preventing one rollout's installs from changing another rollout.

Copy `coordinator.env.example` to an ignored local file, replace every path,
create the mode-0600 file named by `ROLLOUT_COORDINATOR_TOKEN_FILE`, and set
the inference origin and model allowlists explicitly. The token value stays
out of process arguments and Ray job metadata. Load that local file and start
the coordinator:

```bash
source /path/to/private-coordinator.env
uvicorn omnicoding.rl.coordinator.app:app --host 127.0.0.1 --port 8910
```

If a tunnel or reverse proxy is used, keep bearer authentication enabled and
use TLS. Never publish the coordinator directly without an additional network
policy and rate limit.

The full Relax actor stack is intentionally not duplicated in this monorepo.
Use the audited upstream revision in `UPSTREAM.md` and apply the small core
patches in `patches/`; this keeps third-party history and licensing visible.
For the locally validated Code-X-SFT-9B rollout and actor-update path,
including dependency pins, downloads, Slurm commands, and performance
interpretation, see `SLURM_QWEN35.md`. The standard entry point is
`recipes/rl_code_x_sft_9b_kira_gspo.sh`: it defaults to one eight-H100 actor
with TP=8/CP=1 and eight colocated TP=1 rollout engines. The four-H100 wrapper
remains available for diagnostics, but is not the training default.
The sanitized 27B GSPO entry point is `recipes/rl_27b_gspo.sh`, with a Slurm
wrapper at `infra/slurm/train_rl.sbatch`. It starts from
`shuaishuaicdp/Code-X-SFT-27B`, wires the public rollout/reward/filter hooks,
requires a writable `SAVE_DIR`, checkpoints every `SAVE_INTERVAL` updates
(50 by default), and retains the paper run's 27B tensor-parallel and
long-context defaults.
Cluster GPU topology, scheduler directives, network interface, storage, and
compatible Megatron/Relax environments remain deployment inputs.
