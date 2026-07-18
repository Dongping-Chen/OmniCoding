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
The sanitized 27B GSPO entry point is `recipes/rl_27b_gspo.sh`, with a Slurm
wrapper at `infra/slurm/train_rl.sbatch`. It starts from
`shuaishuaicdp/Code-X-SFT-27B`, wires the public rollout/reward/filter hooks,
requires a writable `SAVE_DIR`, checkpoints every `SAVE_INTERVAL` updates
(50 by default), and retains the paper run's 27B tensor-parallel and
long-context defaults.
Cluster GPU topology, scheduler directives, network interface, storage, and
compatible Megatron/Relax environments remain deployment inputs.
