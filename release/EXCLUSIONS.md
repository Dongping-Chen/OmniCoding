# Release exclusions

The migration must not copy any of the following from source workspaces:

- `.env`, API keys, OAuth `auth.json`, cookies, Cloudflare credentials/config;
- live domains, tunnel IDs, Modal tokens, account IDs, or private endpoint URLs;
- dataset JSONL/Parquet, benchmark media, decoded images, audio, or video;
- model weights, adapters, optimizer state, checkpoints, or framework caches;
- output/trajectory payloads, logs, W&B runs, Slurm logs, PID files, or
  scratch; sanitized selection manifests containing no raw model/tool content
  may be generated in staging after review;
- virtual environments, Node installs, compiled files, or `__pycache__`;
- historical retry/redo/revive/fill scripts unless rewritten as a canonical,
  parameterized recovery mechanism;
- `codex-router/accounts`, personal ChatGPT OAuth automation, and its logs;
- the broken `kira_scale_1000_1777404796` run or symlink-dependent result trees;
- third-party repository checkouts or benchmark data mirrors.
- Relax `.env`, `cloudflared.yml`, the bundled `cloudflared` binary,
  `rl_prompts.parquet`, tunnel state, coordinator logs, and job scratch.

Code containing hard-coded `/fs`, `/nfshomes`, cluster hostnames, Slurm account
names, or live Modal/Cloudflare URLs may be copied only into staging and must be
parameterized before it is eligible for release.
