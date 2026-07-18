# Synthetic trajectory to SFT workflow

1. Run a supported benchmark through Kira and retain per-item
   `messages.json`, `results.json`, and `run_meta.json` artifacts.
2. Filter correct trajectories and convert them to ms-swift Agent JSONL.
3. Audit the JSONL and media paths, then publish a versioned dataset revision.
4. Train from that immutable revision and record the resulting run manifest.

Example conversion:

```bash
omnicoding-data-filter \
  --batch_dir /path/to/run/out \
  --items_file /path/to/items.json \
  --out_dir /path/to/release-stage \
  --multimodal
```

For Slurm collection, submit `omnicoding-data-dispatch` from a login/submit
node and pass `infra/slurm/harness_item.sbatch` explicitly. Cluster account,
partition, QoS, and GPU requests are intentionally absent from the committed
template; pass them with your site's `sbatch` options or maintain an ignored
local cluster profile.

Before training, verify prompt parity:

```bash
omnicoding-verify-prompt-parity --trajectories /path/to/run/out
```

See `integrations/ms_swift/README.md` for training and LoRA merge commands.
