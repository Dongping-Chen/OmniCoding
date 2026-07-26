# Qwen3.5-9B Relax/GSPO on H100 Slurm nodes

This runbook records the source environment validated on a single Slurm node
with four and eight H100 80 GB GPUs. It uses the repository-local `.venv`; model, data,
outputs, and the environment itself are ignored by Git.

## Validated local layout

```text
.venv/                         Python 3.12 source environment
.venv/src/Relax/               patched Relax checkout
.venv/src/Megatron-LM/         checkout with Relax's Megatron patch
models/Qwen3.5-9B/             official base model (about 18 GB)
data/dapo-math-17k/            public rollout data
data/omnicoding/               public RL prompts and extracted media
runtime/*.sqsh                 generated Pyxis worker images
```

The important tested pins are:

```text
Relax              6932be2fad9488f37cdbadfac1d14dbce98fe2f1
Megatron-LM        3714d81d418c9f1bca4594fc35f9e8289f652862
Python             3.12.13
torch              2.9.1+cu128
transformers       5.3.0
sglang             0.5.10.post1
flashinfer         0.6.7.post3
flash-attn         2.7.4.post1
flash-linear-attn  0.4.1
transformer-engine 2.10.0
ray                2.56.1
fastapi            0.133.0
```

Apply every numbered patch under `patches/` to the pinned Relax checkout. Also
apply `.venv/src/Relax/docker/patch/latest/megatron.patch` to the pinned
Megatron-LM checkout. SGLang 0.5.9 is not suitable for this model: a hybrid
GDN/Mamba cache bug can produce an illegal-memory-access failure on a later
request even if the first rollout succeeds.

FastAPI must remain at 0.133.0 with Ray 2.56 in this environment. FastAPI
0.139.2 adds an unpicklable lock to the application object and breaks
`@serve.ingress` deployment serialization.

## Public assets

With `.venv` activated, the public inputs can be restored without a token:

```bash
hf download Qwen/Qwen3.5-9B \
  --local-dir models/Qwen3.5-9B

hf download zhuzilin/dapo-math-17k \
  --repo-type dataset \
  --local-dir data/dapo-math-17k

hf download shuaishuaicdp/OmniCoding \
  --repo-type dataset \
  --revision ecc1fa1b8297aca618a931ad322de4d4cb75fd65 \
  --local-dir data/omnicoding \
  --include README.md AGENT.md processed/rl_train.jsonl 'media/*.tar.gz'

for archive in data/omnicoding/media/*.tar.gz; do
  tar -xzf "${archive}" -C data/omnicoding/media
done

python -m omnicoding.rl.build_prompt_set \
  --input data/omnicoding/processed/rl_train.jsonl \
  --output data/omnicoding/processed/rl_prompts.parquet
```

The tested DAPO file has 17,398 JSONL rows. The fixed OmniCoding revision has
1,993 unique RL records and 3,954 media references; all references resolved
after extraction. The download is about 182 GB compressed and about 340 GB
with both archives and extracted media retained. A private SFT checkpoint should be
downloaded only after authenticating the local Hugging Face client; do not put
the token in a Slurm script, Ray runtime environment, or command line.

## Slurm launch

Request a single four-GPU node, then run the stock rollout isolation test:

```bash
salloc --partition=h100 --nodes=1 --ntasks=1 --cpus-per-task=48 \
  --gres=gpu:h100:4 --mem=950G --time=1-00:00:00

srun --nodes=1 --ntasks=1 --cpus-per-task=48 --gpus=4 \
  bash recipes/rl_qwen35_9b_4gpu_rollout_smoke.sh
```

Run one colocated GSPO update after the rollout-only test passes:

```bash
srun --nodes=1 --ntasks=1 --cpus-per-task=48 --gpus=4 \
  bash recipes/rl_qwen35_9b_4gpu_gspo_smoke.sh
```

Both scripts discover the compute interface and node IP, use a deliberately
short Ray temporary path to stay under Linux's 107-byte Unix-socket limit, and
propagate the selected interface into Ray workers. They also increase the Ray
Serve proxy health-check timeout from 10 to 120 seconds because source imports
from shared storage can otherwise trigger a false unhealthy restart. Ray is
explicitly limited to the CPUs assigned by Slurm; it otherwise detects every
CPU on the physical node and can create an import storm outside the allocation.
Do not
replace them with Relax's `scripts/entrypoint/local.sh` in a shared allocation;
that helper uses broad process cleanup.

The same rollout-only recipe supports eight GPUs:

```bash
NUM_GPUS=8 ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=2 \
  bash recipes/rl_qwen35_9b_4gpu_rollout_smoke.sh
```

For the real coding-agent path, first build the worker image and matching Kira
harness environment, then run all trajectories as isolated Pyxis job steps
inside the existing allocation:

```bash
uv venv --python /usr/bin/python3 .venv-harness
uv pip install --python .venv-harness/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match \
  -r requirement-harness.txt

ROLLOUT_CONTAINER_IMAGE="$PWD/runtime/omnicoding-rl-worker-20260724-v5.sqsh" \
  infra/container/build_rl_worker_image.sh

NUM_GPUS=8 ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=4 \
ROLLOUT_STEP_CONCURRENCY=16 \
ROLLOUT_CONTAINER_IMAGE="$PWD/runtime/omnicoding-rl-worker-20260724-v5.sqsh" \
ROLLOUT_HARNESS_VENV="$PWD/.venv-harness" \
  bash recipes/rl_qwen35_9b_4gpu_agent_rollout_smoke.sh
```

The standard Code-X-SFT-9B training entry now defaults to eight H100s:

```bash
# Submit a new one-day allocation.
sbatch recipes/rl_code_x_sft_9b_8gpu_kira_gspo_smoke.sbatch

# Or, from inside an existing eight-H100 allocation:
bash recipes/rl_code_x_sft_9b_kira_gspo.sh
```

Its actor topology is TP=8/CP=1/DP=1 because the installed Qwen3.5
GatedDeltaNet implementation does not support context parallelism. Rollout
uses eight independent TP=1 SGLang engines. Each update starts with two RL
prompts and four Kira trajectories per prompt, so eight trajectories are in
flight together. Kira uses the released SFT harness format, `max_turns=50`,
and a 48k-token final-trajectory cap. Groups containing a failed or overlong
sample are discarded and dynamically resampled before reaching Megatron.

SGLang calculates log probabilities against the final canonical Kira
trajectory, and Relax passes them into actor training with
`--use-rollout-logprobs`. KL and the reference model are disabled, so there is
no reference forward or actor-side old-logprob forward. Dynamic batching packs
short trajectories up to the 48k per-GPU budget; an individual sequence still
cannot exceed that budget.

When sandbox execution is hosted on another server, point the same training
entry at that coordinator instead of starting local Slurm/Pyxis workers:

```bash
ROLLOUT_COORDINATOR_PUBLIC_URL=https://sandbox-coordinator.example \
ROLLOUT_COORDINATOR_TOKEN_FILE=/absolute/path/to/mode-0600-token \
ROLLOUT_SGLANG_PUBLIC_URL=http://TRAIN_NODE_ROUTED_IP:ROUTER_PORT \
  bash recipes/rl_code_x_sft_9b_kira_gspo.sh
```

In this mode the training node reserves no CPUs or GPUs for sandbox workers
and starts only Ray, the eight SGLang rollout engines, and the TP=8 actor. The
remote coordinator must contain the pinned OmniCoding RL split and Kira
harness, allow the advertised SGLang origin/model, and be able to route back
to the training node. Omitting the coordinator URL keeps the fully local smoke
mode for regression testing.

This is one Slurm allocation, not one `sbatch` job per trajectory. Each
trajectory gets a separate read-only container root and private writable
workspace/tmp mounts. The shared `.venv-harness` is mounted read-only. A
worker cannot see the user's home, coordinator token, dataset root, or sibling
workspaces; it sees only the explicitly assigned sandbox GPU.

Do not start two independent Relax/Ray roots on spare GPUs of the same
physical node. Relax's Ray Serve endpoint still uses node port 8000, and Ray's
local process state is not isolated by the Slurm job ID. The launcher rejects
an occupied port 8000. Put all rollout fan-out behind the one rollout service
and coordinator instead.

The SGLang rollout topology defaults to one TP=1 engine per allocation GPU.
The Kira trajectories remain separate containers, while their media tools may
share one explicitly selected H100: eight trajectories in the four-GPU recipe
or 16 in the eight-GPU recipe. Extra trajectories remain in the coordinator
queue. Filesystems and processes are isolated; GPU memory is a deliberately
oversubscribed shared resource, not a hard per-container partition.

Useful overrides include:

```bash
MODEL_PATH=/absolute/private/sft-checkpoint
PROMPT_DATA=/absolute/data.jsonl
ROLLOUT_MAX_RESPONSE_LEN=8192
SGLANG_SERVER_CONCURRENCY=32
SGLANG_MAX_RUNNING_REQUESTS=32
SGLANG_DISABLE_RADIX_CACHE=1
```

`SGLANG_MAX_RUNNING_REQUESTS` should reflect real request concurrency. Leaving
it on automatic sizing made the TP=2 smoke case reserve capacity for 235
requests and capture 34 CUDA graphs even though only four requests were sent.

## Interpreting apparent serialization

Relax submits prompt groups with asyncio tasks, submits the samples inside each
group concurrently, and bounds the aggregate with
`sglang_server_concurrency * rollout_num_gpus / rollout_num_gpus_per_engine`.
The definitive runtime signal is an SGLang line such as:

```text
Decode batch, #running-req: 2, ... gen throughput (token/s): ...
```

`Using serial creation mode (fully_async=False)` is unrelated. It means a
colocated Actor and rollout service are created in sequence so they do not
compete for GPU memory during initialization.

For the coding-agent path, the Ray Serve rollout deployment also needs patch
`0003-rollout-proxy-concurrency.patch`. Without it, Serve's default
`max_ongoing_requests=5` can throttle a large Relax fan-out before requests
reach SGLang. The coordinator must have
`ROLLOUT_MAX_IN_FLIGHT >= rollout_batch_size * n_samples_per_prompt`; requests
above that value receive HTTP 429 rather than waiting in a queue.

Apply patch `0005-sglang-parent-gpu-affinity.patch` as well. In a colocated
one-TP1-engine-per-GPU setup, SGLang's scheduler processes already honor
`base_gpu_id`, but their HTTP/tokenizer parents otherwise all select CUDA
device 0. The patch selects each engine's assigned local GPU before launching
its parent and prevents rollout processes from leaving all of those
allocations on actor rank 0.

Patch `0006-train-stage-profiling.patch` adds opt-in, CUDA-synchronized timing
for actor forward/backward, optimizer step, and gradient clearing. The 9B
training entry enables it with `RELAX_PROFILE_TRAIN_STAGES=1`, emitting
`RELAX_TRAIN_STAGE` records from rank zero so slow updates can be separated
from rollout wait time.

Qwen3.5's hybrid GDN/Mamba scheduler selects `no_buffer`, which disables the
SGLang overlap schedule while preserving continuous request batching.
`SGLANG_DISABLE_RADIX_CACHE=1` is an optional tradeoff that enables overlap;
measure it on representative prompt lengths before using it in a long run.

On the four-H100 stock DAPO smoke test (four prompts, two samples each, 256
generated tokens), both rounds saved eight samples. The warm TP=1 baseline
reached about 249 token/s per engine. With radix cache disabled it reached
about 281 token/s, roughly 12% higher. Independently, setting
`SGLANG_MAX_RUNNING_REQUESTS=32` reduced TP=1 CUDA-graph capture from 16 graphs
and 33 seconds to eight graphs and about six seconds, and reduced the Mamba
cache request capacity from 290 to 32. These are small-run measurements, not a
substitute for an agent-workload benchmark with long repeated prefixes.

## Measured concurrency

All rows below used the official Qwen3.5-9B base checkpoint. "Trajectories"
means complete multi-turn coding-agent executions; one trajectory can issue
several LM requests over time.

| GPUs | Workload | Live work | Measured result |
| ---: | --- | ---: | --- |
| 4 | DAPO LM-only, TP=1 | 8 requests | warm batch 2.8 s; about 249 token/s/engine |
| 4 | real OmniCoding agent tasks | 8 trajectories | all eight staged in the same second; private-container wall time 15–35 s; Relax completed in about 45 s |
| 8 | DAPO LM-only, TP=1 | 16 requests | warm batch 2.37 s (6.74 samples/s); about 290 token/s/engine |
| 8 | real OmniCoding agent tasks | 16 trajectories | all 16 staged in the same second; results returned in 18–56 s; Relax completed in 67 s |

The eight-GPU agent run used four real records (AVUT, OVB, and Omnimodal), four
samples each. SGLang logged up to five simultaneously decoding requests on one
engine and about 600 aggregate generated token/s for that engine, so the
rollouts were not serialized. All 16 containers completed successfully.

That run deliberately stopped every trajectory after four tool turns to bound
the smoke-test cost. Consequently the samples exited with `step_limit` and were
removed from training; it is a concurrency/integration result, not an RL
quality result. The original worker image also lacked the complete SFT harness
stack. The `20260724-v5` image mounts the locked `.venv-harness` behind a
private writable overlay. In the isolated container, PyTorch 2.5.1+cu121 saw
exactly one H100, and faster-whisper completed a real CUDA/FP16 transcription
of `media/audios/000096.wav`.

Cold-start time is not rollout time. The first source-environment launch
imports the same large dependency graph in several Ray processes, converts a
Hugging Face checkpoint to Megatron, JIT-compiles some SM90 kernels, and
captures CUDA graphs. Keep one Ray job alive for many updates and save the
converted SFT model as a Megatron sharded checkpoint rather than repeating
HF-to-Megatron conversion for every job.
