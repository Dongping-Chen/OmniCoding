# ms-swift integration

The Python package produces ms-swift Agent JSONL with
`omnicoding-data-filter` or `omnicoding-data-convert`. Training is deliberately
kept as a thin integration because ms-swift has its own installation and GPU
compatibility matrix.

```bash
export MODEL=Qwen/Qwen3.6-27B
export DATASET=/path/to/sft_train.jsonl
export OUTPUT_DIR=/path/to/checkpoints/run-001
bash integrations/ms_swift/train_sft.sh
```

All important values can be overridden through environment variables. Extra
ms-swift arguments may be appended after the script name. Record the exact
base-model and dataset revisions in a run manifest before a full run.

After LoRA training:

```bash
python integrations/ms_swift/merge_lora.py \
  --base Qwen/Qwen3.6-27B \
  --adapter /path/to/checkpoint \
  --output /path/to/merged-model
```

The merge helper loads the full image-text model and saves its processor, so
the consolidated checkpoint retains the visual tower and multimodal input
configuration. It requires a recent Transformers release that supports
`Qwen3_5ForConditionalGeneration` and the Qwen3.6 model family.

The defaults in `train_sft.sh` reproduce the released checkpoint's recorded
SFT settings where they are portable: two epochs, 32,000-token examples,
LoRA rank 64, alpha 128, all-linear targets, and frozen vision/alignment
modules. GPU count, DeepSpeed mode, gradient accumulation, and storage paths
remain cluster parameters and should be recorded in a run manifest.
