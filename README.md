# OmniCoding

<div align="center">
  <p>
    <a href="https://arxiv.org/abs/2606.00579"><img src="https://img.shields.io/badge/arXiv-2606.00579-b31b1b.svg" alt="arXiv"></a>
    <a href="https://huggingface.co/shuaishuaicdp/Code-X-SFT-27B"><img src="https://img.shields.io/badge/Hugging_Face-Code--X--SFT--27B-FFD21E?logo=huggingface&amp;logoColor=black" alt="Hugging Face model"></a>
    <img src="https://img.shields.io/badge/python-%E2%89%A53.10-blue?logo=python&amp;logoColor=white" alt="Python 3.10 or newer">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-CC_BY--NC_4.0-green" alt="CC BY-NC 4.0 license"></a>
    <a href="https://github.com/Dongping-Chen/OmniCoding/actions/workflows/tests.yml"><img src="https://github.com/Dongping-Chen/OmniCoding/actions/workflows/tests.yml/badge.svg?branch=main" alt="Tests"></a>
  </p>
  <p>
    <a href="https://github.com/Dongping-Chen/OmniCoding/graphs/commit-activity"><img src="https://img.shields.io/github/commit-activity/m/Dongping-Chen/OmniCoding?label=commit%20activity" alt="Monthly commit activity"></a>
    <a href="https://github.com/Dongping-Chen/OmniCoding/issues?q=is%3Aissue%20is%3Aclosed"><img src="https://img.shields.io/github/issues-search?query=repo%3ADongping-Chen%2FOmniCoding%20is%3Aissue%20is%3Aclosed&amp;label=issues%20closed&amp;labelColor=%237d89b0&amp;color=%235d6b98" alt="Issues closed"></a>
  </p>
</div>

Official codebase for **Sandboxed Coding Agents are Competitive Omni-modal
Task Solvers** ([arXiv:2606.00579](https://arxiv.org/abs/2606.00579)).

OmniCoding studies how text-and-image coding agents can solve video, audio,
image, document, and cross-modal tasks without placing all raw media directly
in the model context. Media stays inside a sandboxed terminal workspace, where
the agent can use tools such as `ffmpeg`, `ffprobe`, ASR, OCR, search, and
Python to extract compact evidence before answering.

This repository packages the complete research workflow in one codebase:
multiple coding-agent harnesses, five benchmark adapters, verified synthetic
trajectory processing, supervised fine-tuning, checkpoint serving, evaluation,
and Relax-based reinforcement learning. **Code-X** is the accompanying
post-training recipe, using synthetic SFT followed by GSPO-style RL with a
process-aware verifiable reward.

## Resources

- Paper: [arXiv:2606.00579](https://arxiv.org/abs/2606.00579)
- Model: [Code-X-SFT-27B](https://huggingface.co/shuaishuaicdp/Code-X-SFT-27B)
- Data and benchmark release policy: [docs/DATA_RELEASE.md](docs/DATA_RELEASE.md)
- Source/provenance map: [PROVENANCE.md](PROVENANCE.md)

The 27B release is a supervised fine-tune of
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B). Synthetic dataset
links will be added separately after the corresponding source and media
license checks are complete.

## Approach

![Accuracy-token tradeoff](assets/token_usage.png)

Sandboxed coding agents selectively inspect media through terminal tools and
can remain competitive with native omni-modal systems while consuming much
less media context.

![Tool-use distribution](assets/tool_use.png)

The resulting trajectories contain staged tool pipelines across search, media
extraction, transcription, OCR, and Python. OmniCoding turns those trajectories
into auditable SFT and RL data while keeping the harness, benchmark, model, and
execution environment independently configurable.

## Paper highlights

- GPT-5.4 xHigh under Codex reaches **75.0%** on OmniGAIA, compared with
  **66.1%** for Gemini 3.1 Pro in the study.
- The best coding-agent setting reaches **27.6%** on VideoZeroBench, above the
  strongest native omni baseline reported in the paper (**17.8%**).
- The Code-X 27B setting reaches **43.3%** on OmniGAIA and **60.0%** on
  LVOmniBench.
- Log-driven skill self-distillation improves GPT-5.4 high on OmniGAIA from
  **61.4%** to **76.7%** average accuracy.

Evaluations in the paper cover OmniGAIA, SocialOmni, LVOmniBench,
VideoZeroBench, and TerminalBench-O. This initial code release includes public
adapters for OmniGAIA, SocialOmni Levels 1 and 2, LVOmniBench, and
VideoZeroBench.

## Repository layout

```text
src/omnicoding/
  agents/kira/          Kira coding-agent implementation
  benchmarks/           shared runtime, benchmark specs, prompts, evaluation
  harnesses/            Codex, Claude Code, and Kira runners
  data/                 trajectory auditing, filtering, and SFT conversion
  rl/                   rollout, reward, and authenticated coordinator code
integrations/
  ms_swift/             SFT and LoRA consolidation integration
  sglang/               checkpoint serving integration
  relax/                pinned RL integration and reviewable upstream patches
infra/slurm/             portable Slurm templates for harness, SFT, serving, RL
recipes/                 example run configuration
release/                 source map, exclusions, and license review records
tests/                   unit, integration, security, and packaging tests
```

Large datasets, benchmark media, outputs, and model weights are deliberately
kept outside Git and referenced through versioned repositories.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,benchmark,rl,rl-data]"
pytest
```

Benchmark, SFT, serving, and RL dependencies are intentionally separated. See
the relevant integration README before installing GPU-specific stacks.

## Running a harness

One launcher selects the harness, benchmark, model, and data paths
independently:

```bash
omnicoding-run \
  --harness kira \
  --bench omnigaia \
  --model openai/shuaishuaicdp/Code-X-SFT-27B \
  --input-file /path/to/items.json \
  --dataset-root /path/to/media \
  --output-dir outputs/smoke \
  -- --provider qwen --max_items 8
```

Use `--dry-run` to inspect the delegated command. The same entry point accepts
`--harness codex` and `--harness claude`. The `openai/` prefix selects
LiteLLM's OpenAI-compatible transport; `--provider qwen` independently keeps
the model's Qwen chat-template and multimodal tool-message behavior.

## Synthetic data, SFT, inference, and RL

- [Synthetic trajectory to SFT workflow](docs/SYNTHETIC_AND_SFT.md)
- [ms-swift training integration](integrations/ms_swift/README.md)
- [SGLang serving integration](integrations/sglang/README.md)
- [Relax coding-agent RL integration](integrations/relax/README.md)
- [27B GSPO recipe](recipes/rl_27b_gspo.sh)
- [Slurm templates](infra/slurm)

The checked-in Slurm scripts are portable templates. Cluster account,
partition, QoS, storage paths, and credentials must be supplied by the user or
through an ignored local profile.

## Security

Coding agents execute tools and shell commands. Run them inside an isolated
container or job sandbox, never expose production credentials, and review
generated commands. The RL coordinator requires bearer authentication,
request bounds, model/origin allowlists, media-path containment, and a small
environment allowlist. See [docs/RL_SECURITY.md](docs/RL_SECURITY.md).

## License

Project-specific code and documentation are released under
[CC BY-NC 4.0](LICENSE). This is a non-commercial research release. Third-party
and adapted components retain their original licenses and attribution; see
[PROVENANCE.md](PROVENANCE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Datasets, benchmark media, and model checkpoints may have separate terms.

## Citation

```bibtex
@misc{chen2026sandboxedcodingagentscompetitive,
      title={Sandboxed Coding Agents are Competitive Omni-modal Task Solvers},
      author={Dongping Chen and Xuanao Huang and Zhihan Hu and Qingyuan Shi and Dianqi Li and Tianyi Zhou},
      year={2026},
      eprint={2606.00579},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.00579},
}
```
