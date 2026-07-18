# Third-party notices

OmniCoding includes or adapts portions of third-party open-source projects.
Those portions remain subject to their original terms; the project-level
CC BY-NC 4.0 license does not replace them. A copy of the Apache License 2.0
is provided in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## KIRA

- Project: KRAFTON-AI/KIRA
- Source: https://github.com/KRAFTON-AI/KIRA
- License: Apache-2.0
- Use here: coding-agent loop, tool, provider, serialization, and recovery
  patterns adapted into `src/omnicoding/agents/kira`.

## Relax

- Project: redai-infra/Relax and the audited Dongping-Chen/Relax fork
- Sources: https://github.com/redai-infra/Relax and
  https://github.com/Dongping-Chen/Relax
- License: Apache-2.0
- Use here: the external RL runtime pinned in `integrations/relax/UPSTREAM.md`,
  reviewable core patches, and the OmniCoding rollout/coordinator integration.

The Relax lineage also attributes portions of its implementation to
[THUDM/slime](https://github.com/THUDM/slime) and
[OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), both distributed
under Apache-2.0. OmniCoding preserves that attribution; it does not vendor the
complete upstream frameworks.

## ms-swift

- Project: ModelScope ms-swift
- Source: https://github.com/modelscope/ms-swift
- License: Apache-2.0
- Use here: parameterized SFT and checkpoint-consolidation integration scripts
  under `integrations/ms_swift`.

## Benchmarks and datasets

Benchmark adapter code is distributed here, but benchmark records, media, and
labels are not relicensed by OmniCoding. Their upstream terms and current
redistribution decisions are recorded in
[`release/DATA_LICENSE_REVIEW.md`](release/DATA_LICENSE_REVIEW.md). Users must
obtain benchmark assets from their official sources and comply with those
terms.
