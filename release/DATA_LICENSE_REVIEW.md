# Dataset and benchmark release review

This is a staging decision record, not legal advice. It reflects the license
statements present in the audited local snapshots; verify each upstream card
again immediately before release.

| Asset | Local evidence | Release action |
|---|---|---|
| OmniGAIA | Project README says MIT; copied dataset card says Apache-2.0 and points to the official HF repository | Do not mirror media by default. Pin the official HF dataset revision and preserve its card/terms. Resolve the project-vs-dataset license distinction. |
| LVOmniBench | Local README points to `KD-TAO/LVOmniBench`; no license file was found in the audited checkout | Reference the upstream repository. Do not republish data/media until its dataset license is explicit. |
| SocialOmni | Local README points to `alexisty/SocialOmni`; local data docs say datasets are not tracked; no license file was found | Reference the upstream repository. Do not create a mirror until redistribution terms are explicit. |
| VideoZeroBench | Dataset card says CC BY-NC-ND 4.0, prohibits commercial use and modified distributions, and says original video copyright is not owned by the dataset authors | Keep adapter code here, but reference the official dataset. Do not publish a normalized/modified mirror without written permission. |
| Coding-agent trajectories | Messages and tool outputs may quote benchmark metadata or contain transformed frames, audio, OCR, ASR, web content, private paths, endpoints, or credentials | Publish text trajectories only after content/security scanning. Keep processed images and other media derivatives excluded until each upstream license permits redistribution. Keep partial stdout tails in a separate split or exclude them. |
| Omnimodal-Agent-SFT-2K source rows | Local aggregate card records Apache-2.0 | Preserve row-level source/license provenance and verify model-output redistribution terms. |
| OmniVideoBench source rows | Local aggregate card records CC BY-NC-ND 4.0 | Exclude transformed rows/media from a public synthetic release pending permission; ND is incompatible with casually publishing modified derivatives. |
| AVUTBenchmark source rows | Local aggregate card records license as unspecified | Exclude from public release until explicit permission/license is obtained. |
| Video-MME-v2 source rows | Local aggregate card records MIT | Verify media provenance and official dataset terms, then preserve attribution and row-level source IDs. |

## Current staging decision

The `coding-agent-rl` curation scripts remain listed as
`pending_license_review` in `SOURCE_MAP.yaml`; no dataset, media, or prompt
parquet was copied. The public Git repository can still release the adapters,
schemas, trajectory converter, filtering logic, and reconstruction recipes.

For Hugging Face, prefer links to official benchmark repositories over new
mirrors. Publish synthetic SFT/RL data only as license-compatible source
subsets, each with immutable revisions, source IDs, filters, model/provider
terms, contamination analysis, and checksums.

The trajectory files under `release/trajectory_manifests/` are metadata-only
audits. They do not grant permission to redistribute any referenced benchmark
media or raw trajectory payload.
