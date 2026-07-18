# Provenance

This monorepo is assembled from local research workspaces. Source code is
copied, never moved, and source workspaces remain the archival record.

## Primary sources

| Component | Source | Recorded revision | Notes |
|---|---|---|---|
| Harness, Kira, benchmark specs, synthetic pipeline | `Dongping-Chen/OmniCoding` | `341bb33f81fa1c8ed486628e49e2f42bf74fd2be` plus audited working-tree changes | Primary project source |
| Coding-agent RL integration | `Dongping-Chen/Relax` | router snapshot audited at `9a717a82aa20a1088f725cb56990c6186e0fb7a0` plus working-tree changes | Router code is namespaced here; core patches apply to baseline `6932be2` |
| SFT integration | `modelscope/ms-swift` | `68eae8c20bdffce8bab05f732d8b2934578bee88` | Only local integration scripts are copied |
| Dataset preparation | local `coding-agent-rl` workspace | no Git revision | Scripts and schema documentation only; no dataset/media copied |

## Upstream and adapted code

The release must retain attribution for at least:

- RedNote/Xiaohongshu Relax (Apache-2.0)
- ModelScope ms-swift (Apache-2.0)
- KRAFTON KIRA where code was adapted (Apache-2.0)
- Slime-derived Relax files identified by the Relax upstream history
- OpenRLHF-derived utility code identified in source headers
- Every benchmark implementation and dataset source listed in the dataset cards

The Relax core is not vendored. Three reviewable patches preserve the local
dual-clip, multimodal-input, and rollout-concurrency changes. Their byte-level
application is verified against the pinned `6932be2` tree during staging.

Project-specific code and documentation are released under CC BY-NC 4.0.
Third-party portions remain under their original terms, summarized in
`THIRD_PARTY_NOTICES.md`. Dataset and media licenses are tracked per record and
are not overridden by the project license.
