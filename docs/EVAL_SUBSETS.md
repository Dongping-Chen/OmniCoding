# Reproducing the evaluation subsets

The archived OmniCoding runs did not all evaluate every item exposed by each
upstream benchmark. The exact metadata slices are defined in
`recipes/eval_subsets.json`; every recipe pins the upstream Hugging Face
revision, source filename, deterministic selection rule, row count, stable ID
field, and expected output SHA256.

| Recipe | Archived evaluation slice | Construction |
|---|---:|---|
| `omnigaia_full360` | 360 | Entire official test metadata file; used by the audited Codex gpt-5.5 xhigh and Code-X 27B runs |
| `omnigaia_first100` | 100 | First 100 rows in official order; retained for development/baseline jobs that used the smaller slice |
| `lvomnibench_first100` | 100 | First 100 rows in official order |
| `socialomni_l1_first100` | 100 | First 100 Level-1 rows in official order |
| `socialomni_l2_first100` | 100 | First 100 Level-2 rows in official order, preserving the upstream wrapper |
| `videozerobench_official500` | 500 | The official fixed `VideoZeroBench_500_v0.json`; no additional local sampling |

“First 100” means positional slicing of the pinned upstream JSON, not seeded
random sampling. VideoZeroBench describes a larger multi-level benchmark, but
the archived runs use its official 500-question file. Do not compare a score
from one of these slices with a score over a different split or question set.

## Build and verify a subset

Install the Hugging Face CLI, download the exact upstream file, then pass it to
the checksum-enforcing builder. For example:

```bash
pip install -U huggingface_hub

hf download \
  --repo-type dataset \
  --revision adf542cf70735e061b8d73300cdbae9c847b9bc3 \
  --local-dir upstream/omnigaia \
  RUC-NLPIR/OmniGAIA test_metadata.json

omnicoding-bench-subset \
  --recipes recipes/eval_subsets.json \
  --name omnigaia_first100 \
  --input upstream/omnigaia/test_metadata.json \
  --output benchmark_data/omnigaia/test_first100.json
```

List every available recipe with:

```bash
omnicoding-bench-subset --recipes recipes/eval_subsets.json --list
```

Replace the repository, revision, source file, and recipe name with the values
from `recipes/eval_subsets.json` for the other benchmarks. The command refuses
to write an output when the source digest, row count, stable IDs, or output
digest differs from the audited snapshot.

LVOmniBench is gated on Hugging Face. Accept its access conditions and
authenticate with `hf auth login` before downloading. The recipe reconstructs
metadata only; obtain media from the pinned official repository and comply
with each upstream dataset's terms. OmniCoding does not mirror benchmark media.
