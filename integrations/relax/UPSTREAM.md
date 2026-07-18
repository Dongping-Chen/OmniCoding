# Relax upstream pin

The audited integration baseline is `Dongping-Chen/Relax` commit
`6932be2fad9488f37cdbadfac1d14dbce98fe2f1`. Its direct parent is
`redai-infra/Relax` commit
`5a9d977b687d1efc18a581bb6becda7986f0a9c7`; that upstream commit and history
were verified against the public RedNote repository before release. The fork
baseline adds the project-specific `relax-router` subtree and a Qwen3.6-27B
model recipe on top of that upstream revision.

Apply the numbered patches in this directory in order, then install this
monorepo into the same Python environment so Relax can import
`omnicoding.rl.rollout.generate`.

The base project is `redai-infra/Relax` under Apache-2.0 and also
contains code attributed upstream to Slime and OpenRLHF. Preserve their
notices and history.

Supported through Relax's standard registry: GRPO, GSPO, and SAPO. PPO and
REINFORCE++ have partial lower-level code but are not wired through the audited
standard registry and must not be advertised as runnable recipes.

For GRPO/GSPO/SAPO, use Relax's explicit KL-loss path when needed. The audited
reward-shaping `kl_coef` path computes KL but does not subtract it in
`get_grpo_returns`; this remains a documented upstream issue rather than a
silent recipe assumption.
