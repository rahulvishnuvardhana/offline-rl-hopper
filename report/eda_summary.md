# EDA summary: Hopper expert-v0 and simple-v0

All numbers come from `report/eda_summary.json` (code commit `1a333f2`, `git_dirty: false`, 20 of 20 checks passed). Figures are in `report/figures/`. Spec: `docs/specs/phase-2b-eda.md`.

## 1. The data is clean and leak-free
- The counts reconcile across three sources (documentation, metadata, our sums), and the files re-hash to the Phase 2A audit values.
- Every reward rebuilds from survival + forward velocity - control cost to within -0.002 (expert) and -0.003 (simple) per step on average.
- No episode appears in both datasets.
- The datasets were collected from `reset(seed=123+i)` (verified for all 3,038 episodes). That is why their only shared states are the 1,086 initial states.
- Our evaluation seeds (10000+) and random-baseline seeds (0 to 99) never reuse a collection seed. A CI test keeps it that way.

## 2. Returns are bimodal, so every score also reports the fall rate (`returns_by_fall.png`)

| | Episodes | Fall rate | Mean return | Fell: mean (share of transitions) | No fall: mean (share) |
|---|---|---|---|---|---|
| expert | 1,086 | 28.4% | 3,858 | 3,103 (22%) | 4,157 (78%) |
| simple | 1,952 | 66.3% | 1,656 | 881 (34%) | 3,179 (66%) |

Simple's mean of 1,656 falls in the gap between its two groups: falling episodes have median 666 (`returns_by_fall.png`), and no non-falling episode scores below 3,136. This is why ADR-6 uses the D4RL scale: 0 = random policy (17.0 ± 1.6 SE over 100 episodes, every one falls, 23 steps on average) and 100 = expert mean (3,857.8). On that scale the simple dataset scores **42.7**.

**Phase 4 sanity bands** (fell mean to no-fall mean): expert **80.3 to 107.8**, simple **22.5 to 82.3**. The bands overlap. A simple-style policy that never falls (82) scores the same as an expert that falls now and then (80). So a score is only interpretable together with its fall rate. The H2 threshold of 90 is above the whole simple band, so a policy cannot pass H2 by copying the simple gait without falling.

## 3. H1: the sources visit different states, so BC is not averaging two actions in one state (`pca_coverage.png`, `knn_actions.png`)
- **Classifier:** a kNN classifier predicts the source from the state alone with **AUC 1.000** (min 0.9999 over 3 seeds, held-out episodes, standardized on train only).
- **Distance:** simple states are **10 times farther** from the nearest expert state (median 2.23) than from the nearest simple state (0.22).
- **PCA** (`pca_coverage.png`, read from the figure): simple occupies a compact core, and expert traces a wide outer cycle.
- **Actions:** near simple states, expert and simple neighbours act very differently (median gap 0.99 vs 0.18 for expert vs expert, ratio 5.6). The distances above show those "neighbours" are not the same states, so this gap mostly reflects different regions, not disagreement at one state.

Under the reading pre-registered in the spec, this **weakens H1's averaging explanation**. A refined hypothesis fits the data better:
- The two behaviours share only their **initial states**.
- At those states BC sees both behaviours. With little expert data the simple behaviour dominates.
- The rollout then enters the simple region, where only simple actions exist, and stays there.
- **BC's failure at low p would then be early mode selection followed by commitment, not averaging everywhere.**

This is testable in Phase 8 at no extra training cost: apply the classifier to evaluation states and report the fraction of time each policy spends in the expert region.

## 4. What the mixtures look like (`mixture_reference.png`)

Selected p values (all nine are in the JSON):

| p | 0 | 0.01 | 0.05 | 0.2 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|---|---|
| score of the mixture's own episodes | 42.8 | 43.1 | 44.4 | 49.8 | 63.2 | 78.6 | 100.1 |
| IQL reward scale `1000/(max-min)` | 0.368 | 0.269 | 0.261 | 0.256 | 0.251 | 0.251 | 0.251 |

- **The score curve is not linear in p.** Simple episodes are shorter, so a transition budget holds more of them. A method "doing as well as its data" at p = 0.5 should be compared with 63, not 71.
- **The reward scale is 27% lower already at p = 0.01** (about 10 expert episodes). A non-falling expert episode (about 4,157) out-earns the best simple episode (3,298), so it widens the return range. IQL's advantage weights scale with the rewards, so its effective temperature differs between p = 0 and every other p. Phase 3 must decide (ADR) whether to keep the per-mixture scale, as the IQL paper does, or fix one scale. This matters for H3.

## 5. Preprocessing implications
- **States must be normalized.** Per-dimension std ranges from 0.05 to 6.8, a 130-fold spread (`state_std.png`). The statistics in this report are descriptive only. Normalizers are fit on each mixture (ADR-5).
- **Velocities are clipped at ±10 in 25.3% of expert `obs[10]` (foot angular velocity) values**, 5.3% of `obs[9]` and 9.6% of simple `obs[10]`. Those dimensions lose information beyond the clip. This is acceptable as is, and noted for interpreting any foot-related failure.
- **Actions are not saturated** (at most 0.06% at the bound; std 0.39 to 0.72). A tanh-squashed Gaussian policy fits them without bound effects.
- **Most of the reward is forward velocity** (3.20 of 4.19 per step for expert, 2.24 of 3.23 for simple). Survival contributes 1 and control cost about 0.001 (`reward_terms.png`). The expert advantage comes from speed and from not falling.

## 6. Limitations
- The control-cost weight cannot be verified from the data: the term is too small, and it moves with speed. It is taken from the env definition.
- kNN results depend on standardized Euclidean distance over 11 dimensions with mixed units.
- `d_mix` is measured at simple test states and `d_same` at expert test states, so the action-gap ratio compares different query regions.
- The determinism check (C7) holds on one machine only.
- Whether the data was collected with stochastic SAC actions is still open. If it was, deterministic BC at p = 1 may score above 100. This is measured in Phase 4.
