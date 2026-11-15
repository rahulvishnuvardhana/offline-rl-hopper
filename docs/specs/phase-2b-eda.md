# Spec: Phase 2B, exploratory data analysis (EDA)

**Status:** draft for review | **Branch:** `phase-2-data-eda` | **Roadmap:** Phase 2, "Tasks: EDA" | **Depends on:** Phase 2A (validator, `report/data_audit.json`)

## 1. Goal
Produce the dataset facts the rest of the project depends on: the two reference returns that define the normalized score, evidence for or against hypothesis H1 (unimodal Gaussian BC averaging two behaviors), and numbers that justify the preprocessing choices (state normalization, reward scaling). Everything machine-readable, reproducible, and reconciled against the Phase 2A audit.

## 2. Inputs
- `mujoco/hopper/expert-v0` and `mujoco/hopper/simple-v0`, loaded with `offrl.data.load.load_episodes` (already validated: all 9 checks pass in `report/data_audit.json`).
- The recovered evaluation env (`dataset.recover_environment()`, Hopper-v5) for the random-policy baseline.

## 3. Outputs
| File | Content |
|---|---|
| `report/eda_summary.json` | All numbers below, per dataset, plus provenance (git commit, data sha256 copied from the audit) |
| `report/figures/*.png` | 6 figures listed in Section 4 |
| `report/eda_summary.md` | One page of human interpretation, written after reading the numbers |
| `configs/base.yaml` | `score.r_simple`, `score.r_expert` filled in |

## 4. Analyses
Observation layout (Hopper-v5, x excluded): `obs[0]` torso z, `obs[1]` torso angle, `obs[2:5]` thigh/leg/foot angles, `obs[5]` x velocity, `obs[6]` z velocity, `obs[7]` angular velocity of torso, `obs[8:11]` thigh/leg/foot angular velocities. Velocities are clipped to [-10, 10] by the env.

| # | Analysis | Output |
|---|---|---|
| A1 | Episode lengths: mean, std, min, median, max; histogram expert vs. simple | numbers + `lengths.png` |
| A2 | Undiscounted episode returns: mean, std, min, median, max; histogram | numbers + `returns.png` |
| A3 | Termination rate (fraction of episodes ending by falling vs. time limit) | numbers |
| A4 | Per-dimension state mean and std (over `obs[:-1]` of every episode, i.e. states that have an action) | numbers + `state_std.png` |
| A5 | Velocity clipping: fraction of values at abs 10 per dimension | numbers |
| A6 | Action mean/std per dimension; fraction at the bound | numbers |
| A7 | Reward vs. forward velocity `obs[5]` (2-D histogram, subsample) | `reward_vs_velocity.png` |
| A8 | State coverage: PCA (2 components) fitted on standardized states from both datasets (20k subsample each, seed 0) | `pca_coverage.png` + explained variance |
| A9 | Action disagreement in shared states: for 5k simple states, nearest expert state in standardized space; for pairs closer than the median expert-to-expert nearest-neighbour distance, the L2 action difference. Baseline: same statistic for expert-vs-expert pairs. | numbers + `action_disagreement.png` |
| A10 | Random policy: 10 episodes, seeds 0 to 9, uniform actions from the seeded action space; mean and std return | numbers |

## 5. Score reference values
`r_simple` = A2 mean return of simple-v0, `r_expert` = A2 mean return of expert-v0, as stated in the proposal. Normalized score = `100 * (R - r_simple) / (r_expert - r_simple)`. The random-policy return (A10) is reported for context only.

## 6. Design
- `src/offrl/analysis/eda.py`: pure functions taking `Sequence[Episode]` (and arrays) and returning dicts/arrays. No file I/O, no Minari import, so all are unit-testable on synthetic episodes.
- `scripts/eda.py`: loads data, calls the functions, writes JSON and figures. Deterministic: every subsample and the random policy use fixed seeds.
- PCA and nearest neighbours implemented with NumPy only (SVD; brute-force or chunked distance on at most 20k x 20k), no new dependencies.
- Figures: matplotlib, Agg backend, fixed size and DPI.

## 7. Reconciliation and audit checks (script fails if any is false)
1. Sum of episode lengths and episode counts equal the values in `report/data_audit.json`.
2. Dataset sha256 values equal the audit's.
3. Each episode return equals the sum of its rewards (recomputed independently).
4. `r_expert > r_simple`.
5. `configs/base.yaml` score values equal `eda_summary.json` (also a unit test).
6. Running the script twice produces identical JSON apart from the timestamp.

## 8. Tests
Synthetic episodes with known answers for: length/return stats, termination rate, clipping fraction, state stats excluding the final observation, PCA on data with a known dominant direction, nearest-neighbour action difference on constructed pairs, and the config-equals-summary check.

## 9. Definition of Done
- All tests and CI green; all Section 7 checks pass.
- `eda_summary.json`, 6 figures and `eda_summary.md` committed.
- `score.r_simple` and `score.r_expert` set in `configs/base.yaml`.

## 10. Open questions
1. Should the 0-point of the score be the simple dataset mean (as in the proposal) or the random policy?
2. Is the A9 threshold (median expert-to-expert NN distance) a sensible definition of "same state"?
