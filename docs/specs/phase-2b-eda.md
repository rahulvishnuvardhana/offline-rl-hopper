# Spec: Phase 2B, exploratory data analysis (EDA)

**Status:** v2, approved 2026-11-15 (v1 reviewed by two fresh reviewers; 17 changes accepted, 1 rejected, 1 deferred to Phase 4) | **Branch:** `phase-2-data-eda` | **Roadmap:** Phase 2, "Tasks: EDA" | **Depends on:** Phase 2A (validator, `report/data_audit.json`)

## 1. Goal
Produce the dataset facts the rest of the project depends on:
- the reference returns that define the normalized score (Section 5);
- the per-`p` reference curves and sanity bands that later phases are checked against;
- evidence for or against H1 (a unimodal Gaussian BC averaging two behaviors);
- numbers that justify the preprocessing choices (state normalization, reward scaling).

Everything is machine-readable, reproducible and reconciled against the Phase 2A audit.

## 2. Inputs
- `mujoco/hopper/expert-v0` and `mujoco/hopper/simple-v0`, loaded with `offrl.data.load.load_episodes` (all 9 checks pass in `report/data_audit.json`, audit commit `19c118c`).
- The recovered evaluation env (`dataset.recover_environment()`, Hopper-v5) for the random policy.

## 3. Outputs
| File | Content |
|---|---|
| `report/eda_summary.json` | Every number below, per dataset, plus provenance (Section 7) |
| `report/figures/*.png` | 8 figures (Section 4), each under 1 MB |
| `report/eda_summary.md` | One page of interpretation, written after reading the numbers |
| `configs/base.yaml` | `score.r_random`, `score.r_expert` (and `score.r_simple` for reference), written only with `--write-config` |

## 4. Analyses
Observation layout (Hopper-v5, x excluded): `obs[0]` torso z, `obs[1]` torso angle, `obs[2:5]` thigh/leg/foot angles, `obs[5]` x velocity, `obs[6]` z velocity, `obs[7]` torso angular velocity, `obs[8:11]` thigh/leg/foot angular velocities. The env clips velocities to [-10, 10]. The health rule's `|state| < 100` bound applies to the unclipped state, so it cannot be checked from observations; we only report clipping.

"State statistics" always use `obs[:-1]` (states that have an action). **All dataset-wide state statistics in this phase are descriptive only.** Training normalizers are fit on each mixture (ADR-5) and never on these numbers.

**Subsampling rule (A8, A9):** draw states uniformly over all transitions (seeded) and keep each state's episode id. Every train/test split is by episode, so states of one episode never sit on both sides. (Implementation note: drawing whole episodes first would give only about 22 expert episodes per 20k states, which is too few for an 80/20 split.) Seeds 0, 1, 2 for each subsampled analysis; report the mean and the min to max range across seeds.

| # | Analysis | Output |
|---|---|---|
| A1 | Episode lengths: mean, std, min, median, max | numbers + `lengths.png` |
| A2 | Episode returns: mean, std, min, median, max | numbers + `returns.png` |
| A3 | Fall rate: fraction of episodes ending in termination vs. truncation at 1,000 | numbers |
| A3b | **Fall split:** A1 and A2 statistics for `fell` and `no_fall` episodes separately, plus each group's share of episodes and transitions | numbers + `returns_by_fall.png` |
| A4 | Per-dimension state mean and std (descriptive only) | numbers + `state_std.png` |
| A5 | Velocity clipping: fraction of values with `abs == 10` (exact, `np.isclose(atol=1e-9)`) per dimension `obs[5:11]`, per dataset | numbers |
| A6 | Action mean and std per dimension; fraction at the bound | numbers |
| A7 | **Reward decomposition:** per step, `healthy = 1.0` except 0 on the terminal (fall) step (Hopper-v5 pays it only while healthy), `forward = next_obs[5]`, `ctrl = 1e-3 * ||a||^2`; mean of each term per dataset; residual `rew - (healthy + forward - ctrl)` summary (mean, median and p99 of abs) | numbers + `reward_terms.png` |
| A8 | State coverage: PCA (2 components) on pooled standardized states, 20k per source per seed; signs fixed so the largest-abs loading of each component is positive | `pca_coverage.png` + explained variance |
| A9 | **Source separability and action disagreement** (replaces v1's nearest-neighbour threshold), Section 4.1 | numbers + `knn_actions.png` |
| A10 | Random policy: 100 episodes, seeds 0 to 99, `env.reset(seed=s)` and `env.action_space.seed(s)`; mean, std, SE | numbers |
| A11 | **Per-`p` dataset reference curve:** the expected normalized score of each mixture's own episodes for `p` in {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0}, using the Phase 3 sampling rule (Section 4.2) | numbers + `mixture_reference.png` |
| A12 | **IQL reward-scale preview:** for the same `p` values, `1000 / (max_ret - min_ret)` over the mixture's untrimmed episodes (Section 4.2) | numbers |

### 4.1 A9 detail
Pools per seed: 20k expert and 20k simple states (by the subsampling rule). Split by **episode**, 80% train and 20% test, then standardized with the mean and std of the **train** states only (no test information in the distance metric).
1. **Separability:** kNN classifier (k = 10, Euclidean) predicting the source from the state alone. Score = fraction of expert neighbours in the train pool. Report held-out ROC AUC.
2. **Action disagreement:** for each simple test state, `d_mix` = L2 distance between the mean action of its k expert train neighbours and the mean action of its k simple train neighbours. Baseline `d_same`: split the expert train episodes into two halves and compute the same statistic between the halves for each expert test state. All four neighbour pools are half of a source's train episodes, so no pool is denser than another. Report the medians and the ratio `d_mix / d_same`.
3. **Neighbour distances** (added during implementation): median distance from simple test states to their nearest expert and nearest simple train state. If the expert neighbours are much farther away, `d_mix` partly measures distance between states, not between policies.

Pre-registered reading:

| Result | Reading |
|---|---|
| AUC near 1 | The sources occupy different states, so a state-conditioned BC can tell them apart. This weakens H1's averaging explanation. |
| AUC well below 1 and `d_mix / d_same` clearly above 1 | The same states get different actions. This supports H1. |

Memory: distances are computed in blocks of 512 test rows against the train pool (512 x 32k x 8 bytes is about 130 MB), never the full matrix.

### 4.2 Mixture sampling rule used by A11 and A12
For each of seeds 0, 1, 2: permute each source's episode indices (0 to n-1 in load order; equal to the episode ids for both datasets) with one generator seeded by the mixture seed, expert first, then simple, and take whole episodes until the source's budget (`p * 900,000` expert, the rest simple) is reached. The last episode is trimmed (it counts as truncated, not fallen). A11 averages episode returns over the mixture, with a trimmed episode counted at its trimmed return. A12 uses untrimmed episodes only: a trimmed tail can be a few steps long and would otherwise set the minimum, so the scale would depend on where the budget cut fell (found in code review). Report the mean and range across seeds.

This is a preview function in `eda.py`. Phase 3's `mixture.py` must reproduce these numbers for the same seeds, and its test reconciles them against `eda_summary.json`.

## 5. Score definition (changes ADR-6, before any training result exists)
v1 used 0 = simple mean and 100 = expert mean. The returns are bimodal, so that scale gives no-fall simple episodes a score of 69 and no-fall expert episodes 114 (fall rates 66.3% and 28.4%). We therefore use the D4RL convention:

- `r_random` = A10 mean return, `r_expert` = A2 mean return of expert-v0.
- Normalized score = `100 * (R - r_random) / (r_expert - r_random)`.
- `r_simple` (A2 mean of simple-v0) is recorded and shown as a reference line.
- **Every reported score also gives the raw return and the fall rate.**
- H2's threshold (score >= 90) is unchanged; on this scale it means 90% of the expert dataset mean above random.

**Sanity bands for Phase 4 (pre-registered here, written to `eda_summary.json` as `sanity_bands`):** for each dataset, the band is [score of the `fell` mean return, score of the `no_fall` mean return] from A3b. BC at `p = 1` should land in the expert band and BC at `p = 0` in the simple band. Its fall rate is reported next to it.

## 6. Design
- `src/offrl/analysis/eda.py`: pure functions taking `Sequence[Episode]` and arrays, returning dicts and arrays. No file I/O, no Minari import, so all are unit-testable on synthetic episodes. Includes `knn_indices`, `pca_2d` (NumPy SVD), `roc_auc` (rank based), `sample_mixture_episodes` and `subsample_states`.
- `src/offrl/rollout.py`: `run_policy(env, policy_fn, seeds) -> returns, lengths, fell`, used here for the random policy and reused by Phase 4's `eval.py`.
- `src/offrl/analysis/eda_plots.py`: one function per figure, taking the summary dict and arrays and returning a `Figure`. Agg backend, fixed size and DPI, `metadata={"Software": None}` in `savefig` so PNGs carry no version stamp; scatter plots are subsampled and rasterized to stay under 1 MB.
- `scripts/eda.py`: loads the data, calls the functions, runs the Section 7 checks, writes the JSON (`allow_nan=False`, `indent=2`, LF line endings) and figures. Flags: `--write-config` (writes the Section 5 values to `configs/base.yaml`), `--allow-dirty` (development only; the run is then marked `git_dirty: true`).
- `offrl.runmeta.PACKAGES` gains `matplotlib`. `scripts/validate_data.py` also writes LF line endings.
- NumPy only for PCA, kNN and AUC. No new dependencies.

## 7. Reconciliation and audit checks (the script exits 1 if any fails)
| # | Check | Independent of |
|---|---|---|
| C1 | Step and episode counts equal `report/data_audit.json` | The audit's own computation |
| C2 | **Re-hash** the dataset files now and compare with the audit's `file_sha256` | The audit's stored hashes |
| C3 | Reward decomposition (A7): abs(mean residual) < 0.02 and median abs residual < 0.1 per dataset. Measured while writing this spec: mean -0.002 / -0.004, median 0.034 / 0.022 (official run, after the survival-bonus fix: -0.0021 / -0.0029). A wrong reward scale or a missing survival bonus would fail it. Limitation: the control cost (about 0.001 per step) is too small to verify this way, and a regression on `||a||^2` is confounded with speed (fitted weight -0.006 on expert), so that term is taken from the env definition. | The recorded rewards |
| C4 | Fall split adds up: `n_fell + n_no_fall = n_episodes`, transitions likewise, and the group means weighted by count reproduce the A2 mean (`rtol=1e-12`) | Bookkeeping check of the split only (it reuses the A2 returns) |
| C5 | `r_expert > r_simple > r_random` | |
| C6 | A11 at `p = 0` and `p = 1` equals the A2-based scores of simple and expert within the trimming effect (abs diff < 1 point) | A2 |
| C7 | Determinism: the analysis is computed twice in one process and every number agrees (`rtol=1e-9`). Same-machine guarantee only. | |
| C8 | Clean tree: refuses to run if `git status --porcelain` shows any change, including stale outputs or an edited audit in `report/` (unless `--allow-dirty`); passes only if the recorded `git_dirty` is false. `git` always runs in the repo root. Records `git_commit`, `git_dirty` and the SHA-256 of the **committed blob** of `report/data_audit.json` (what a fresh clone gets). C8b: the disk copy must be byte-identical to that blob, because `git status` ignores line-ending differences that `.gitattributes` normalizes (found in pre-merge review). | |
| C9 | **Leakage, collection seeds:** episode i of each dataset starts at `env.reset(seed=123 + i)` exactly (verified for all 1,086 + 1,952 episodes), so the collection seeds are known: expert 123 to 1208, simple 123 to 2074 | |
| C10 | **Leakage, rollout seeds:** evaluation seeds (`offrl.rollout.eval_seeds`) and random-baseline seeds never reuse a collection seed. MuJoCo is deterministic, so such an episode would start from a training state and a memorizing policy could replay it. Also a CI test on `configs/base.yaml` | |
| C11 | **Leakage, cross-dataset:** no episode appears in both datasets (by action hash and by observation hash), so `p` means what it says | |
| C12 | **Leakage, shared states:** exact states present in both datasets are all initial states (same reset seeds); none mid-trajectory | |

Separately, `tests/test_eda_config.py` asserts that `configs/base.yaml` score values equal `eda_summary.json` (`rtol=1e-9`). It is a pytest, not a script check, so the script never checks its own output.

**Commit order (provenance):**
1. Commit the code.
2. Run `python scripts/eda.py --write-config` on the clean tree.
3. Commit `report/` and `configs/base.yaml` together.

The summary's `git_commit` then names the code commit, as with the Phase 2A audit.

## 8. Tests
Synthetic episodes with known answers, each asserting exact values:
- length, return and fall statistics, and the fall split (C4 identities);
- clipping fraction; state stats exclude the final observation;
- reward decomposition with a constructed residual of 0;
- PCA recovers a known dominant direction, with the sign convention;
- `knn_blocks` equals a brute-force result for several block sizes;
- `roc_auc` gives 1.0, 0.5 and 0.0 on constructed scores;
- `sample_mixture_episodes` hits the budget exactly, with `p = 0` taking no expert and `p = 1` no simple;
- `run_policy` with a fake env.
- Plot smoke tests: every figure function runs on synthetic data and returns a `Figure` with the expected number of axes.
- The config-equals-summary test.

## 9. Definition of Done
- All tests and CI green; all Section 7 checks pass; the summary has `git_dirty: false`.
- `eda_summary.json`, 8 figures and `eda_summary.md` committed; every figure under 1 MB.
- Section 5 values in `configs/base.yaml`; ROADMAP updated (ADR-6, Phase 2 deliverables, Phase 4 DoD, layout).
- A fresh reviewer has reviewed the PR diff before merge.

## 10. Deferred
- Whether simple-v0 and expert-v0 were collected with stochastic SAC actions. If so, deterministic BC at `p = 1` may score above the dataset mean. Measured in Phase 4.
