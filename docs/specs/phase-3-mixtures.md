# Spec: Phase 3, mixture builder and data pipeline

**Status:** v2, after review by two fresh reviewers (engineering, RL): 2 blockers, 8 majors and 11 minors addressed; user decisions 2026-11-15: ADR-9 Option A with a report note, no validation split | **Branch:** `phase-3-mixtures` | **Roadmap:** Phase 3 | **Depends on:** Phase 2 (merged `79f54f6`)

## 1. Goal
Turn the two validated datasets into the project's independent variable: deterministic mixtures of exactly 900,000 transitions with expert fraction `p`, plus the normalizers and the replay buffer every training run uses. Every mixture is reproducible from `(p, seed)`, carries a manifest with hashes, and reconciles against the original data and the Phase 2 previews.

## 2. Scope and seeds
- **In:** `offrl/data/mixture.py`, `offrl/data/normalize.py`, `offrl/data/buffer.py`, `scripts/build_mixtures.py`, tests, manifests, audit.
- **Out:** networks, losses, training loop, evaluation (Phase 4).
- **Final grid:** `p` in {0, 0.05, 0.2, 0.5, 1.0} (F2) x mixture seeds {0, 1, 2}. A run with seed `s` uses mixture seed `s` for both BC and IQL, so the two algorithms train on identical data (N4; batch order is a separate, run-seeded stream).
- **Development mixtures:** seeds 100+ (built on demand). Phases 4 to 6 (debugging, the IQL gate) use only development mixtures and development evaluation seeds, so no debugging decision is tuned on final data. Seeds 3 and 4 and stretch `p` in {0.01, 0.02} are built on demand with the same code.
- **No validation split** (user decision): models are judged only by environment evaluation at fixed, pre-registered training steps (final score = mean of the last 5 evaluations, ROADMAP F4). No early stopping, no checkpoint selection.

**Statistical structure (document in the report):** one generator draws the expert order and then the simple order, independently of `p`. Mixtures with the same seed are therefore **nested across `p`** (the expert episodes at `p = 0.05` are a prefix of those at `p = 0.5`; likewise for simple). This is common random numbers: comparisons along `p` within a seed are less noisy, and results at different `p` within a seed are correlated. Across seeds the episode overlap (Jaccard) is about 0.81 at `p` in {0, 1} and 0.71 at `p = 0.05` (reviewer measurement), so seed-to-seed variance mostly reflects training randomness plus the mixture variation at intermediate `p`. Phase 8 reports the paired per-seed difference IQL minus BC alongside the rliable IQM and the probability of improvement.

## 3. Mixture construction
1. **Module layout (no import cycle):** `EXPERT`, `SIMPLE`, `SOURCES = ("expert", "simple")` (order fixed: the RNG stream depends on it) and `sample_mixture_episodes` move to `offrl/data/mixture.py`. `offrl/analysis/eda.py` imports them from there, so Phase 2 and Phase 3 cannot drift. A test pins `SOURCES` and checks that both modules expose the same function object.
2. **Episode index:** position in the list returned by `load_episodes`, which equals the episode id for both datasets (verified in Phase 2 and by the reviewer).
3. **Picks:** `sample_mixture_episodes(lengths, p, 900_000, seed)` returns, per source, an ordered list of `(episode_index, n_taken)`. The last pick per source may be trimmed (`n_taken < length`). The unused tail of a trimmed episode is simply not part of the mixture.
4. **Flatten** within each picked episode only, for `t < n_taken`: `obs = ep.obs[t]`, `act = ep.act[t]`, `rew = ep.rew[t]` (unscaled), `next_obs = ep.obs[t + 1]`, `done = ep.term[t]`. A trimmed episode's last row therefore has `done = 0` and a real `next_obs` (the episode continued). Analysis columns: `source` (int8: 1 expert, 0 simple), `episode_id` (int32), `t` (int32). Row order: expert picks then simple picks, each in pick order, steps ascending.
5. **File:** `{mixture_dir}/mix_p{p:g}_s{seed}.npz` via `np.savez` (byte-deterministic, verified twice by reviewers), containing the **raw, unnormalized** arrays above as float32 (`obs`, `act`, `rew`, `next_obs`, `done`) plus the analysis columns, and the normalizer arrays, `state_std_eps` and `reward_scale` (Section 4), so a mixture file is self-contained. About 105 MB each. Local `data/mixtures/` (gitignored), Explorer `/scratch/$USER/mixtures/`.
6. **`content_sha256`:** SHA-256 over the arrays in sorted key order; for each key its name, dtype string, shape, then the C-order little-endian bytes. Independent of the zip container, so it is the value compared across machines (laptop Windows vs. Explorer Linux). `file_sha256` is also recorded but only compared on the same machine. `data_sha256` hashes the rows and columns only: if two machines disagree on `content_sha256` but agree here, the difference is float summation in the normalizer, not the data.
7. **Manifest** `report/mixtures/mix_p{p:g}_s{seed}.json` (committed): `p_target`, `p_actual` (`= round(p * 900_000) / 900_000`), `n_transitions`, `seed`, `picks` (per source, the `(episode_id, n_taken)` list), `n_episodes`, `n_trimmed`, `source_versions`, `source_sha256` (from the Phase 2 audit), `file_sha256`, `content_sha256`, `data_sha256`, normalizer values, the Section 5 summary, `git_commit`, package versions (numpy pinned: PCG64 permutation and summation must match across machines).

## 4. Normalization (ADR-5: fit on the mixture's own transitions only)
- **States:** mean and population std (`ddof = 0`) of the mixture's `obs` rows, **accumulated in float64 over the float64 source rows** (via the picks, not the float32 file). Stored: `state_mean` and `state_std` **without** the epsilon, both float64; the transform is `(x - mean) / (std + 1e-3)` (`data.state_std_eps`), applied in the buffer to `obs` and `next_obs` and by the evaluator to environment observations.
- **Rewards:** `reward_scale = 1000 / (max_ret - min_ret)` over the mixture's **untrimmed** episodes, float64, from the source rewards. Stored for every mixture; **applied only when `data.scale_rewards` is true (IQL)**, inside the buffer. BC receives unscaled rewards (and does not use them).
- **ADR-9 (decided: Option A, per-mixture scale).** Wording, corrected after review: this is the IQL paper's reward normalization applied to complete episodes only, as CORL does; the original IQL code also keeps the final partial segment in the min/max, which here would let a 9-step tail set the minimum (scale 0.229 at `p = 0.5`, seed 0). State normalization follows CORL (the original IQL code does not normalize states); recorded as a deviation under ADR-3.
  - Measured per-seed scales: `p = 0` 0.367 to 0.369; `p = 0.05` 0.251, 0.266, 0.266 (depending on whether one low-return expert episode is drawn); `p >= 0.5` 0.251.
  - **Report note (pre-registered):** IQL's advantage weight is `exp(beta * A)` and `A` scales with the reward scale, so the effective temperature at `p = 0` is about 1.4x that at other `p` (`beta_eff` about 4.4 vs. 3.0). The `p = 0` IQL result is interpreted with that caveat. Every run logs `beta_eff = beta * reward_scale / 0.2512` (0.2512 = the `p = 1` scale).

## 5. Per-mixture summary (manifest and `report/mixtures/summary_{seeds}.json`)
`p_actual`, `n_episodes`, `n_trimmed`, mean episode return, normalized score (ADR-6), fall rate (untrimmed episodes), `reward_scale`, `beta_eff`, and two dataset-quality measures in the spirit of [18]:
- **TQ:** mean normalized score of the untrimmed episodes.
- **SACo proxy:** unique `(s, a)` cells after standardizing with **one fixed reference normalizer per seed** (the `p = 0` mixture's; analysis only, so ADR-5 does not apply) and binning each of the 14 dimensions at **0.5 and 1.0** reference std, divided by the same count for the `p = 0` mixture. Labelled a proxy (at 0.5 std about half the rows occupy their own cell).

## 6. Replay buffer
`ReplayBuffer.from_mixture(path_or_arrays, scale_rewards, device, seed)` (the only constructor training code uses; the normalizer, eps and reward scale come from the mixture itself, so another mixture's statistics cannot leak in): normalizes and scales once, moves `obs`, `act`, `rew`, `next_obs`, `done` to `device` as float32 (about 100 MB). `sample(batch_size) -> Batch` (ROADMAP 3.4) draws indices with a **CPU** `torch.Generator` seeded from the run seed and transfers them, so the index sequence is identical on laptop and cluster. `rew` and `done` are shaped `(B, 1)`.

## 7. Checks (`offrl/data/mixture_audit.py`, run by `scripts/build_mixtures.py`, which exits 1 on any failure; `report/mixtures/audit_{seeds}.json`)

A corrupted mixture yields a FAIL, never a crash; `tests/test_mixture_audit.py` corrupts a synthetic mixture ten ways and asserts each is caught.
| # | Check | Independent of |
|---|---|---|
| M1 | 900,000 rows; `p_actual` as defined (guard: true by construction, not evidence) | |
| M2 | `p = 0` has no expert rows; `p = 1` no simple rows | |
| M3 | For every row, `(obs, act, rew, next_obs, done)` equals the float64 source episode at `(episode_id, t)` cast to float32, exactly | The flattening code |
| M4 | `done` equals source `term` for every row; trimmed last rows have `done = 0`; no truncated step has `done = 1`; a trimmed row's `next_obs` differs from the following row's `obs` | |
| M5 | Picks re-derived from the saved `episode_id`/`t` columns equal the manifest picks; the mean return and `reward_scale` recomputed from the float64 sources via those picks equal `report/eda_summary.json` per seed for every `p` in its grid (picks exact, sums rtol 1e-12) | The mixture file and Phase 2 code paths |
| M6 | The **stored** `state_mean`, `state_std`, `state_std_eps` and `reward_scale` (what training reads) equal a float64 recomputation from the sources via the index (rtol 1e-12; eps exact); mean and std of the float32 file rows within 1e-6 std of the stored values | The fitting code |
| M7 | Rebuilding the same `(p, seed)` gives the same `file_sha256` and `content_sha256`; hashes differ across `p` within a seed and across seeds for each `p` (guards, not evidence) | |
| M8 | Clean tree, committed-blob hashing of the audit, `git_commit` recorded (as Phase 2 C8/C8b) | |
| M9 | **Informational, never blocks (errors are recorded, not raised):** replay 3 episodes per source (per-source index `i`, seed `123 + i`) with the dataset actions; record max error over steps 1 to 100 and the MuJoCo version (reviewer measured at most 4e-14) | |

**Cross-machine reconciliation (DoD):** mixtures built on Explorer reproduce the laptop `content_sha256` values.

## 8. Tests (CI: logic on synthetic data; the local audit covers the real data)
- Budget exactness and trimming; `p = 0` and `p = 1`; picks format.
- Flattening never crosses episodes (trap: adjacent episodes with distinguishable values); trimmed last row `done = 0` and `next_obs` is not the next row's `obs`.
- `done` on terminated, truncated and trimmed episodes.
- `content_sha256` is order- and dtype-sensitive and container-independent; determinism and seed sensitivity.
- Normalizer: float64 accumulation, `ddof = 0`, epsilon applied in the transform not stored; reward scale ignores trimmed episodes; scaling applied only when enabled.
- Buffer: shapes `(B, 11)`, `(B, 3)`, `(B, 1)`, `(B, 11)`, `(B, 1)`; float32; same seed gives the same indices; CPU generator. A CUDA variant is skipped when CUDA is absent (CI is CPU-only).
- `SOURCES` order pinned; `eda.sample_mixture_episodes is mixture.sample_mixture_episodes`.
- Manifest round-trip; filename format `p{p:g}`.

## 9. ROADMAP and config updates in this PR
- **ADR-9** row (per-mixture reward scale, Option A, with the `p = 0` caveat); ADR-3 notes the two deviations (complete-episode min/max as in CORL; CORL-style state normalization).
- **3.4 contracts:** manifest fields as in Section 3.7 (removes the impossible `p_actual: 0.0501` example and the single `sha256` key); manifests live in `report/mixtures/`.
- **Seeds:** `configs/base.yaml` `eval.seed_offset: 20000` (development); the final grid sets 10000 explicitly. Development mixture seeds 100+. `tests/test_leakage.py` covers both offsets.
- **Phase 4:** final score over 50 evaluation episodes (10 per periodic evaluation); seed disjointness prevents exact replay of a training trajectory but starting states stay close to training ones (reset noise 0.005), stated in the report; Phase 8's "time in the expert region" always comes with the distance to the nearest dataset state.
- **Phase 8:** paired per-seed IQL minus BC; the `p = 0` reward-scale note.

## 10. Definition of Done
All tests and CI green; M1 to M8 pass for all 15 final mixtures and the development mixtures used later; M9 recorded; manifests, `summary.json` and `audit.json` committed with `git_dirty: false`; Explorer `content_sha256` values equal the laptop's; ROADMAP and config updated; a fresh reviewer has read the PR diff.
