# Pre-registration: verdict criteria for H1 to H3

**Status:** APPROVED by the user on 2026-11-21, before any final-grid data existed (Phase 6 spec Section 10, decision D7). It is committed before the grid commit, and the commit holding it must be an ancestor of the commit the grids run at. It is not changed after any grid result (Section 12 covers amendments).

**Data it applies to:** the Phase 7 main grid (`configs/grid_main.yaml`: BC and IQL at `tau = 0.7`, `p` in {0, 0.05, 0.2, 0.5, 1}, seeds 0 to 4, 50 runs) and the Phase 8 ablation (`configs/grid_ablation.yaml`: IQL at `tau` in {0.5, 0.9}, `p` in {0.05, 0.5}, seeds 0 to 4, 20 runs), 300k gradient steps each, at one commit on one hardware type. Nothing from the development runs (mixture seeds 100+) enters any verdict; Phase 8's optional extras (extra `p`, reweighting, CQL) are descriptive and enter no verdict either.

**Score:** a run's final score is the mean of its last 5 evaluations (steps 280k to 300k), each over 50 episodes on the final evaluation seeds 10000 to 10049, as normalized score `100 x (R - r_random) / (r_expert - r_random)` with `r_random = 17.03` and `r_expert = 3857.80` (ADR-6). Its final fall rate is the fraction of those 250 episodes that end in a fall. A cell is an (algorithm, `tau`, `p`) triple; it holds 5 runs, one per seed.

## 1. Pairing
Analysis is **paired by seed**. Seed `s` fixes the mixture at each `p` (the same file for BC and IQL, and for every `tau`), so every comparison (two methods at one `p`, or one method across `p` or `tau`) uses the per-seed differences `d_s = score_A(s) - score_B(s)`, `s` = 0 to 4.

## 2. Statistics reported
- **Primary:** the per-cell mean over the 5 seeds of the final score; for a comparison, the mean of the 5 paired differences. Every verdict rests on these and on the tests of Sections 4 to 7.
- **Required descriptive output, for every cell and every named comparison:** the **5 per-seed scores** (and per-seed differences), the **median**, the mean, the final fall rate per seed, and the plateau statistic (Section 9).
- **Secondary, descriptive only (no verdict rests on them):** the standard deviation, the interquartile mean and the probability of improvement (rliable, ROADMAP reference 14; its stratified bootstrap is used for these only), the raw return, and the learning curves.

## 3. Intervals
Two intervals are reported for every mean and mean difference; neither decides a verdict (the tests do):
- **Percentile bootstrap over seeds:** 95%, 10,000 resamples, bootstrap random seed 20261122; paired resampling (one draw of 5 seed indices with replacement serves both methods and every `p` in a contrast).
- **t-interval:** 95%, `mean +/- t(0.975, df = 4) x sd / sqrt(5)`, with `t = 2.776` and `sd` the sample standard deviation (n - 1).
- **Limitation, stated with every interval:** with 5 seeds the bootstrap has only **126 distinct resamples** (multisets of 5 drawn from 5), so percentile intervals are coarse and run **narrow**; the t-interval is wider and assumes roughly normal per-seed values. Both are indicative only.

## 4. Tests (exact, one per question)
All tests are the **exact paired sign-flip (randomization) test** on the 5 per-seed values `x_s`: the statistic is their mean; the null distribution is the mean over all `2^5 = 32` sign patterns of the `x_s`; the **one-sided** p-value is the fraction of patterns whose mean is at least the observed one. Level **0.05**. One-sided because each hypothesis names a direction, and because with 5 pairs a two-sided version has a smallest possible p of `2/32 = 0.0625` and could never reach 0.05; the smallest one-sided p is `1/32 = 0.031`, reached when all 5 values have the predicted sign (and none is 0).

**What the test decides, and its power:** at level 0.05 it rejects exactly when all 5 per-seed values have the predicted sign and none is 0 (one value against, or equal to 0, gives p >= 2/32), so each verdict is in effect a 5-of-5 sign rule, with an actual false-positive rate of 1/32 = 0.031 under a symmetric null. Power is low: if each seed favors the hypothesis with probability 0.9, the test rejects with probability 0.9^5 = 0.59. "Inconclusive" is therefore not evidence of no effect.

## 5. H1: "BC's score rises with `p`, possibly non-monotonically"
- **Test:** `x_s = score_BC(p = 1, s) - score_BC(p = 0, s)`, one-sided (greater).
- **Supported:** p <= 0.05. **Not supported (reversed):** the same test in the other direction (less) gives p <= 0.05. **Inconclusive:** otherwise.
- **Trend (descriptive):** per seed, the Spearman correlation of BC's score with `p` over the 5 grid values, reported as its mean over seeds. **Dip (descriptive):** any intermediate `p` whose BC mean is below both `score_BC(0)` and `score_BC(1)` (the proposal's averaging story), reported with its per-seed values.
- **Monotonicity, reported with the verdict, not part of it (the four adjacent comparisons are uncorrected and descriptive):** for each adjacent pair `p_i < p_j` of the grid, the sign-flip test of `score_BC(p_i, s) - score_BC(p_j, s)` (greater, that is a drop); BC is called **non-monotone** if some adjacent drop exceeds 5 points in mean with p <= 0.05, else monotone (as far as 5 seeds show). The hypothesis allows either.

## 6. H2: "IQL reaches near-expert performance (normalized score >= 90) at a much lower `p` than BC"
- **Near-expert:** a cell mean of at least **90** (ROADMAP H2, fixed 2026-11-14; see Section 10 for what was seen since).
- **Crossing `p` of a method:** the smallest grid `p` whose cell mean is at least 90 **and** for which every larger grid `p` is at least 90 too (a stable crossing; a method that reaches 90 at one `p` and falls below it at a larger `p` crosses only where it stays). A method with no such `p` does not cross.
- **Much lower:** IQL's crossing `p` is at least **one grid level** below BC's (levels 0 < 0.05 < 0.2 < 0.5 < 1), or IQL crosses and BC does not.
- **Test:** at IQL's crossing `p`, `x_s = score_IQL(p, s) - score_BC(p, s)`, one-sided (greater).
- **Supported:** IQL crosses, the "much lower" condition holds, and the test gives p <= 0.05. **Not supported:** IQL does not cross (whether or not BC does: H2 asserts that IQL reaches 90; user decision, 2026-11-21), or IQL's crossing `p` is at or above BC's (including both crossing at the same level, `p = 0` among them). **Inconclusive:** IQL crosses lower but the test gives p > 0.05 (including a BC lead at that `p`, reported as such).
- **Limitation:** grid levels are uneven in expert data. One level below `p = 1` is `p = 0.5` (half the transitions expert), one level below `p = 0.05` is `p = 0` (none), so "one level" means very different amounts of expert data at different points of the grid; the crossing `p` values themselves are always reported.

## 7. H3: "higher `tau` helps IQL more at `p = 0.05` than at `p = 0.5`"
- **Contrast:** `x_s = [score(tau 0.9, s) - score(tau 0.5, s)] at p = 0.05 minus [score(tau 0.9, s) - score(tau 0.5, s)] at p = 0.5`. The main grid's `tau = 0.7` cells do not enter `x_s`; they are reported beside it.
- **Test:** one-sided (greater) sign-flip test on `x_s`.
- **Supported:** p <= 0.05. **Not supported (reversed):** the test in the other direction gives p <= 0.05. **Inconclusive:** otherwise.
- **Wording:** with every verdict the two simple effects (`tau 0.9 - tau 0.5` at each `p`) are reported; if the effect at `p = 0.05` is not positive, a supported verdict is worded "the `tau` effect is larger at `p = 0.05`", not "higher `tau` helps".

## 8. Verdicts and multiple comparisons
- Each hypothesis gets exactly one verdict from its rule: supported, not supported or inconclusive, with the numbers that decided it.
- One primary test per hypothesis: three primary tests (H1 and H3 also run in the reverse direction, only to label a reversal), each at an actual level of 1/32 = 0.031, with no correction for multiplicity; each verdict's false-positive rate is per hypothesis, not for the family of three, and this is stated with the verdicts. Per-`p` comparisons beyond the named ones (for example IQL minus BC at every `p`) are descriptive and carry no verdict.

## 9. Run outcomes and the plateau statistic
**Run outcomes** (Phase 6 spec Section 2):
1. Only infrastructure failures lead to a resume or a rerun (apart from the single rerun of rule 5): a time-out, node failure, pre-emption, a stall, or a provenance failure found by `sweep --check`. A rerun is from scratch, same seed and commit.
2. A run that finishes is a result, whatever its score; it is never rerun or excluded. A **non-finite** final score (any of the 5 final evaluations non-finite; the code writes it to `final.json` as a string, so the run still finishes) is scored **0**, enters every statistic as 0, and is reported as such. A run is **flagged as collapsed** if its final score is below **10**, and separately **flagged as high-fall** if its final fall rate is at least **0.5** (it falls in at least half of its final episodes; an expert-like gait falling that often still scores about 94, so this flag marks instability, not failure; Phase 5's development IQL at `p = 0.2` scored about 17 with fall rates of 0.88 to 0.97, so it would carry the high-fall flag but not the collapsed one). Flagged runs keep their scores and enter every statistic.
3. A run is cancelled only on evidence of misconfiguration or infrastructure trouble (wrong config, wrong hardware, a stall), never for its score or curve.
4. A code bug found mid-grid means a new commit and a rerun of both grids in full; the trigger (what was seen, where and when) is recorded, and the superseded results are archived and reported in an appendix next to the rerun.
5. A run that stops with an error that is not an infrastructure failure is rerun once from scratch (same seed and commit); if the same error recurs, it is scored 0, flagged as failed, and reported.

**Plateau statistic** (reported for every cell, not acted on): over evaluations 51 to 60 (steps 255k to 300k), using only the evaluation seeds every evaluation shares (10000 to 10009: `score_mean` for evaluations 51 to 55, `sub_score_mean` for 56 to 60), the seed mean of [mean of evaluations 56 to 60 minus mean of evaluations 51 to 55]. 300k is the only budget (D6); a cell is described as still climbing if its statistic exceeds 2 points, a stated limitation. A run with a non-finite score is left out of this statistic and counted.

## 10. Results already seen (development data only)
These were seen before this document was written; none is final-grid data and none enters a verdict.

| Development result | Value | Bearing on these criteria |
|---|---|---|
| IQL (ours) at `p = 1`, seeds 100 to 104 | mean **92.5**, median 89.0; per seed 81.6, 89.0, 103.6, 83.9, 104.5 | mean just above H2's threshold of 90, median below it; 2 of 5 seeds at or above 90; seeds 100 and 101 on A100, 102 to 104 on T4, at different commits |
| CORL's IQL at `p = 1`, same seeds | mean 103.1 | |
| BC at `p = 1`, seeds 100, 101 | 104.6, 105.6 | |
| IQL (ours) and CORL at `p = 0.2`, seeds 100, 101 | 22.8, 10.7 and 12.7, 22.6; fall rates 0.88 to 0.97 | far below BC at the same `p` (bears directly on H2: development IQL does not cross at `p = 0.2`); also motivated the high-fall flag |
| BC at `p = 0.2`, seeds 100, 101 | 81.9, 82.1 | |
| BC at `p = 0` and `p = 1`, seeds 100, 101 (Phase 4, 100k steps) | 82.1, 81.9 and 104.4, 105.5 | BC's development `p = 1` minus `p = 0` is positive on both seeds (bears on H1) |
| IQL (ours) at `p = 0`, seeds 100, 101 | 82.2, 82.0; fall rates 0.012, 0.0 | below 90 (bears on H2's crossing) |

**Why 90 stays:** the threshold was written into ROADMAP H2 on 2026-11-14 (commit `e871495`), before any IQL run (the first, a timing smoke test, job 11958589, ran on 2026-11-17; the IQL code was first committed that day), and is kept unchanged now that a development mean near it has been seen; moving it in either direction after seeing 92.5 would be the post-hoc choice pre-registration exists to prevent. On its own terms 90 is principled: 100 is the expert dataset's mean, and on expert data a fall costs about 27 points (no-fall mean 107.8 vs. fell mean 80.3, Phase 5 spec D4), so 90 is expert-like hopping that falls (late) in up to about 65% of episodes (107.8 - 0.65 x 27.4 = 90; the expert data itself falls in 28.4% and scores 100), and lies above every simple-data level (the simple dataset scores 42.7, its band tops out at 82.3, and BC on it scores about 82).

## 11. Deviations from the proposal (`proposal.tex`), all decided before any grid result
| Proposal | As run | Decided |
|---|---|---|
| Score `100 (R - R_simple) / (R_expert - R_simple)` | 0 = random-policy mean, 100 = expert dataset mean | ADR-6, 2026-11-15 |
| Sanity: BC near 0 at `p = 0`, near 100 at `p = 1`; IQL near 100 at `p = 1` | Phase 4 bands and plumbing checks; Phase 5 V1 (IQL within 10 of BC at `p = 1`), 90 reported | Phase 4 D3, D4; Phase 5 D3 |
| IQL update order unspecified | CORL's (pre-update V for the actor and the Q target) | Phase 5 D1 |
| Validation at `p = 1` vs. CORL's curves | V0 to V4 on 14 runs, CORL through our pipeline; V1 failed on seeds 100 and 101 (recorded), the seed 102 to 104 follow-up gave "proceed"; both reported beside H2 | Phase 5 D2, D4 to D7 |
| H2 "near-expert" unquantified | cell mean >= 90 | ROADMAP 1.2, 2026-11-14 |
| Main figure: mean +/- std over 3 seeds | mean with both intervals over 5 seeds; std in the table | Phase 6 D2, this document |
| 3 seeds | **5 seeds**, a superset | Phase 6 D2, 2026-11-21 |
| Ablation 12 runs (3 seeds) | **20 runs** (5 seeds, matching) | Phase 6 D5 |
| 300k steps, rerun at 500k if still climbing (ROADMAP) | **300k only** | Phase 6 D6 |
| Statistics: mean and std over seeds | exact one-sided sign-flip tests, bootstrap and t-intervals, medians and per-seed scores (this document) | Phase 6 D7 |

## 12. Amendments
Any change after the grid commit is dated, listed as a deviation, and its effect shown: the verdict under the original rule is reported next to the amended one.
