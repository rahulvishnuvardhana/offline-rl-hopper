# Spec: Phase 4, BC and the evaluation harness

**Status:** v2, after review by two fresh reviewers (engineering, RL): 3 blockers, 12 majors and 13 minors addressed; user approved D1 to D4 on 2026-11-16, with the D4 reporting rule in Section 8 | **Branch:** `phase-4-bc-eval` | **Roadmap:** Phase 4 | **Depends on:** Phase 3 (merged `63bf8fa`)

## 1. Goal
A working BC baseline and the shared plumbing every later run uses: networks, the shared policy loss, the evaluator, the training loop with logging, checkpoints and resume, and per-run provenance. BC is the cheapest algorithm, so if it fails, the bug is most likely in the plumbing (data, normalization, evaluation), not in value learning.

## 2. Scope, seeds and randomness
- **In:** `offrl/nets.py`, `offrl/algos/{base,losses,bc}.py`, `offrl/evaluate.py`, `offrl/runlog.py` (CSV + TensorBoard; not `logging.py`, which would shadow the standard library), `offrl/train.py`, `scripts/bc_sanity.py`, tests. Small changes to existing code: `normalize.normalizer_from_arrays()` and `ReplayBuffer.normalizer` (the evaluator and the buffer share one normalizer object), `rollout.eval_seeds(eval_cfg, n=None)`, a `requires_data` pytest marker.
- **Out:** IQL (Phase 5), the Slurm grid and `sweep.py` (Phase 6). `nets.py` already contains `TwinQ` and `ValueNet` with shape tests.
- **Seeds:** sanity runs use **development** mixtures (seeds 100 and 101) and **development** evaluation seeds (`eval.seed_offset: 20000`). Final mixture seeds 0 to 2 and evaluation seeds 10000+ are not touched.
- **Randomness streams (common random numbers, N4):** run seed `s` gives mixture seed `s`. `np.random.SeedSequence(s).spawn(3)` derives three independent seeds: network initialization (global torch/NumPy/Python via `seed_everything`), the buffer's index generator, and the diagnostic action sampler (Section 5). Without this, `torch.manual_seed(s)` and the buffer's `Generator().manual_seed(s)` would produce the same uniform stream. **The policy is constructed first**, immediately after seeding, so BC and IQL with the same seed start from identical policy weights and draw identical batches (one batch per step for all losses, as in CORL and official IQL). A test asserts both.

## 3. Networks (`offrl/nets.py`)
- `MLP(in, out, hidden=(256, 256))`: Linear-ReLU-Linear-ReLU-Linear, **PyTorch default init**, as CORL (the Phase 5 reference). Official IQL uses orthogonal init; recorded as a known difference.
- `GaussianPolicy(obs_dim=11, act_dim=3)`, as CORL's IQL policy: `mean = tanh(MLP(obs))`; `log_std` is a state-independent `nn.Parameter` (init 0), **clamped to `[-20, 2]` in `forward`** (the raw parameter is not clamped; official IQL uses -10 as the lower bound, irrelevant in practice). `dist(obs) = Normal(mean, exp(clamped log_std))`, log-prob summed over the 3 action dimensions, **no tanh on samples**, so dataset actions at `|a| = 1` have finite log-prob. `act(obs)` returns the mean, in `[-1, 1]`. No dropout (official IQL uses dropout only for Kitchen and Adroit).
- `TwinQ(obs, act) -> (q1, q2)` and `ValueNet(obs) -> v`, each `(B, 1)`.

## 4. Losses and the BC trainer (ADR-2: one code path)
- `losses.policy_loss(policy, obs, act, weights)`: `log_prob` has shape `(B,)`; `weights` is accepted as `(B,)` or `(B, 1)`, squeezed, **shape-asserted to `(B,)`** and detached; returns `mean(weights * -log_prob)`. Without the assert, `(B, 1) * (B,)` broadcasts to `(B, B)`: constant BC weights hide it (same mean), but IQL's per-sample weights would train on a wrong loss. This is the only policy loss in the project; IQL calls it with `exp(beta * A)` clipped at 100.
- `BCTrainer.update(batch)` = `policy_loss(weights = ones)`, i.e. IQL's loss at `beta = 0` exactly. It returns **device tensors** (`policy_loss`, effective `log_std` mean, actor lr); the loop accumulates them on the device and calls `.item()` only at log time (no GPU sync per step, for Phase 6).
- **Optimizer:** Adam, lr 3e-4, no weight decay. **Actor LR: cosine decay to 0 over `train.steps`, for BC too (D2).** The flag moves to `base.yaml` as `train.actor_cosine_lr: true` and is deleted from `algo/iql.yaml`.
- `Trainer` protocol (`algos/base.py`): `update`, `act`, `state_dict`, `load_state_dict`. `act` takes **normalized** observations; the evaluator owns normalization.

## 5. Evaluator (`offrl/evaluate.py`)
- **Environment:** `minari.load_dataset(expert_id).recover_environment()`, created once per run.
- **Policy function:** `obs -> trainer.act(buffer.normalizer.transform(obs))`, under `torch.no_grad()`, float32. The normalizer is the buffer's own object (the mixture's statistics), not a second copy.
- **Seeds:** only `rollout.eval_seeds(cfg["eval"], n)`, with `n = eval.episodes` (10) or `eval.final_episodes` (50); episodes via `rollout.run_policy`.
- **Per evaluation:** `step`, `n_episodes`, `return_mean`, `return_std`, `score_mean` (ADR-6), `fall_rate`, `length_mean`, plus the **10-seed subset** values for every evaluation (so the curve has no variance jump at the final evaluations). Every episode's `(step, seed, return, length, fell)` goes to `eval_episodes.csv`. Wall-clock times go to `timing.csv`, not to the CSVs compared bitwise.
- **Schedule:** every 5,000 steps, 10 episodes. Pre-flight requires `steps % eval_every == 0`, `checkpoint_every % log_every == 0` and `steps % checkpoint_every == 0`.
- **Final score (pre-registered, D1):** mean of the last 5 evaluations (steps `T - 20k` to `T`), each over **50 episodes** (seeds `offset` to `offset + 49`). Wording: this averages 5 correlated checkpoints on the same 50 starting states. It reduces evaluation noise and resolves all-or-nothing falls, but it is not 250 independent samples. `final.json` holds score, raw return, fall rate, the per-evaluation values and their 10-seed subsets.
- **Diagnostic, descriptive only (not part of any score):** at the end of a sanity run, one 50-episode evaluation with **sampled** actions, using a dedicated generator (Section 2), so it is reproducible and does not disturb any other stream. With a state-independent std it cannot by itself show whether the data actions were noisy. That question is answered in Section 8 instead.

## 6. Training loop (`offrl/train.py`)
Entry points: `run(cfg) -> Path` (imported by Phase 6's `sweep.py`) and `main(argv)` for `python -m offrl.train configs/algo/bc.yaml [key=value ...]`.
1. Load `base.yaml`, the algo file (and `final.yaml` only in Phases 7 and 8), apply overrides. **Schema check:** every key in every YAML file must exist in `base.yaml` or be declared by the algo file's own section (today a typo in a YAML file is silently accepted).
2. **Pre-flight audit (refuses to start; results written to `meta.json`):**
   - the mixture file exists; `content_sha256` recomputed on the loaded arrays equals the committed manifest `report/mixtures/{name}.json` (so its manifest must be committed before the run);
   - the manifest's `p_target` equals `data.p_expert` and its `seed` equals `seed`;
   - `data.normalize_states` is true and `data.state_std_eps` equals the mixture's `state_std_eps` (otherwise both keys are dead config);
   - **seed-set disjointness:** the actual evaluation seeds (`episodes` and `final_episodes`) are disjoint from `COLLECTION_SEEDS` (via the leakage helpers); for a development mixture (`seed >= 100`) they are disjoint from `range(10000, 10050)`, and for a final mixture (`seed < 100`) they equal it;
   - the step-alignment rules in Section 5.
3. **Run folder** `{train.out_dir}/{run_name}/`, `run_name` defaulting to `{algo}_p{p:g}_s{seed}` plus an optional `run_suffix` (Phase 8 ablations, e.g. `_tau0.9`). Contents: `config.yaml`, `meta.json` (runmeta + mixture hash, `reward_scale`, `beta_eff`, `torch.get_num_threads()`, and a list of resume records), `metrics.csv`, `eval.csv`, `eval_episodes.csv`, `timing.csv`, `ckpt/latest.pt`, `final.json`, TensorBoard events (CSV is the source of truth).
4. **Loop** `step = 1 .. T`: one batch, `trainer.update`; every `log_every`, log the mean since the last log; evaluate every `eval_every`; checkpoint every `checkpoint_every`; at `T` write `final.json`. CSV rows are flushed as they are written.
5. **Checkpoint:** written to `latest.pt.tmp`, `fsync`, then `os.replace`. Contents: step, trainer state (networks, optimizer, scheduler), the buffer generator state, the torch CPU (and CUDA) RNG state, and NumPy and Python RNG states **as plain lists and ints**, so `torch.load(weights_only=True)` (the torch 2.11 default) accepts them. A test round-trips a checkpoint with `weights_only=True`.
6. **Resume:** if `ckpt/latest.pt` exists, **refuse unless** the saved `config.yaml` equals the newly resolved config exactly and `meta.json`'s `git_commit` equals HEAD. Then restore everything, truncate the CSVs to `step <= checkpoint step` (tolerating a half-written last line), reopen TensorBoard with `purge_step`, append a resume record to `meta.json`, and continue. If the checkpoint is at `T` but `final.json` is missing, write it from the logged evaluations. A run with `final.json` exits immediately (Phase 6 resubmission).
7. **Extending a run is not resume:** the cosine schedule depends on `T`, so a longer budget means a fresh run (ROADMAP Phase 7 wording updated: "rerun at 500k", not "extend").
8. **Device and threads:** `train.device: auto` (CUDA if available, else CPU). New key `train.num_threads` (default `null` = the torch default); Phase 6's sweep sets it to `cpus_per_task // runs_per_bundle` so bundled runs do not oversubscribe a node.
9. **Test hook:** `run(cfg, stop_at=k)` returns after step `k` (any step, not only a checkpoint), as a kill would; rows logged after the last checkpoint must then be discarded on resume. Used by R1 and the real-data resume demonstration. The sampled-action diagnostic runs in `scripts/bc_sanity.py` after training, so its generator is not part of the checkpoint.

## 7. Checks (results in `report/phase4/bc_sanity.json`; `requires_data` tests auto-skip in CI)
| # | Check | Where |
|---|---|---|
| B1 | `policy_loss` equals the closed form `mean_i w_i * sum_d [0.5 z^2 + log sigma + 0.5 log 2 pi]` computed **per sample in a float64 Python loop**, with **random non-constant weights** (atol 1e-5); a `(B, B)` broadcast is impossible (shape asserts); weights are detached | CI |
| B2 | `BCTrainer.update` equals `policy_loss(weights = 1)` on the same batch and parameters | CI |
| B3 | `act` is deterministic, equals `tanh(mu)`, lies in `[-1, 1]`; the **effective** `log_std` stays in `[-20, 2]` when the raw parameter is pushed outside | CI |
| B4 | Overfit: synthetic actions are a fixed function of obs plus noise of std 0.05; after 2,000 steps the BC loss is at least 1 nat below the best constant Gaussian's | CI |
| E1 | Evaluation reproducibility: the same policy evaluated twice on `gym.make("Hopper-v5")` gives identical per-episode returns; `eval_seeds(cfg, n)` for every `n` used is disjoint from the collection seeds, and development and final sets are disjoint | CI |
| E2 | The evaluator's observation transform is the buffer's (same object), and transforms equal the stored buffer rows bitwise | CI |
| E3 | **Evaluator vs. data, staged:** replay expert and simple episodes 0 to 2 from their collection seeds (`123 + i`) with their dataset actions, via `run_policy` directly (deliberately not the evaluator, which never sees collection seeds). Hard: the reset observation equals `obs[0]` to 1e-8, the first 100 steps match as M9 (2e-10), and length and fall flag are exact. The full-episode return error is measured first and its tolerance pre-registered from that measurement before the check is enforced; if the installed MuJoCo version differs from the dataset's, the full-episode part is reported, not enforced | Local |
| E4 | `score(r_random) = 0` and `score(r_expert) = 100` exactly | CI |
| R1 | **Exact resume:** CPU, synthetic mixture, `steps = 600`, checkpoint every 200. A run stopped by `stop_at = 350` (between checkpoints, after an evaluation that must be discarded) and then resumed is bitwise identical to an uninterrupted run (parameters, optimizer, scheduler, `metrics.csv`, `eval.csv`). Holds on one machine with a fixed thread count; stated as such | CI |
| R2 | Same seed twice gives the same first 100 losses (N1); different seeds differ; BC and IQL-shaped trainers with the same seed have identical initial policy weights and batch indices | CI |
| R3 | Pre-flight refuses: wrong hash; `p` or seed mismatch; a development mixture with seeds overlapping 10000 to 10049 (including `seed_offset = 10001`); a final mixture without them; eps mismatch; misaligned steps; resume with a changed config or commit | CI |
| R4 | Checkpoint round-trips under `torch.load(weights_only=True)` | CI |
| R5 | Provenance of the sanity runs: `git_dirty` false, commit recorded, mixture hash equals the committed manifest | Local |

## 8. Sanity runs (DoD) and compute
- **Build:** `python scripts/build_mixtures.py --p 0 1 --seeds 100 101`; commit `report/mixtures/` (manifests, summary, audit) before training.
- **Runs:** BC at `p = 0` and `p = 1`, mixture seeds 100 and 101, 100k steps, evaluation seeds 20000+. Four runs.
- **Compute (user decision, 2026-11-16: training runs on Explorer GPUs, never the laptop):** `slurm/bc_sanity.sbatch`, one T4 on `gpu-short` (2 h cap), 4 CPUs for evaluation rollouts. It runs the test suite on the GPU node (so the CUDA-only tests run), rebuilds the development mixtures into `/scratch` (the committed manifests are rewritten identically and restored), then `scripts/bc_sanity.py --rerun-misses` with `train.device=cuda`. `--set` accepts only device, path and thread overrides, never hyperparameters. Outputs go to `logs/phase4_<jobid>/` and are copied to the laptop and committed there. The laptop dry run measured 8 to 12 ms per update on CPU; the GPU time is measured and reported as Phase 6 input. Resume on a GPU is bitwise only with deterministic cuBLAS (`CUBLAS_WORKSPACE_CONFIG=:4096:8`, set in the job); R1 in CI covers the CPU case.
- **Plumbing checks (pass/fail):**
  1. **No-fall return match** (added 2026-11-16, Section 8.1), at both `p = 0` and `p = 1`: the mean return of the policy's no-fall episodes over the 5 final evaluations is within **10%** of the source dataset's no-fall mean (expert 4157, simple 3179), judged on **at least 10** no-fall episodes; fewer is a FAIL (cannot be evaluated). For a cell with a 300k rerun, the rerun is judged.
  2. **Held-out action fit** on dataset transitions **not in the mixture** (about 99k unused expert transitions at `p = 1` and 99k unused simple ones at `p = 0`, taken from the Phase 3 picks; no training data is removed, so the no-validation-split decision stands): action MSE and NLL of the mean action vs. the training rows; expert held-out R^2 of the mean action pre-registered at >= 0.8 (D3).
  3. Bands (D4): `p = 1` in `[80.3, 113.5]` and `p = 0` in `[22.5, 85.4]`. The lower bounds are Phase 2's fell-mean scores. The upper bounds are the datasets' **maximum** episode scores, because deterministic BC can legitimately beat the no-fall mean.
- **Reported, not gated:** `score(p = 1)` vs. `score(p = 0)` per seed (Section 8.1).
- **On a band miss:** rerun that cell at 300k steps (BC's compounding errors can raise the fall rate without any bug). The original 100k run is **kept and reported alongside** the rerun, never replaced (user decision, D4): `bc_sanity.json` lists both, with the miss marked. A miss at 300k is investigated as a plumbing bug, not tuned away.
- **Were the data actions noisy?** Compare (a) the learned std with the held-out residual std of the mean action, and (b) deterministic BC's no-fall return with the dataset's no-fall mean (4157 expert). The sampled-action evaluation is shown alongside but is descriptive only.
### 8.1 Changes after the 25k dry run (2026-11-16, before the official runs; user decision)
A development dry run (seeds 100 and 101, **25k** steps, so the cosine schedule reached 0 at 25k; uncommitted code, so not evidence for the DoD) gave:

| Run | Score | Fall rate | No-fall return (n of 250) | Dataset no-fall mean | Held-out R^2 | Learned std vs. held-out residual std |
|---|---|---|---|---|---|---|
| p = 0, s100 | 82.2 | 0.01 | 3180 (248), +0.05% | 3179 | 0.833 | 0.174/0.236/0.230 vs. 0.175/0.236/0.230 |
| p = 0, s101 | 82.3 | 0.00 | 3188 (249), +0.3% | 3179 | 0.835 | 0.174/0.236/0.230 vs. 0.173/0.235/0.229 |
| p = 1, s100 | 58.7 | 0.83 | 3934 (42), -5.4% | 4157 | 0.871 | 0.298/0.216/0.173 vs. 0.298/0.215/0.173 |
| p = 1, s101 | 55.9 | 0.94 | 3783 (15), -9.0% | 4157 | 0.872 | 0.300/0.216/0.173 vs. 0.299/0.214/0.171 |

Training and held-out R^2 agree (0.87 vs. 0.87 at `p = 1`), so BC does not overfit; the learned std equals the residual std, so the data actions carry noise of that size, and the deterministic mean action removes it. At `p = 0` this removes almost all falls (dataset 66%, BC 1%), so BC at `p = 0` scores about 82, not the dataset's 42.7.

1. **`score(p = 1) > score(p = 0)` is demoted from a gate to a reported result.** Reason: it tested a hypothesis ("expert data makes BC better"), not the plumbing. The dry run shows a correct pipeline can fail it, because the simple policy's main weakness is action noise, which deterministic BC removes. Plumbing is covered by the bands, the held-out fit, the no-fall return match and the resume test.
2. **No-fall return match added, at both `p`, tolerance fixed here before the official runs:** within 10% on at least 10 no-fall episodes. A broken normalizer, action path or evaluator changes the gait itself, so the return of the episodes that do not fall moves far from the data; compounding error mainly changes how often the policy falls. 10% admits the 25k-step deviations above (+0.05% to -9.0%), which come from an under-trained, schedule-truncated policy, and the expert dataset's own no-fall spread is about 1.3% (std 53), the simple dataset's 0.6% (std 19). The 10-episode floor makes a policy that almost always falls a FAIL instead of a vacuous pass.
3. **Resume check tightened:** the demonstration must record exactly one resume. In the dry run the only checkpoint was at 25k, so the stopped run restarted from 0 and the bitwise match proved nothing; at 100k the stop at 37.5k resumes from the 25k checkpoint.
4. **Research note for the report:** the headroom between BC at `p = 0` and the expert is smaller than the dataset means suggest (about 82 vs. 100, not 43 vs. 100), which bears on H2 and H3.

- **Report notes:** the sanity cosine schedule spans 100k steps, not the grid's 300k; PyTorch default init (CORL) vs. orthogonal (official IQL).
- **Outputs (committed):** `report/phase4/bc_sanity.json` (scores, curves, checks, timings, provenance) and `report/figures/bc_sanity_curves.png` (score vs. step per run, bands shaded, dataset means as lines; fall rate in a second panel, not a second axis). `scripts/bc_sanity.py` exits 1 on any failed check.
- **Resume on real data:** a copy of the `p = 1`, seed 100 run (`run_suffix: _resume_demo`) is stopped at 37.5% of its steps (between checkpoints) and resumed; its `metrics.csv`, `eval.csv` and `eval_episodes.csv` must equal the uninterrupted run bitwise.

## 9. ROADMAP, config and Phase 5 notes in this PR
- `base.yaml`: `train.actor_cosine_lr: true`, `train.num_threads: null`, `train.run_suffix: ""`, `eval.final_episodes: 50`; remove `algo.actor_cosine_lr` from `iql.yaml`.
- ROADMAP: Phase 4 final-score definition and bands; Phase 7 "rerun at 500k" wording; Section 3.2 file names (`evaluate.py`, `runlog.py`, `algos/losses.py`).
- **Phase 5 heads-up:** CORL computes the actor advantage with the pre-update V; official IQL with the updated V. Phase 5 follows CORL, the reference used for its overlay.

## 10. Definition of Done
All tests and CI green; B1 to B4, E1 to E4 and R1 to R5 pass; the four sanity runs meet Section 8's plumbing checks (bands, held-out fit, no-fall return match, resume), `p = 1` vs. `p = 0` reported with fall rates reported; the noise analysis and the real-data resume recorded; `report/phase4/` and the figure committed with `git_dirty: false`; a fresh reviewer has read the PR diff.

## 11. Decisions (approved by the user, 2026-11-16)
- **D1, final evaluation size:** **Approved:** the last 5 evaluations use 50 episodes each.
- **D2, BC actor LR:** **Approved:** the same cosine schedule as IQL, so BC is exactly IQL with `beta = 0`.
- **D3, held-out action fit:** **Approved:** use the dataset transitions that are not in the mixture as a held-out set for a plumbing check, with expert R^2 >= 0.8.
- **D4, sanity bands:** **Approved:** widen the upper bounds to the dataset maximum scores (113.5, 85.4), keep the lower bounds, add the 300k rerun rule, and keep a missed run, reported next to its 300k rerun.
