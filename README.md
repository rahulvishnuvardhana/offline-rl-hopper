# How Much Expert Data Does Offline RL Need?

Behavior Cloning vs. Implicit Q-Learning on mixed-quality Minari Hopper datasets (CS 5180 project).
Plan and architecture: [ROADMAP.md](ROADMAP.md).
Detailed overview: [One Leg, Two Teachers](docs/story.md), the project in plain language with its charts, demos and verdicts (an [interactive version](https://claude.ai/artifact/TT1ATHyBdmQUFu5iGy5gZr) adds 3D replays of every run).

![BC (left) hops for all 1,000 steps; IQL (right) falls after 27](report/demo/bc_vs_iql_p0.05_seed10000.webp)

*With 5% expert data (`p` = 0.05), BC (left) hops for the full episode while IQL (right) falls after 27 steps. Each is its cell's median-scoring run on evaluation seed 10000, the pre-set demo rule, and both match their logged episodes exactly. [Full-quality MP4](report/demo/bc_vs_iql_p0.05_seed10000.mp4).*

## Results

We mix expert and low-quality Hopper data at expert fractions `p` in {0, 0.05, 0.2, 0.5, 1} and train BC and IQL on each: 5 seeds, 300k steps, one commit (`ea11e14`), one CPU type, 70 runs, with hypotheses and tests [pre-registered](docs/preregistration.md) before any grid result.

![(a) Final score vs. expert fraction, BC and IQL](report/figures/score_vs_p_a.png)

![(b) IQL tau ablation at p = 0.05 and 0.5](report/figures/score_vs_p_b.png)

Dots are seeds, the marker the seed mean, the thin whisker the 95% t-interval and the thick bar the 95% bootstrap interval; the dashed line is near-expert (90). The paper uses both panels as one figure ([PDF](report/figures/score_vs_p.pdf)).

| Hypothesis | Test (exact one-sided sign-flip, 5 seeds) | Verdict |
|---|---|---|
| H1: BC's score rises with `p` | BC(`p` = 1) minus BC(`p` = 0) = +23.2, all 5 seeds, P = 0.031 | **Supported** |
| H2: IQL reaches near-expert (90) at a much lower `p` than BC | both cross 90 only at `p` = 1; IQL minus BC there = -5.7 | **Not supported** |
| H3: higher `tau` helps IQL more at `p` = 0.05 than at 0.5 | contrast -35.0, P = 0.969 | **Inconclusive** (observed the other way) |

BC stays at about 82 from `p` = 0 to 0.5 and reaches 105.4 at `p` = 1. IQL collapses when expert data is scarce: at `p` = 0.05 it scores 6.7 and falls in 98% of final episodes, and its learning curves show the failure developing during training. No correction across H1 to H3; the per-`p` "BC leads" tests are exploratory.

![Learning curves](report/figures/learning_curves.png)

**A second example, illustrative and not typical.** The same `p` = 0.05 mixture and evaluation seed, with a different IQL run (seed 2) chosen by hand to show a late fall: BC (its median run, as above) completes all 1,000 steps while IQL hops for about 600 steps and then falls. The median IQL run falls after 27 steps (above), so this is not what IQL usually does at `p` = 0.05. It is also a re-render, not an exact replay: Explorer's logged episode falls at step 613, while this laptop render falls at step 622, because floating-point differences between CPU types grow over a long episode (BC's side matches its log exactly). Made with [scripts/render_showcase_tolerant.py](scripts/render_showcase_tolerant.py), which accepts the same fall outcome within 15 steps of the log.

![Illustrative, not typical: BC completes 1,000 steps; IQL seed 2 falls at step 622 (logged 613)](report/demo/bc_vs_iql_p0.05_seed10000_fall622.webp)

[Full-quality MP4](report/demo/bc_vs_iql_p0.05_seed10000_fall622.mp4)

- **Report:** [report/paper/main.pdf](report/paper/main.pdf) (NeurIPS 2026 style, preprint option)
- **Poster:** [report/poster/poster.pdf](report/poster/poster.pdf)
- **Demo:** [BC vs. IQL at `p` = 0.05](report/demo/bc_vs_iql_p0.05_seed10000.mp4), each cell's median-scoring run on evaluation seed 10000 (BC hops for all 1,000 steps; IQL falls after 27)
- **Numbers:** [report/phase7/prereg_analysis.json](report/phase7/prereg_analysis.json); compute log [report/compute.md](report/compute.md) (70 runs, 3.1 h wall clock, 195 CPU-hours)

### Reproduce the results
On Explorer, from a clean clone at the grid commit (procedure in [docs/specs/phase-6-compute.md](docs/specs/phase-6-compute.md) Sections 7 to 9, including the login-node workaround; the exact submission script run is [report/phase7/submission/grid_submit.sh](report/phase7/submission/grid_submit.sh)). Outline:
```bash
sbatch --export=ALL,EXPECTED_COMMIT=<sha>,GRIDS="main ablation" slurm/tests_gate.sbatch   # tests, checkout, sweep --init
bash report/phase7/submission/grid_submit.sh                                              # --sbatch-args, then sbatch slurm/grid.sbatch per array
```
The run directories are not in this repository (`runs/` is ignored): they are archived on Explorer in `~/archive/phase6_grid_ea11e14d4ab7` with a laptop copy, both checked against its `SHA256SUMS`. From that archive:
```bash
python scripts/prereg_analysis.py --runs <archive>/runs --commit ea11e14d4ab7d0c6d2dc5eaabff6b41049fe17c3
python scripts/learning_curves.py --runs <archive>/runs
python -m offrl.analysis.plots          # report/figures/score_vs_p.* and learning_curves.*
```
Without the archive, `python -m offrl.analysis.plots` still regenerates both figures from the committed JSON in `report/phase7/`.

## Setup

### Laptop (Windows, CPU)
```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.11.0 "minari[all]==0.5.4" "gymnasium[mujoco]==1.4.0" mujoco==3.2.3 numpy==2.4.6 pandas==3.1.1 matplotlib==3.11.2 tensorboard==2.21.0 rliable==1.2.0 pytest==9.2.0 ruff==0.17.5
$env:MINARI_DATASETS_PATH = "$PWD\data\minari"
.venv\Scripts\python.exe scripts\smoke.py --datasets
```

### Explorer (Northeastern HPC)
```bash
srun --partition=short --cpus-per-task=4 --mem=8G --time=01:00:00 --pty bash
module load miniconda3/24.11.1
conda env create --prefix /scratch/$USER/envs/offrl -f environment.yml
exit
sbatch slurm/smoke.sbatch
```

## Training and evaluation

```bash
python -m offrl.train configs/algo/bc.yaml seed=100 data.p_expert=1.0 train.steps=100000
python -m offrl.train configs/algo/iql.yaml seed=100 data.p_expert=1.0       # IQL (Phase 5)
python scripts/bc_sanity.py          # Phase 4 sanity runs and checks (on Explorer: sbatch slurm/bc_sanity.sbatch)
```
A run writes `config.yaml`, `meta.json` (commit, packages, GPU, mixture hash), CSV logs, TensorBoard events, a resumable checkpoint and `final.json` to `runs/{algo}_p{p}_s{seed}/`. Pre-flight refuses a mixture whose hash differs from its committed manifest and evaluation seeds that overlap data collection or the reserved final-grid seeds.

## Development results (Phases 4 and 5)

**Phase 4, BC sanity (development data, 100k steps, Tesla T4):** all 15 plumbing checks pass ([report](report/phase4/bc_sanity.json), [spec](docs/specs/phase-4-bc-eval.md)).

| Expert fraction `p` | Seed 100 | Seed 101 | Expected band |
|---|---|---|---|
| 0 (simple data only) | 82.1 | 81.9 | 22.5 to 85.4 |
| 1 (expert data only) | 104.4 | 105.5 | 80.3 to 113.5 |

![BC sanity curves](report/figures/bc_sanity_curves.png)

Normalized score: 0 = random policy, 100 = expert dataset mean. Deterministic BC on the simple data scores about 82, well above that dataset's 42.7, because the mean action removes most of its falls.

**Phase 5, IQL validation (development data, 300k steps):** V0, V2, V3 and V4 pass; V1 (IQL within 10 points of BC at `p = 1`) fails and is recorded, 37 of 38 checks ([report](report/phase5/iql_validation.json), [job log](report/phase5/iql_validation_job.log), [spec](docs/specs/phase-5-iql.md) Section 8). The 14 runs trained at `1b0ccbe` and were first analysed with that commit's code; a reanalysis with the checks added in later review (job 12053724, [report](report/phase5/iql_validation_reanalysis.json)) gives the same outcomes and identical numbers. "Ours" is this repository's IQL; "CORL" is CORL's IQL, vendored and run through the same pipeline as the reference.

| `p = 1`, final score | Seed 100 | Seed 101 | Seed 102 | Seed 103 | Seed 104 |
|---|---|---|---|---|---|
| Ours | 81.6 | 89.0 | 103.6 | 83.9 | 104.5 |
| CORL | 99.5 | 104.5 | 104.4 | 103.8 | 103.3 |
| BC | 104.6 | 105.6 | not run | not run | not run |

Seeds 100 and 101 are the registered V1 runs (A100, job array 11981886, analysed by 11981887, which exits 1 on the V1 miss); seeds 102 to 104 are the follow-up the spec fixed in advance for a V1 miss (T4, job 12034721, scored by 12048550, [report](report/phase5/seed_followup.json)). The V1 bar is BC's two-seed mean minus 10, 95.1: ours falls short on 3 of 5 seeds, CORL on 0 of 5. The follow-up rule investigates only if ours trails CORL by more than 5 points on at least 2 of seeds 102 to 104; it does on 1 (seed 103), so **Phase 5 proceeds** and V1's miss is recorded as seed variance. An exploratory same-weights check run afterwards at the user's request (float64, laptop CPU, from seed 103's initial weights and from CORL's seed 103 state at 200k steps) found ours and CORL with identical policy weights and actions at every 50-step check, while a single rounding-sized nudge grew into action differences up to 1.12 within 5,000 steps ([log](report/phase5/same_weights_EXPLORATORY_partial.log)): consistent with seed variance; implementation difference ruled out at the arithmetic level. Five seeds cannot show whether falling short is equally likely for both implementations.

| Seeds 100 and 101, final score | Ours | CORL | BC |
|---|---|---|---|
| `p = 0.2` | 22.8, 10.7 | 12.7, 22.6 | 81.9, 82.1 |
| `p = 0` (simple data only) | 82.2, 82.0 | not run | 82.1, 81.9 (Phase 4, 100k steps) |

At `p = 0.2` both IQL implementations agree (V2: 16.7 vs. 17.6) and fall in 88% to 97% of final evaluation episodes, while BC never falls; both score far below BC. These are development runs that check the plumbing, not the experiment; the comparison across `p` is Phase 6 and later.

![IQL validation curves](report/figures/iql_validation_curves.png)

## Cluster notes (checked 2026-11-13)

- **Login:** `ssh explorer` (alias for `login.explorer.northeastern.edu`, SSH key auth).
- **Approval:** HPC use for this project approved by the professor whose course allocation provides access, on condition of responsible use (ROADMAP 4.2.1).
- **Slurm account:** `we.shi` (default, no partition restriction, QoS `normal`).
- **GPU partitions:** `gpu` (8 h max), `gpu-short` and `gpu-interactive` (2 h max), `sharing` (1 h max). CPU: `short` (2 days max).
- **GPU gres strings on `gpu` / `gpu-short`:** `t4`, `v100-pcie`, `v100-sxm2`, `a100`, `h200`. This project uses `t4` or `v100-*` only.
- **Modules:** `miniconda3/24.11.1` (also `25.9.1`), `anaconda3/2024.06`, `cuda/12.1.1`, `12.3.0`, `12.8.0`, `13.2.0`.
- **Storage:** code in `~/offline-rl-hopper`; env at `/scratch/$USER/envs/offrl`; datasets at `/scratch/$USER/minari` via `MINARI_DATASETS_PATH`. Scratch is purged monthly: copy results out.

## Verified versions

| Package | Version |
|---|---|
| Python | 3.11.18 |
| torch | 2.11.0 (cu128 on Explorer, CPU on laptop) |
| minari | 0.5.4 |
| gymnasium | 1.4.0 |
| mujoco | 3.2.3 |

Datasets verified: `mujoco/hopper/expert-v0` 999,164 steps / 1,086 episodes; `mujoco/hopper/simple-v0` 999,206 steps / 1,952 episodes; both recover `Hopper-v5` with a 1,000-step limit.

## Credits

`src/offrl/_vendor/corl_iql.py` holds selected, unmodified lines of [CORL](https://github.com/corl-team/CORL)'s IQL (Apache-2.0; license and line ranges in `src/offrl/_vendor/`). It is the reference our IQL is tested against (`tests/test_corl.py`) and is run as `configs/algo/corl_iql.yaml`.
