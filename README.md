# How Much Expert Data Does Offline RL Need?

Behavior Cloning vs. Implicit Q-Learning on mixed-quality Minari Hopper datasets (CS 5180 project).
Plan and architecture: [ROADMAP.md](ROADMAP.md).

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
python scripts/bc_sanity.py          # Phase 4 sanity runs and checks (on Explorer: sbatch slurm/bc_sanity.sbatch)
```
A run writes `config.yaml`, `meta.json` (commit, packages, GPU, mixture hash), CSV logs, TensorBoard events, a resumable checkpoint and `final.json` to `runs/{algo}_p{p}_s{seed}/`. Pre-flight refuses a mixture whose hash differs from its committed manifest and evaluation seeds that overlap data collection or the reserved final-grid seeds.

## Results so far

**Phase 4, BC sanity (development data, 100k steps, Tesla T4):** all 15 plumbing checks pass ([report](report/phase4/bc_sanity.json), [spec](docs/specs/phase-4-bc-eval.md)).

| Expert fraction `p` | Seed 100 | Seed 101 | Expected band |
|---|---|---|---|
| 0 (simple data only) | 82.1 | 81.9 | 22.5 to 85.4 |
| 1 (expert data only) | 104.4 | 105.5 | 80.3 to 113.5 |

![BC sanity curves](report/figures/bc_sanity_curves.png)

Normalized score: 0 = random policy, 100 = expert dataset mean. Deterministic BC on the simple data scores about 82, well above that dataset's 42.7, because the mean action removes most of its falls.

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
