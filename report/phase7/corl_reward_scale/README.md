# CORL-style reward scale vs. ours (2026-11-24)

`corl_scale.py` recomputes CORL's `return_reward_range` factor, `1000 / (max_ret - min_ret)` with an
episode closed at a terminal or at 1000 steps in file order (partial episodes included), and prints it
next to the mixture's own `reward_scale` (ours: complete episodes only). Read-only.

- `explorer_job_12114943.txt`: development mixtures from `~/archive` on Explorer (Slurm job 12114943):
  `python corl_scale.py <archive>/phase6_mixtures/mix_p0.05_s100.npz ... mix_p0.2_s101.npz`.
- `laptop_grid_seed0.txt`: grid seed 0 mixtures in the laptop's `data/mixtures/`:
  `python report/phase7/corl_reward_scale/corl_scale.py data/mixtures/mix_p{0.05,0.2,0.5,0}_s0.npz`.

Equal on 5 of the 9 mixtures; lower by 0.8% (`p = 0.05` s100, `p = 0.2` s101), 3.2% (`p = 0.2` s0)
and 8.9% (`p = 0.5` s0). Cited in ROADMAP Phase 10 (limitations).
