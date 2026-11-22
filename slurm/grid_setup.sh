# shellcheck shell=bash
# The approved grid setup (D1, user, 2026-11-21; report/compute.md "Phase 6 benchmark and hardware
# choice"), in one place for slurm/tests_gate.sbatch (sweep --init) and the submit commands.
# CPU on short with the cascadelake constraint, one run per job, 4 CPUs, 8G; --time 1:30 for IQL
# and 0:20 for BC (h x 1.5, given here in hours); short shows no submit limit. Throttle (D3, user,
# 2026-11-21): 25 concurrent jobs in total over the three arrays that run together, split by
# run length (an IQL run takes about 5 x a BC run): main IQL 10, ablation IQL 8, main BC 7.
# shellcheck disable=SC2034  # read by the scripts that source this file
GRID_ROOT="/scratch/$USER/runs_grid"
GRID_SETS=(data.mixture_dir="/scratch/$USER/mixtures")
GRID_INIT_ARGS=(--hardware cpu --partition short --constraint cascadelake
    --model "Intel(R) Xeon(R) Platinum 8276 CPU @ 2.20GHz"
    --algo-setup bc:1:4:8:0.33 iql:1:4:8:1.5 --max-submit 100)
GRID_THROTTLE_main="bc:7 iql:10"
GRID_THROTTLE_ablation="iql:8"
GRID_THROTTLE_dryrun_main="bc:7 iql:10"
GRID_THROTTLE_dryrun_ablation="iql:8"
