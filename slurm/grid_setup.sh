# shellcheck shell=bash
# The approved grid setup (D1, user, 2026-11-21; report/compute.md "Phase 6 benchmark and hardware
# choice"), in one place for slurm/tests_gate.sbatch (sweep --init) and the submit commands.
# CPU on short with the cascadelake constraint, one run per job, 4 CPUs, 8G; --time 1:30 for IQL
# and 0:20 for BC (h x 1.5, given here in hours); throttle 25 (D3); short shows no submit limit.
# shellcheck disable=SC2034  # read by the scripts that source this file
GRID_ROOT="/scratch/$USER/runs_grid"
GRID_SETS=(data.mixture_dir="/scratch/$USER/mixtures")
GRID_INIT_ARGS=(--hardware cpu --partition short --constraint cascadelake
    --model "Intel(R) Xeon(R) Platinum 8276 CPU @ 2.20GHz"
    --algo-setup bc:1:4:8:0.33 iql:1:4:8:1.5 --throttle 25 --max-submit 100)
