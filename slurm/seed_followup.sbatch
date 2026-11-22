#!/bin/bash
# Phase 5 V1 follow-up: apply the seed 102 to 104 rule (scripts/seed_followup.py; spec Section 8,
# D6 and D7) to the runs of slurm/iql_seeds.sbatch. CPU only, `short` partition. Submit pinned,
# naming the commit the array was pinned to (the analysis may run at a later commit):
#   sbatch --export=ALL,EXPECTED_COMMIT=$(git rev-parse HEAD),TRAINING_COMMIT=<array's sha> \
#       slurm/seed_followup.sbatch
# Add RERUN_OUT_DIR=<dir> to the export list once failed runs were rerun there (D7), and only once
# the rerun job has ended (--dependency=afterany:<rerun job>): a rerun still training reads as a
# second failure.
# Exit 0 proceed, 10 investigate, 11 rerun needed: Slurm shows 10 and 11 as FAILED by design; the
# report is written for all three. 3 = refusal (bad args or dirty tree), 1 = error or pin refusal;
# neither writes a report. Works in a throwaway worktree; outputs go to
# logs/phase5_<jobid>/ for the laptop to copy and commit.
#SBATCH --job-name=offrl-seed-followup
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load miniconda3/24.11.1
source activate /scratch/$USER/envs/offrl
: "${TRAINING_COMMIT:?export TRAINING_COMMIT=<the sha iql_seeds.sbatch was pinned to>}"
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/common.sh
pin_commit
out="$SLURM_SUBMIT_DIR/logs/phase5_${SLURM_JOB_ID}"
enter_worktree

rerun=()
[ -z "${RERUN_OUT_DIR:-}" ] || rerun=(--rerun-out-dir "$RERUN_OUT_DIR")
status=0
python scripts/seed_followup.py --commit "$TRAINING_COMMIT" \
    --set data.mixture_dir=/scratch/$USER/mixtures train.out_dir=/scratch/$USER/runs_phase5_seeds \
    "${rerun[@]}" || status=$?

if [ -f report/phase5/seed_followup.json ]; then
    mkdir -p "$out"
    cp report/phase5/seed_followup.json "$out"/ || { [ "$status" -ne 0 ] || status=1; }  # keep a non-zero status
fi
exit "$status"
