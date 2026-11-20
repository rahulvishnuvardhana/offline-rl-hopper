#!/bin/bash
# Phase 5 checks V0 to V4 after the validation array (scripts/iql_validate.py). CPU only, `short`
# partition; submit pinned with --dependency=afterok:<array job id> (see slurm/iql_validate.sbatch).
# Works in a throwaway worktree, so the shared clone stays clean; outputs go to
# logs/phase5_<jobid>/ for the laptop to copy and commit.
#SBATCH --job-name=offrl-iql-analyze
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load miniconda3/24.11.1
source activate /scratch/$USER/envs/offrl
export MINARI_DATASETS_PATH=/scratch/$USER/minari
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/common.sh
pin_commit
out="$SLURM_SUBMIT_DIR/logs/phase5_${SLURM_JOB_ID}"
enter_worktree

status=0
python scripts/iql_validate.py --set data.mixture_dir=/scratch/$USER/mixtures \
    train.out_dir=/scratch/$USER/runs_phase5 || status=$?

# The report is written whether checks pass (exit 0) or fail (exit 1); a refusal writes none.
if [ -f report/phase5/iql_validation.json ]; then
    mkdir -p "$out"
    cp report/phase5/iql_validation.json report/figures/iql_validation_curves.png "$out"/ || status=1
fi
exit "$status"
