#!/bin/bash
# Phase 5 checks V0 to V4 after the validation array (scripts/iql_validate.py). CPU only, `short`
# partition; submit with --dependency=afterok:<array job id> (see slurm/iql_validate.sbatch).
# Outputs are moved to logs/phase5_<jobid>/ so the clone stays clean; copy them to the laptop.
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
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
cd "$SLURM_SUBMIT_DIR"

status=0
python scripts/iql_validate.py --set data.mixture_dir=/scratch/$USER/mixtures \
    train.out_dir=/scratch/$USER/runs_phase5 || status=$?

out="logs/phase5_${SLURM_JOB_ID}"
mkdir -p "$out"
mv report/phase5/iql_validation.json report/figures/iql_validation_curves.png "$out"/ || true
rmdir report/phase5 2>/dev/null || true
git status --short
exit "$status"
