#!/bin/bash
# Phase 5 V1 follow-up (user decision, 2026-11-20): build the p = 1 development mixtures for
# seeds 102 to 104 into scratch. The manifests, summary and audit are copied to
# logs/phase5_seedmix_<jobid>/ for the laptop to commit before any run (spec Section 8).
# CPU only, `short` partition (ROADMAP 4.2.1).
# Submit from the repo root, on a clean checkout, after `mkdir -p logs`.
#SBATCH --job-name=offrl-seed-mixtures
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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
python scripts/build_mixtures.py --p 1 --seeds 102 103 104 --out-dir /scratch/$USER/mixtures || status=$?
git status --short

out="logs/phase5_seedmix_${SLURM_JOB_ID}"
mkdir -p "$out"
cp report/mixtures/mix_p1_s102.json report/mixtures/mix_p1_s103.json report/mixtures/mix_p1_s104.json \
   report/mixtures/summary_s102_s103_s104.json report/mixtures/audit_s102_s103_s104.json "$out"/ || status=1
git restore report/mixtures/
git clean -fq report/mixtures/
exit "$status"
