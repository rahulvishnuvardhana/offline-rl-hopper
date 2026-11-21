#!/bin/bash
# Phase 0 smoke test: 1 small GPU for 10 minutes. See ROADMAP 4.2.1 (responsible use).
# Submit from the repo root after `mkdir -p logs` (gitignored; Slurm cannot create it), clean and pinned:
#   sbatch --export=ALL,EXPECTED_COMMIT=$(git rev-parse HEAD) slurm/smoke.sbatch
#SBATCH --job-name=offrl-smoke
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load miniconda3/24.11.1
source activate /scratch/$USER/envs/offrl
export MINARI_DATASETS_PATH=/scratch/$USER/minari
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/common.sh
pin_commit

nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/smoke.py --datasets --require-gpu
