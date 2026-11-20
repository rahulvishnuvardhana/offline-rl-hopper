#!/bin/bash
# The test suite once per commit on a GPU node (the CUDA tests run here), before any training
# array at that commit: chain the array with --dependency=afterok:<this job>. Replaces running
# pytest on every held training GPU (ROADMAP 4.2.1 rule 2: tests belong on gpu-short).
# Submit from the repo root, clean, pinned:
#   sbatch --export=ALL,EXPECTED_COMMIT=$(git rev-parse HEAD) slurm/tests_gate.sbatch
#SBATCH --job-name=offrl-tests-gate
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load miniconda3/24.11.1
source activate /scratch/$USER/envs/offrl
export MINARI_DATASETS_PATH=/scratch/$USER/minari
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/common.sh
pin_commit

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA: the CUDA tests would skip'"
python -m pytest -o addopts="" -q -rs
pin_commit  # the tests left the clone clean
mkdir -p "$GATE_DIR"
echo "job $SLURM_JOB_ID $(date -Is)" > "$GATE_DIR/$EXPECTED_COMMIT.ok"  # require_gate reads this
