#!/bin/bash
# Phase 4 DoD on one GPU: the test suite (incl. the CUDA-only tests), then the BC sanity runs
# (scripts/bc_sanity.py: 4 runs + resume demo, 300k reruns of any band miss). One small GPU on
# gpu-short for at most 2 h (ROADMAP 4.2.1); evaluation rollouts use the CPUs.
# Submit from the repo root, on a clean checkout, after `mkdir -p logs`.
# Outputs are moved to logs/phase4_<jobid>/ so the clone stays clean; copy them to the laptop.
#SBATCH --job-name=offrl-bc-sanity
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load miniconda3/24.11.1
source activate /scratch/$USER/envs/offrl
export MINARI_DATASETS_PATH=/scratch/$USER/minari
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # deterministic cuBLAS, so the resume demo can match bitwise
cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv
python -m pytest -o addopts="" -q -rs  # a failing test stops the job before any GPU training
status=0

# Development mixtures (npz) into scratch; the committed manifests are rewritten identically,
# then restored, so the tree is clean again before training records git_dirty.
python scripts/build_mixtures.py --p 0 1 --seeds 100 101 --out-dir /scratch/$USER/mixtures || status=$?
git restore report/mixtures/

python scripts/bc_sanity.py --rerun-misses --set train.device=cuda \
    data.mixture_dir=/scratch/$USER/mixtures train.out_dir=/scratch/$USER/runs_phase4 || status=$?

out="logs/phase4_${SLURM_JOB_ID}"
mkdir -p "$out"
mv report/phase4/bc_sanity.json report/figures/bc_sanity_curves.png "$out"/ || true
rmdir report/phase4 2>/dev/null || true
git status --short
exit "$status"
