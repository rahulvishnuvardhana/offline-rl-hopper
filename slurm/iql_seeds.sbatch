#!/bin/bash
# Phase 5 V1 follow-up (user decision, 2026-11-20): our IQL and CORL's at p = 1 on development
# seeds 102 to 104, to tell seed variance from a real difference. V1 itself stays on seeds 100
# and 101, threshold unchanged. One task per seed; its two runs share one A100 and one mixture.
# --time as iql_validate.sbatch (two packed runs per task; that array's tasks took at most 85 min).
# Needs the mix_p1_s102..s104 manifests committed and the mixtures in scratch (seed_mixtures.sbatch).
# Own out_dir: the first attempt (array 12014822, cancelled) left partial runs at another commit
# in runs_phase5, which a resume would refuse. Submit pinned, after the test gate at the same
# commit (see iql_validate.sbatch):
#   sbatch $pin --dependency=afterok:$gate slurm/iql_seeds.sbatch
#SBATCH --job-name=offrl-iql-seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --array=0-2%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
module load miniconda3/24.11.1
source activate /scratch/$USER/envs/offrl
export MINARI_DATASETS_PATH=/scratch/$USER/minari
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # deterministic cuBLAS: a preempted task resumes bitwise
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/common.sh
pin_commit
require_gate

seed=$((102 + SLURM_ARRAY_TASK_ID))
nvidia-smi --query-gpu=name,memory.total --format=csv

threads=$((SLURM_CPUS_PER_TASK / 2))
pids=()
for algo in iql corl_iql; do
    python -m offrl.train "configs/algo/$algo.yaml" seed=$seed data.p_expert=1 train.device=cuda \
        train.num_threads=$threads data.mixture_dir=/scratch/$USER/mixtures \
        train.out_dir=/scratch/$USER/runs_phase5_seeds \
        > "logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${algo}_s${seed}.log" 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
git status --short
exit "$status"
