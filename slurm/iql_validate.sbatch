#!/bin/bash
# Phase 5 validation runs (spec Section 8): 14 runs at 300k steps, 7 array tasks of 2 runs that
# share one A100 (ROADMAP 4.1 and 4.3; A100 is the user's choice). At most 4 tasks at once.
# --time from the smoke test (slurm/iql_smoke.sbatch); see report/compute.md.
# Submit from the repo root on a clean checkout, then the analysis with a dependency:
#   jid=$(sbatch --parsable slurm/iql_validate.sbatch)
#   sbatch --dependency=afterok:$jid slurm/iql_analyze.sbatch
#SBATCH --job-name=offrl-iql-validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-6%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
module load miniconda3/24.11.1
source activate /scratch/$USER/envs/offrl
export MINARI_DATASETS_PATH=/scratch/$USER/minari
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # deterministic cuBLAS: a preempted task resumes bitwise
cd "$SLURM_SUBMIT_DIR"

tasks=("iql 1" "iql 0.2" "iql 0" "corl_iql 1" "corl_iql 0.2" "bc 1" "bc 0.2")
read -r algo p <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m pytest -o addopts="" -q -x  # a failing test stops the task before any GPU training

threads=$((SLURM_CPUS_PER_TASK / 2))
pids=()
for seed in 100 101; do
    python -m offrl.train "configs/algo/$algo.yaml" seed=$seed data.p_expert=$p train.device=cuda \
        train.num_threads=$threads data.mixture_dir=/scratch/$USER/mixtures \
        train.out_dir=/scratch/$USER/runs_phase5 \
        > "logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_s${seed}.log" 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
git status --short
exit "$status"
