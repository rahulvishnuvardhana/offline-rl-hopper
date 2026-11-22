#!/bin/bash
# Phase 5 V1 follow-up (user decision, 2026-11-20): our IQL and CORL's at p = 1 on development
# seeds 102 to 104, to tell seed variance from a real difference. V1 itself stays on seeds 100
# and 101, threshold unchanged. One task per seed; its two runs share one T4 and one mixture.
# All six runs on T4 (user decision D6, so an ours-vs-CORL gap cannot come from the GPU).
# --time from the T4 timing test (iql_seeds_smoke.sbatch, job 12031927, packed as here): ours
# 16.26 ms/update and 1.408 ms per eval env step -> (300k updates + 800 x 1000 eval steps) x 1.5
# = 2.50 h, set to 2:40 for storage stalls (2026-11-20). The gpu partition has one 4-T4 node, enough for the 3 tasks.
# Needs the mix_p1_s102..s104 manifests committed and the mixtures in scratch (seed_mixtures.sbatch).
# Own out_dir: the first attempt (array 12014822, cancelled) left partial runs at another commit
# in runs_phase5, which a resume would refuse. Submit pinned, after the test gate at the same
# commit (see iql_validate.sbatch):
#   sbatch $pin --dependency=afterok:$gate slurm/iql_seeds.sbatch
#SBATCH --job-name=offrl-iql-seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:40:00
#SBATCH --array=0-2%3
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
