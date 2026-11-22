#!/bin/bash
# Phase 5 smoke test (spec Section 8): 2,000 steps each of BC, IQL and CORL on the p = 1, seed 100
# development mixture, then two IQL runs at once on the same GPU (the packing the array uses).
# Prints ms per update and seconds per evaluation, to size iql_validate.sbatch's --time.
# One A100 on gpu-short for at most 30 minutes (ROADMAP 4.2.1; A100 is the user's choice).
# Needs /scratch/$USER/mixtures (slurm/dev_mixtures.sbatch). Submit from the repo root, pinned:
#   sbatch --export=ALL,EXPECTED_COMMIT=$(git rev-parse HEAD) slurm/iql_smoke.sbatch
# after slurm/tests_gate.sbatch passed at that commit.
#SBATCH --job-name=offrl-iql-smoke
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
require_gate

nvidia-smi --query-gpu=name,memory.total --format=csv
out=/scratch/$USER/runs_smoke5
rm -rf "$out"  # smoke runs only: never reused
common=(seed=100 data.p_expert=1.0 train.device=cuda train.num_threads=4 train.steps=2000
        train.log_every=100 train.eval_every=1000 train.checkpoint_every=1000
        eval.episodes=10 eval.final_episodes=10 eval.final_score_last_n=1
        data.mixture_dir="/scratch/$USER/mixtures" train.out_dir="$out")

for algo in bc iql corl_iql; do
    python -m offrl.train configs/algo/$algo.yaml "${common[@]}" train.run_suffix=_single > /dev/null
done
pids=()
for run in _packedA _packedB; do
    python -m offrl.train configs/algo/iql.yaml "${common[@]}" train.run_suffix=$run > /dev/null &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done  # set -e: a failed packed run fails the job

python - "$out" <<'PY'
import csv, pathlib, statistics, sys
for run in sorted(pathlib.Path(sys.argv[1]).iterdir()):
    rows = list(csv.DictReader(open(run / "timing.csv")))
    train = [float(r["seconds"]) for r in rows if r["kind"] == "train"][1:]  # skip warm-up window
    evals = [float(r["seconds"]) for r in rows if r["kind"] == "eval"]
    print(f"{run.name:28s} {1000 * statistics.mean(train) / 100:6.2f} ms/update   "
          f"{statistics.mean(evals):6.1f} s per 10-episode evaluation")
PY
