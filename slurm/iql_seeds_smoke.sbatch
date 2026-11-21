#!/bin/bash
# Phase 5 V1 follow-up, T4 timing test (spec Section 8, D6): sizes iql_seeds.sbatch's --time on
# the GPU the follow-up runs on. Packs our IQL and CORL's on one T4 exactly as iql_seeds.sbatch
# does (same CPUs and threads), 10,000 steps each on the p = 1, seed 102 mixture (long enough that
# the two runs overlap), and prints ms per update over the middle 80% of the windows (both runs
# training) and evaluation seconds per environment step (early policies fall sooner than trained
# ones, so evaluation is sized per step: 800 episodes x 1000 steps). One T4 on gpu-short for at
# most 30 minutes (ROADMAP 4.2.1). The estimate errs high (log_every 100 here vs. 1000 in the real
# job: more syncs and writes per window) and leaves out the real job's startup and checkpoint
# writes, which the x 1.5 margin covers.
# Needs /scratch/$USER/mixtures with mix_p1_s102 (seed_mixtures.sbatch). Submit pinned, after
# slurm/tests_gate.sbatch passed at the same commit:
#   sbatch --export=ALL,EXPECTED_COMMIT=$(git rev-parse HEAD) slurm/iql_seeds_smoke.sbatch
#SBATCH --job-name=offrl-iql-seeds-smoke
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
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
require_gate

nvidia-smi --query-gpu=name,memory.total --format=csv
out=/scratch/$USER/runs_seeds_smoke_$SLURM_JOB_ID
rm -rf "$out"  # smoke runs only: never reused
threads=$((SLURM_CPUS_PER_TASK / 2))
common=(seed=102 data.p_expert=1 train.device=cuda train.num_threads=$threads train.steps=10000
        train.log_every=100 train.eval_every=2000 train.checkpoint_every=2000
        eval.episodes=10 eval.final_episodes=10 eval.final_score_last_n=1
        data.mixture_dir=/scratch/$USER/mixtures train.out_dir=$out)

pids=()
for algo in iql corl_iql; do
    python -m offrl.train "configs/algo/$algo.yaml" "${common[@]}" > /dev/null &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[ "$status" -eq 0 ] || exit "$status"

python - "$out" <<'PY'
import csv, json, pathlib, statistics, sys
estimates = []
for run in sorted(pathlib.Path(sys.argv[1]).iterdir()):
    gpu = json.loads((run / "meta.json").read_text())["gpu"]
    assert gpu == "Tesla T4", f"{run.name} ran on {gpu}, not a Tesla T4"
    rows = list(csv.DictReader(open(run / "timing.csv")))
    train = [float(r["seconds"]) for r in rows if r["kind"] == "train"]
    cut = len(train) // 10  # drop the first and last 10%: warm-up, and either run training alone
    mid = train[cut:len(train) - cut]
    eval_s = sum(float(r["seconds"]) for r in rows if r["kind"] == "eval")
    env_steps = sum(int(r["length"]) for r in csv.DictReader(open(run / "eval_episodes.csv")))
    per_update, per_step = 1000 * statistics.mean(mid) / 100, eval_s / env_steps
    hours = (per_update / 1000 * 300_000 + per_step * 1000 * 800) * 1.5 / 3600
    print(f"{run.name:20s} {gpu}: {per_update:.2f} ms/update (packed, middle 80%), "
          f"{1000 * per_step:.3f} ms per eval env step; --time estimate {hours:.2f} h")
    print("  seconds per 100-update window:", " ".join(f"{s:.2f}" for s in train))
    estimates.append(hours)
print(f"use for iql_seeds.sbatch --time: {max(estimates):.2f} h (the slower run)")
PY
