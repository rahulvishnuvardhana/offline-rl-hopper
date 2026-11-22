#!/bin/bash
# Phase 6 benchmark (spec Section 5): K copies of one algorithm on one allocation, 10,000 steps
# each, on the development mixture mix_p1_s100 with development evaluation seeds (spec 0.3: no
# final-grid data before the grid). Logs as the grid does (log_every 1000). Prints ms per update
# over the middle 80% of the update windows, evaluation seconds per environment step with the
# summed episode lengths, the node's CPU and GPU model, peak memory, and the partition limits.
# Partition, device and resources come from the submit line, one job per configuration, e.g.
#   G2:  sbatch -p gpu-short --gres=gpu:t4:1 -c 8 --mem=12G $pin,ALGO=iql,K=2,DEVICE=cuda,GPU="Tesla T4" slurm/benchmark.sbatch
#   C1:  sbatch -p short --constraint=cascadelake -c 4 --mem=8G $pin,ALGO=iql,K=1,DEVICE=cpu slurm/benchmark.sbatch
# with pin="--export=ALL,EXPECTED_COMMIT=$(git rev-parse HEAD)", after slurm/tests_gate.sbatch
# passed at that commit, and after the mixture job (--dependency=afterok:<it>). 4 CPUs and 4G per
# run, plus 4G. At most 30 minutes (ROADMAP 4.2.1). The job refuses an allocation other than the
# one its configuration names (a forgotten -p or --constraint would give plausible numbers).
#SBATCH --job-name=offrl-bench
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
: "${ALGO:?export ALGO=bc|iql}" "${K:?export K=<runs packed>}" "${DEVICE:?export DEVICE=cuda|cpu}"
module load miniconda3/24.11.1
source activate /scratch/$USER/envs/offrl
export MINARI_DATASETS_PATH=/scratch/$USER/minari
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/common.sh
pin_commit
require_gate

if [ $((SLURM_CPUS_PER_TASK % K)) -ne 0 ]; then
    echo "REFUSING: $SLURM_CPUS_PER_TASK CPUs do not divide into $K runs" >&2
    exit 1
fi
job=$(scontrol show job "$SLURM_JOB_ID")
field() { grep -oE "(^| )$1=[^ ]*" <<<"$job" | head -1 | cut -d= -f2-; }
if [ "$DEVICE" = cpu ]; then
    if [ "$(field Partition)" != short ] || [ "$(field Features)" != cascadelake ]; then
        echo "REFUSING: a CPU configuration needs -p short --constraint=cascadelake (spec 5, D1)" >&2
        exit 1
    fi
else
    : "${GPU:?export GPU=<expected model, e.g. Tesla T4>}"
    got=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    if [ "$got" != "$GPU" ]; then
        echo "REFUSING: allocated '$got', configuration names '$GPU'" >&2
        exit 1
    fi
fi
want_mb=$(( (4 * K + 4) * 1024 ))  # a partition default can set SLURM_MEM_PER_NODE too: check the size
if [ "${SLURM_MEM_PER_NODE:-0}" -ne "$want_mb" ]; then
    echo "REFUSING: --mem must be $(( 4 * K + 4 ))G (4G per run plus 4G), got ${SLURM_MEM_PER_NODE:-none} MB" >&2
    exit 1
fi
# Eligible, not Submit: a job waiting on --dependency is not queuing for nodes (cost model, spec 5).
submit=$(field SubmitTime) eligible=$(field EligibleTime) start=$(field StartTime)
echo "QUEUE_WAIT_S=$(( $(date -d "$start" +%s) - $(date -d "$eligible" +%s) )) (submit $submit, eligible $eligible, start $start)"
threads=$((SLURM_CPUS_PER_TASK / K))
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
echo "config: algo=$ALGO k=$K device=$DEVICE cpus=$SLURM_CPUS_PER_TASK mem=${SLURM_MEM_PER_NODE}M threads/run=$threads"
scontrol show node "${SLURMD_NODENAME:-$(hostname -s)}" || true
lscpu | grep -E "^Model name|^Socket|^Core|^Thread" || true
echo "CPU_MODEL=$(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo)"
[ "$DEVICE" = cpu ] || nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "$DEVICE" = cpu ]; then  # free capacity on the constrained nodes: J_eff and queue waits (spec 5)
    echo "cascadelake nodes now (node, features, cores alloc/idle/other/total, free MB):"
    sinfo -p short -N -h -o "%N %f %C %e" | grep cascadelake || true
fi
sacctmgr -n show qos format=name,maxjobspu,maxsubmitpu | grep -E "^ *(gpu|gpu-short|short) " || true
sacctmgr -n show assoc user="$USER" format=user,partition,maxjobs,maxsubmitjobs || true

out=/scratch/$USER/runs_bench_$SLURM_JOB_ID
rm -rf "$out"  # benchmark runs only: never reused
common=(seed=100 data.p_expert=1 train.device="$DEVICE" train.num_threads=$threads train.steps=10000
        train.log_every=1000 train.eval_every=2000 train.checkpoint_every=2000
        eval.episodes=10 eval.final_episodes=10 eval.final_score_last_n=1
        data.mixture_dir=/scratch/$USER/mixtures train.out_dir="$out")
pids=()
for i in $(seq 1 "$K"); do
    python -m offrl.train "configs/algo/$ALGO.yaml" "${common[@]}" train.run_suffix="_copy$i" > /dev/null &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
[ "$status" -eq 0 ] || exit "$status"

python - "$out" "$DEVICE" <<'PY'
import csv, json, pathlib, statistics, sys
out, device = pathlib.Path(sys.argv[1]), sys.argv[2]
u_all, e_all, models = [], [], set()
for run in sorted(out.iterdir()):
    meta = json.loads((run / "meta.json").read_text())
    assert meta["device"].startswith(device), f"{run.name} trained on {meta['device']}, not {device}"
    models.add(meta["gpu"] if device == "cuda" else meta["cpu_model"])
    rows = list(csv.DictReader(open(run / "timing.csv")))
    train = [float(r["seconds"]) for r in rows if r["kind"] == "train"]
    cut = len(train) // 10  # drop the first and last 10%: warm-up, and a run training alone at the end
    mid = train[cut:len(train) - cut]
    eval_s = sum(float(r["seconds"]) for r in rows if r["kind"] == "eval")
    env_steps = sum(int(r["length"]) for r in csv.DictReader(open(run / "eval_episodes.csv")))
    u, e = statistics.mean(mid) / 1000, eval_s / env_steps
    u_all.append(u); e_all.append(e)
    print(f"{run.name:24s} {meta['gpu'] or 'cpu'}: {1000 * u:.2f} ms/update over {len(mid)} of {len(train)} "
          f"windows; {1000 * e:.3f} ms per eval env step over {env_steps} steps")
assert len(models) == 1, f"runs recorded different models: {models}"
print(f"model recorded by every run: {models.pop()}")
u, e = max(u_all), max(e_all)  # the slowest run sets the bundle's time
h = (300_000 * u + 800 * 1_000 * e) / 3600
print(f"model inputs (slowest run): u = {1000 * u:.2f} ms, e = {1000 * e:.3f} ms; h = {h:.2f} h per run, "
      f"--time = {1.5 * h:.2f} h")
PY


# Peak memory of this job's cgroup (cgroup v2), if the node exposes it; seff gives it after the job.
cg=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup 2>/dev/null || true)
[ -n "$cg" ] && [ -r "/sys/fs/cgroup$cg/memory.peak" ] \
    && echo "memory.peak: $(( $(cat "/sys/fs/cgroup$cg/memory.peak") / 1048576 )) MiB" \
    || echo "memory.peak not readable here; use: seff $SLURM_JOB_ID"
exit 0
