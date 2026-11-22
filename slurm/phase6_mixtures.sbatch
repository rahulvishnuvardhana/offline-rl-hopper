#!/bin/bash
# Phase 6 data step (spec Section 4): build into scratch the final-grid mixtures for seeds 3 and 4
# (all five p; user decision D2: 5 seeds) and the development mixtures p in {0.05, 0.5} at seeds
# 100 and 101 (the dry run, D4). Data only: nothing is trained or evaluated.
# Three builds, each on a clean worktree (the builder refuses a dirty tree, M8):
#   1. procedure check: mix_p0.05_s0 and mix_p0.5_s0 (Phase 3, built at 63bf8fa) rebuilt into a
#      temporary directory must match their committed hashes, so seeds 3 and 4 come from the same
#      procedure as seeds 0 to 2; nothing of it reaches scratch or the repo;
#   2. seeds 3 and 4, all five p, into scratch;
#   3. seeds 100 and 101, all five p, into scratch: p 0, 0.2 and 1 already exist there and must
#      come out with their committed hashes (same bytes rewritten); one build for all five p keeps
#      summary_s100_s101.json and audit_s100_s101.json complete (a superset of Phase 5's, superseding
#      them; the six rebuilt manifests are copied out with them).
# No other job may read /scratch/$USER/mixtures meanwhile (files are rewritten in place): submit
# the benchmark with --dependency=afterok:<this job>. New manifests, summaries and audits are
# copied to logs/phase6_mix_<jobid>/ for the laptop to commit. CPU only, `short` (ROADMAP 4.2.1).
# Submit from the repo root, clean, pinned, after `mkdir -p logs`:
#   sbatch --export=ALL,EXPECTED_COMMIT=$(git rev-parse HEAD) slurm/phase6_mixtures.sbatch
#SBATCH --job-name=offrl-p6-mixtures
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load miniconda3/24.11.1
source activate /scratch/$USER/envs/offrl
export MINARI_DATASETS_PATH=/scratch/$USER/minari
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/common.sh
pin_commit
out="$SLURM_SUBMIT_DIR/logs/phase6_mix_${SLURM_JOB_ID}"
enter_worktree
mkdir -p "$out"

# Compares report/mixtures/<name>.json in the worktree with the committed one: same data, or exit 1.
same_data() {
    python - "$1" <<'PY'
import json, subprocess, sys
path = f"report/mixtures/{sys.argv[1]}.json"
new = json.load(open(path))
old = json.loads(subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, check=True).stdout)
same = all(new[k] == old[k] for k in ("content_sha256", "data_sha256"))
print(f"{'SAME DATA' if same else 'CHANGED  '} {sys.argv[1]} content {new['content_sha256'][:12]}")
raise SystemExit(0 if same else 1)
PY
}

# Back to the pinned commit's report/mixtures, so the next build sees a clean tree. Only this
# job's throwaway worktree is touched, never the shared clone.
reset_manifests() {
    git checkout -q -- report/mixtures && git clean -fdq report/mixtures
}

status=0
rebuild="${TMPDIR:-/tmp}/offrl_rebuild_${SLURM_JOB_ID}"
mkdir -p "$rebuild"
python scripts/build_mixtures.py --p 0.05 0.5 --seeds 0 --out-dir "$rebuild" || status=$?
for f in mix_p0.05_s0 mix_p0.5_s0; do same_data "$f" || status=1; done
rm -rf "$rebuild"
if [ "$status" -ne 0 ]; then
    echo "procedure check failed: seeds 3 and 4 are not built" >&2
    exit "$status"
fi
reset_manifests

python scripts/build_mixtures.py --p 0 0.05 0.2 0.5 1 --seeds 3 4 --out-dir /scratch/$USER/mixtures || status=$?
for seed in 3 4; do
    for p in 0 0.05 0.2 0.5 1; do cp "report/mixtures/mix_p${p}_s${seed}.json" "$out"/ || status=1; done
done
cp report/mixtures/summary_s3_s4.json report/mixtures/audit_s3_s4.json "$out"/ || status=1
reset_manifests

python scripts/build_mixtures.py --p 0 0.05 0.2 0.5 1 --seeds 100 101 --out-dir /scratch/$USER/mixtures || status=$?
for f in mix_p0_s100 mix_p0_s101 mix_p0.2_s100 mix_p0.2_s101 mix_p1_s100 mix_p1_s101; do
    same_data "$f" || status=1
done
# All ten s100/s101 manifests go out, the six rebuilt ones too (same data, new commit), so the
# manifests and the superseding audit name one build.
for seed in 100 101; do
    for p in 0 0.05 0.2 0.5 1; do cp "report/mixtures/mix_p${p}_s${seed}.json" "$out"/ || status=1; done
done
cp report/mixtures/summary_s100_s101.json report/mixtures/audit_s100_s101.json "$out"/ || status=1
exit "$status"
