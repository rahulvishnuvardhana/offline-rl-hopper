#!/bin/bash
# Phase 6 data step (spec Section 4): add to scratch the final-grid mixtures for seeds 3 and 4 (all
# five p; user decision D2: 5 seeds) and the development mixtures p in {0.05, 0.5} at seeds 100
# and 101 (the dry run, D4). Data only: nothing is trained or evaluated.
# Existing mixture files are never written (user, 2026-11-22). Every build goes to a staging
# directory in scratch; existing mixtures are checked by rebuilding them there and comparing
# hashes with their committed manifests; new files are installed only after every check passes,
# each by an atomic rename within the same filesystem, and only where no file exists yet. Any
# failure stops the job before the install, so the originals and scratch are left as they were.
# Three builds, each on a clean worktree (the builder refuses a dirty tree, M8):
#   1. procedure check: mix_p0.05_s0 and mix_p0.5_s0 (Phase 3, built at 63bf8fa) must match their
#      committed hashes, so seeds 3 and 4 come from the same procedure as seeds 0 to 2;
#   2. seeds 3 and 4, all five p (new);
#   3. seeds 100 and 101, all five p: p 0, 0.2 and 1 must match their committed hashes (checked,
#      not installed); p 0.05 and 0.5 are new. One build for all five p keeps
#      summary_s100_s101.json and audit_s100_s101.json complete (superseding Phase 5's).
# New and rebuilt manifests, summaries and audits are copied to logs/phase6_mix_<jobid>/ for the
# laptop to commit. CPU only, `short` (ROADMAP 4.2.1). Submit from the repo root, clean, pinned,
# after `mkdir -p logs`; the benchmark goes after it with --dependency=afterok:<this job>:
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
mixtures=/scratch/$USER/mixtures
stage=/scratch/$USER/mixtures_stage_${SLURM_JOB_ID}  # same filesystem as $mixtures: rename is atomic
out="$SLURM_SUBMIT_DIR/logs/phase6_mix_${SLURM_JOB_ID}"
new=()
for seed in 3 4; do for p in 0 0.05 0.2 0.5 1; do new+=("mix_p${p}_s${seed}"); done; done
for seed in 100 101; do for p in 0.05 0.5; do new+=("mix_p${p}_s${seed}"); done; done
for f in "${new[@]}"; do  # refuse before building anything: an install never replaces a file
    if [ -e "$mixtures/$f.npz" ]; then
        echo "REFUSING: $mixtures/$f.npz already exists; this job only adds files" >&2
        exit 1
    fi
done
enter_worktree
mkdir -p "$out" "$stage/check" "$stage/new"
# The staging copies go whatever happens; the job keeps its exit status (_remove_worktree reads $?).
trap 'rc=$?; rm -rf "$stage"; (exit "$rc"); _remove_worktree' EXIT

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

# Every step below must pass; set -e stops the job at the first failure, before the install.
python scripts/build_mixtures.py --p 0.05 0.5 --seeds 0 --out-dir "$stage/check"
same_data mix_p0.05_s0
same_data mix_p0.5_s0
reset_manifests

python scripts/build_mixtures.py --p 0 0.05 0.2 0.5 1 --seeds 3 4 --out-dir "$stage/new"
for seed in 3 4; do
    for p in 0 0.05 0.2 0.5 1; do cp "report/mixtures/mix_p${p}_s${seed}.json" "$out"/; done
done
cp report/mixtures/summary_s3_s4.json report/mixtures/audit_s3_s4.json "$out"/
reset_manifests

python scripts/build_mixtures.py --p 0 0.05 0.2 0.5 1 --seeds 100 101 --out-dir "$stage/new"
for f in mix_p0_s100 mix_p0_s101 mix_p0.2_s100 mix_p0.2_s101 mix_p1_s100 mix_p1_s101; do
    same_data "$f"
    rm "$stage/new/$f.npz"  # checked, never installed: the existing file stays as it is
done
# All ten s100/s101 manifests go out, the six rebuilt ones too (same data, new commit), so the
# manifests and the superseding audit name one build.
for seed in 100 101; do
    for p in 0 0.05 0.2 0.5 1; do cp "report/mixtures/mix_p${p}_s${seed}.json" "$out"/; done
done
cp report/mixtures/summary_s100_s101.json report/mixtures/audit_s100_s101.json "$out"/

# Install: every check passed. Each new file is renamed into place (atomic on one filesystem);
# `mv -n` never replaces a file that appeared meanwhile, and the count below proves none did.
for f in "${new[@]}"; do mv -n "$stage/new/$f.npz" "$mixtures/$f.npz"; done
left=$(find "$stage/new" -name '*.npz' | wc -l)
if [ "$left" -ne 0 ]; then
    echo "INSTALL INCOMPLETE: $left file(s) not moved (a target appeared meanwhile)" >&2
    exit 1
fi
# The installed files against their manifests, read back from scratch.
python scripts/check_mixture_files.py --manifests "$out" --mixture-dir "$mixtures"
echo "installed ${#new[@]} new mixtures; existing files untouched"
