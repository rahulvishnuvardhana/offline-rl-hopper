#!/bin/bash
# Phase 5 data step: build the development mixtures p in {0, 0.2, 1} x seeds {100, 101} into
# scratch. p = 0 and 1 must reproduce the committed Phase 4 manifests exactly; the new p = 0.2
# manifests and the refreshed summary and audit are copied to logs/phase5_mix_<jobid>/ for the
# laptop to commit (spec Section 8). CPU only, `short` partition (ROADMAP 4.2.1). Builds in a
# throwaway worktree of the pinned commit, so the shared clone is never written to.
# Submit from the repo root, clean, pinned, after `mkdir -p logs`:
#   sbatch --export=ALL,EXPECTED_COMMIT=$(git rev-parse HEAD) slurm/dev_mixtures.sbatch
#SBATCH --job-name=offrl-dev-mixtures
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
module load miniconda3/24.11.1
source activate /scratch/$USER/envs/offrl
export MINARI_DATASETS_PATH=/scratch/$USER/minari
cd "$SLURM_SUBMIT_DIR"
source slurm/lib/common.sh
pin_commit
out="$SLURM_SUBMIT_DIR/logs/phase5_mix_${SLURM_JOB_ID}"
enter_worktree

status=0
python scripts/build_mixtures.py --p 0 0.2 1 --seeds 100 101 --out-dir /scratch/$USER/mixtures || status=$?

# Phase 4's mixtures must come out with the same content and data hashes. The manifests' bytes
# do change: they record the commit, time and package builds they were made with.
for f in mix_p0_s100 mix_p0_s101 mix_p1_s100 mix_p1_s101; do
    python - "$f" <<'PY' || status=1
import json, subprocess, sys
path = f"report/mixtures/{sys.argv[1]}.json"
new = json.load(open(path))
old = json.loads(subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, check=True).stdout)
same = all(new[k] == old[k] for k in ("content_sha256", "data_sha256"))
print(f"{'SAME DATA' if same else 'CHANGED  '} {sys.argv[1]} content {new['content_sha256'][:12]}")
raise SystemExit(0 if same else 1)
PY
done

mkdir -p "$out"
cp report/mixtures/mix_p0.2_s100.json report/mixtures/mix_p0.2_s101.json \
   report/mixtures/summary_s100_s101.json report/mixtures/audit_s100_s101.json "$out"/ || status=1
exit "$status"
