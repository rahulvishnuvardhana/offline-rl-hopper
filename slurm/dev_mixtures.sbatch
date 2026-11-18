#!/bin/bash
# Phase 5 data step: build the development mixtures p in {0, 0.2, 1} x seeds {100, 101} into
# scratch. p = 0 and 1 must reproduce the committed Phase 4 manifests exactly; the new p = 0.2
# manifests and the refreshed summary and audit are copied to logs/phase5_mix_<jobid>/ for the
# laptop to commit (spec Section 8). CPU only, `short` partition (ROADMAP 4.2.1).
# Submit from the repo root, on a clean checkout, after `mkdir -p logs`.
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
export PYTHONPATH="$SLURM_SUBMIT_DIR/src"
cd "$SLURM_SUBMIT_DIR"

status=0
python scripts/build_mixtures.py --p 0 0.2 1 --seeds 100 101 --out-dir /scratch/$USER/mixtures || status=$?

# Phase 4's manifests must come out byte-identical; anything else changed is listed.
for f in report/mixtures/mix_p0_s100.json report/mixtures/mix_p0_s101.json \
         report/mixtures/mix_p1_s100.json report/mixtures/mix_p1_s101.json; do
    if git diff --quiet -- "$f"; then echo "UNCHANGED $f"; else echo "CHANGED   $f"; status=1; fi
done
git status --short

out="logs/phase5_mix_${SLURM_JOB_ID}"
mkdir -p "$out"
cp report/mixtures/mix_p0.2_s100.json report/mixtures/mix_p0.2_s101.json \
   report/mixtures/summary_s100_s101.json report/mixtures/audit_s100_s101.json "$out"/ || status=1
git restore report/mixtures/
git clean -fq report/mixtures/
exit "$status"
