#!/bin/bash
# Phase 3 cross-machine reconciliation: rebuild the 15 final mixtures on Explorer and compare
# their content hashes with the manifests committed from the laptop. CPU only, so it runs on
# the `short` CPU partition, never a GPU (ROADMAP 4.2.1, responsible use).
#SBATCH --job-name=offrl-mixtures
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

python scripts/build_mixtures.py --out-dir /scratch/$USER/mixtures

# Compare what Explorer just wrote with the manifests committed from the laptop.
python - <<'EOF'
import json, pathlib, subprocess
mismatch = 0
for path in sorted(pathlib.Path("report/mixtures").glob("mix_*.json")):
    new = json.loads(path.read_text())
    old = json.loads(subprocess.run(["git", "show", f"HEAD:{path.as_posix()}"], capture_output=True, check=True).stdout)
    for key in ("content_sha256", "data_sha256"):
        ok = new[key] == old[key]
        mismatch += not ok
        print(f"{'MATCH   ' if ok else 'MISMATCH'} {path.stem} {key} {new[key][:12]} vs laptop {old[key][:12]}")
print("CROSS-MACHINE RECONCILIATION:", "PASS" if mismatch == 0 else f"FAIL ({mismatch} mismatches)")
EOF

# Keep Explorer's audit as evidence (logs/ is gitignored), then restore the clone to HEAD.
cp report/mixtures/audit_s0_s1_s2.json "logs/explorer_mixture_audit_${SLURM_JOB_ID}.json"
git restore report/mixtures/
