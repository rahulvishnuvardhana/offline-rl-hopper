#!/bin/bash
# Phase 7/8 grid submission at M (spec Sections 8 and 9; login-node workaround: single-threaded).
# Order (D3): main IQL, ablation IQL, main BC. Each `sbatch` follows its `--sbatch-args` at once;
# `set -e` stops at the first failure, so nothing after a failed step is submitted.
set -euo pipefail
M=ea11e14d4ab7d0c6d2dc5eaabff6b41049fe17c3
CK=/scratch/$USER/src/$M
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
module load miniconda3/24.11.1 >/dev/null 2>&1
source activate /scratch/$USER/envs/offrl
export PYTHONPATH="$CK/src" MINARI_DATASETS_PATH=/scratch/$USER/minari
cd "$CK"
test "$(git rev-parse HEAD)" = "$M" && test -z "$(git status --porcelain)"
test -z "$(squeue -h -u "$USER")"
source slurm/grid_setup.sh
sw() { local g=$1; shift; python -m offrl.sweep --grid "configs/grid_$g.yaml" --root "$GRID_ROOT" --set "${GRID_SETS[@]}" --commit "$M" "$@"; }
sw main --grid configs/grid_ablation.yaml --submit-plan
for ga in main:iql ablation:iql main:bc; do
    g=${ga%%:*} a=${ga##*:}
    args=$(sw "$g" --algo "$a" --sbatch-args)
    echo "$g $a: $args"
    # shellcheck disable=SC2086  # the printed arguments are split on purpose
    j=$(sbatch --parsable $args --export=ALL,EXPECTED_COMMIT=$M,GRID=$g,ALGO=$a slurm/grid.sbatch)
    echo "SUBMITTED $g $a $j"
done
squeue -u "$USER" -r -h -o "%i %j %T" | awk '{print $2, $3}' | sort | uniq -c
