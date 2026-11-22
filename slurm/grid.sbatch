#!/bin/bash
# One bundle of one algorithm of one grid (Phase 6 spec Sections 6 to 8). Resources come from the
# grid's sweep.json, so submit with the arguments `sweep --sbatch-args` prints, from the commit's
# checkout (created by slurm/tests_gate.sbatch with GRIDS), with the scratch env active:
#   cd /scratch/$USER/src/<sha> && source slurm/grid_setup.sh
#   args=$(PYTHONPATH=src python -m offrl.sweep --grid configs/grid_<G>.yaml --root "$GRID_ROOT" \
#       --set "${GRID_SETS[@]}" --algo <A> --sbatch-args --indices <one wave of --submit-plan>)
#   sbatch $args --export=ALL,EXPECTED_COMMIT=<sha>,GRID=<G>,ALGO=<A> slurm/grid.sbatch
# Everything runs from the checkout, never the shared clone, which may be pulled meanwhile. The
# sweep is exec'd, so Slurm's SIGTERM (time-out, scancel) reaches it and its locks are released.
#SBATCH --job-name=offrl-grid
# Overridden by --sbatch-args; here only so a bare submission never writes a log into the checkout.
#SBATCH --output=/scratch/%u/logs/offrl-grid_unconfigured_%A_%a.out
#SBATCH --error=/scratch/%u/logs/offrl-grid_unconfigured_%A_%a.err

set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${GRID:?export GRID=main|ablation|dryrun_main|dryrun_ablation}" "${ALGO:?}"
: "${SLURM_ARRAY_TASK_ID:?submit as an array, with the arguments sweep --sbatch-args prints}"
export CHECKOUT_ROOT="/scratch/$USER/src"  # the tree sourced below is the tree entered
# shellcheck source=slurm/lib/common.sh
source "$CHECKOUT_ROOT/$EXPECTED_COMMIT/slurm/lib/common.sh"
use_commit_checkout
pin_commit
require_gate
# shellcheck source=slurm/grid_setup.sh
source slurm/grid_setup.sh
module load miniconda3/24.11.1
source activate "/scratch/$USER/envs/offrl"
export MINARI_DATASETS_PATH="/scratch/$USER/minari"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
lscpu | grep -E "^Model name" || true
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader || true
exec python -m offrl.sweep --grid "configs/grid_$GRID.yaml" --root "$GRID_ROOT" --set "${GRID_SETS[@]}" \
    --commit "$EXPECTED_COMMIT" --algo "$ALGO" --bundle "$SLURM_ARRAY_TASK_ID"
