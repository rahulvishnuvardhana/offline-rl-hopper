# shellcheck shell=bash
# Sourced by the job scripts (from the repo root, after `cd "$SLURM_SUBMIT_DIR"`).
#
# pin_commit: refuse to run unless the shared clone is at the commit pinned at submission and
# clean, so a `git pull` between submission and start cannot change what a task runs. Submit with
#   sbatch --export=ALL,EXPECTED_COMMIT=$(git rev-parse HEAD) slurm/<job>.sbatch
# The clone is checked when a task starts; do not pull while an array is queued or running
# (each run's meta.json records the commit it started at, and V0 checks it).
#
# require_gate: refuse to train unless slurm/tests_gate.sbatch passed at the pinned commit (it
# leaves a marker in scratch), so an array submitted without --dependency cannot skip the tests.
#
# enter_worktree: for jobs that write into the repo (mixture manifests, analysis reports), work in
# a throwaway worktree of the pinned commit, so the shared clone is never written to and runs
# starting meanwhile do not see a dirty tree. Removed on exit, keeping the job's exit status.
# Locked, never pruned: worktrees sit in node-local $TMPDIR, invisible to jobs on other nodes,
# so a `git worktree prune` there would delete a live job's entry. A job killed before its trap
# leaves a locked entry (its reason names the job and node); list with `git worktree list -v`,
# check the job is gone (`squeue -j <id>`), then `git worktree remove --force --force <path>`.
#
# create_commit_checkout / use_commit_checkout (Phase 6 spec Section 7): a grid runs from an
# immutable checkout of its commit, $CHECKOUT_ROOT/<sha>, never from the shared clone, so the
# clone can be pulled and used while a grid of many hours runs. Only the test gate creates the
# checkout (and runs the tests in it, so its marker certifies that checkout); grid jobs source
# this file from the checkout itself and enter it with use_commit_checkout, which refuses a
# missing, moved or modified checkout. Remove it after the copy-out, with no job left:
# scripts/archive_runs.sh --remove-checkout <sha>.

GATE_DIR="/scratch/$USER/gates"
CHECKOUT_ROOT="${CHECKOUT_ROOT:-/scratch/$USER/src}"

# Refuse unless $1 is a git checkout at $EXPECTED_COMMIT with no uncommitted changes.
check_commit_checkout() {
    local dir=$1 head
    if ! head=$(git -C "$dir" rev-parse HEAD 2>/dev/null); then
        echo "REFUSING: $dir is not a git checkout" >&2
        exit 1
    fi
    if [ "$head" != "$EXPECTED_COMMIT" ]; then
        echo "REFUSING: $dir is at $head, not $EXPECTED_COMMIT" >&2
        exit 1
    fi
    if [ -n "$(git -C "$dir" status --porcelain)" ]; then
        echo "REFUSING: $dir has uncommitted changes" >&2
        exit 1
    fi
}

# The test gate only: create the commit's checkout (locked, so no prune removes it), or accept
# an existing one that is at the commit and clean.
create_commit_checkout() {
    : "${EXPECTED_COMMIT:?submit with --export=ALL,EXPECTED_COMMIT=\$(git rev-parse HEAD)}"
    local dir="$CHECKOUT_ROOT/$EXPECTED_COMMIT"
    if [ ! -e "$dir" ]; then
        mkdir -p "$CHECKOUT_ROOT"
        git -C "${SLURM_SUBMIT_DIR:-$PWD}" worktree add --detach --lock \
            --reason "grid checkout of $EXPECTED_COMMIT" -q "$dir" "$EXPECTED_COMMIT"
    fi
    check_commit_checkout "$dir"
}

# Grid jobs: enter the commit's checkout and import from it.
use_commit_checkout() {
    : "${EXPECTED_COMMIT:?submit with --export=ALL,EXPECTED_COMMIT=<sha>}"
    CHECKOUT="$CHECKOUT_ROOT/$EXPECTED_COMMIT"
    if [ ! -d "$CHECKOUT" ]; then
        echo "REFUSING: no checkout $CHECKOUT (slurm/tests_gate.sbatch with GRIDS creates it)" >&2
        exit 1
    fi
    check_commit_checkout "$CHECKOUT"
    cd "$CHECKOUT" || exit 1
    export PYTHONPATH="$CHECKOUT/src"
}

pin_commit() {
    : "${EXPECTED_COMMIT:?submit with --export=ALL,EXPECTED_COMMIT=\$(git rev-parse HEAD)}"
    local head
    head=$(git rev-parse HEAD)
    if [ "$head" != "$EXPECTED_COMMIT" ]; then
        echo "REFUSING: clone is at $head, job pinned to $EXPECTED_COMMIT" >&2
        exit 1
    fi
    if [ -n "$(git status --porcelain)" ]; then
        echo "REFUSING: the clone has uncommitted changes:" >&2
        git status --short >&2
        exit 1
    fi
}

require_gate() {
    if [ ! -f "$GATE_DIR/$EXPECTED_COMMIT.ok" ]; then
        echo "REFUSING: no passed test gate at $EXPECTED_COMMIT (submit slurm/tests_gate.sbatch first)" >&2
        exit 1
    fi
}

_remove_worktree() {
    local rc=$?
    git -C "$SLURM_SUBMIT_DIR" worktree remove --force --force "$WORKTREE" 2>/dev/null \
        || rmdir "$WORKTREE" 2>/dev/null || echo "worktree remove failed: $WORKTREE" >&2
    exit "$rc"
}

enter_worktree() {
    WORKTREE=$(mktemp -d "${TMPDIR:-/tmp}/offrl_wt_${SLURM_JOB_ID}_XXXX")
    trap _remove_worktree EXIT  # before the add: a failed add then leaves no empty directory
    git worktree add --detach --lock --reason "job $SLURM_JOB_ID on $(hostname)" -q "$WORKTREE" "$EXPECTED_COMMIT"
    cd "$WORKTREE" || exit 1
    export PYTHONPATH="$WORKTREE/src"
}
