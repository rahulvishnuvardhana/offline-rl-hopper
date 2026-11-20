# Sourced by the job scripts (from the repo root, after `cd "$SLURM_SUBMIT_DIR"`).
#
# pin_commit: refuse to run unless the shared clone is at the commit pinned at submission and
# clean, so a `git pull` between submission and start cannot change what a task runs. Submit with
#   sbatch --export=ALL,EXPECTED_COMMIT=$(git rev-parse HEAD) slurm/<job>.sbatch
#
# enter_worktree: for jobs that write into the repo (mixture manifests, the analysis report),
# work in a throwaway worktree of the pinned commit under $TMPDIR, so the shared clone is never
# written to and runs starting meanwhile do not see a dirty tree. Removed on exit.

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

enter_worktree() {
    WORKTREE="${TMPDIR:-/tmp}/offrl_wt_${SLURM_JOB_ID}"
    git worktree add --detach -q "$WORKTREE" "$EXPECTED_COMMIT"
    trap 'git -C "$SLURM_SUBMIT_DIR" worktree remove --force "$WORKTREE"' EXIT
    cd "$WORKTREE"
    export PYTHONPATH="$WORKTREE/src"
}
