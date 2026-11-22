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

GATE_DIR="/scratch/$USER/gates"

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
    cd "$WORKTREE"
    export PYTHONPATH="$WORKTREE/src"
}
