#!/bin/bash
# Copy-out before the scratch purge (Phase 6 spec Section 12): rsync <src> to a new <dest>, then
# check every file of <dest> against <src> by SHA256SUMS. Remove a grid's checkout once no job
# of yours is left:
#   scripts/archive_runs.sh <src> <dest>
#   scripts/archive_runs.sh --remove-checkout <sha>
# (--remove-checkout refuses inside an salloc or srun session too: that session is a job.)
set -euo pipefail
if [ "${1:-}" = --remove-checkout ]; then
    sha=${2:?give the commit whose checkout to remove}
    wt="/scratch/$USER/src/$sha"
    if [ ! -d "$wt" ]; then
        echo "REFUSING: no checkout $wt" >&2
        exit 1
    fi
    if ! jobs=$(squeue -u "$USER" -h); then
        echo "REFUSING: squeue failed; cannot tell whether jobs still use $wt" >&2
        exit 1
    fi
    if [ -n "$jobs" ]; then
        echo "REFUSING: jobs of yours are still queued or running" >&2
        exit 1
    fi
    if ! common=$(git -C "$wt" rev-parse --path-format=absolute --git-common-dir); then  # the owning clone
        echo "REFUSING: $wt is not a git checkout" >&2
        exit 1
    fi
    git -C "$(dirname "$common")" worktree remove --force --force "$wt"
    echo "removed checkout $sha"
    exit 0
fi
src=${1:?give the source directory} dest=${2:?give a new destination directory}
if [ -e "$dest" ]; then
    echo "REFUSING: $dest exists; archive into a new directory" >&2
    exit 1
fi
mkdir -p "$dest"
dest=$(cd "$dest" && pwd)  # absolute: the check below runs from $src
rsync -a "$src/" "$dest/"
(cd "$dest" && find . -type f ! -path ./SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
(cd "$src" && sha256sum -c --quiet "$dest/SHA256SUMS")
echo "archived $(wc -l < "$dest/SHA256SUMS") files from $src to $dest; every one matches"
