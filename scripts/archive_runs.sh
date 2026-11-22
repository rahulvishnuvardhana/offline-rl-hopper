#!/bin/bash
# Copy-out before the scratch purge (Phase 6 spec Section 12): rsync <src> to a new <dest>, then
# check every file of <dest> against <src> by SHA256SUMS. Remove a grid's checkout once no job
# of yours is left:
#   scripts/archive_runs.sh <src> <dest>
#   scripts/archive_runs.sh --remove-checkout <sha>
set -euo pipefail
if [ "${1:-}" = --remove-checkout ]; then
    sha=${2:?give the commit whose checkout to remove}
    if [ -n "$(squeue -u "$USER" -h)" ]; then
        echo "REFUSING: jobs of yours are still queued or running" >&2
        exit 1
    fi
    git -C "$HOME/offline-rl-hopper" worktree remove --force --force "/scratch/$USER/src/$sha"
    echo "removed checkout $sha"
    exit 0
fi
src=${1:?give the source directory} dest=${2:?give a new destination directory}
if [ -e "$dest" ]; then
    echo "REFUSING: $dest exists; archive into a new directory" >&2
    exit 1
fi
mkdir -p "$dest"
rsync -a "$src/" "$dest/"
(cd "$dest" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
(cd "$src" && sha256sum -c --quiet "$dest/SHA256SUMS")
echo "archived $(wc -l < "$dest/SHA256SUMS") files from $src to $dest; every one matches"
