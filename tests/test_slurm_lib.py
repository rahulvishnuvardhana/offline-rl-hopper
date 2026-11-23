"""Phase 6 spec S11: slurm/lib/common.sh's commit checkout, against a temporary git repo."""

import os
import shutil
import subprocess

import pytest

from offrl.runmeta import REPO_ROOT

COMMON = (REPO_ROOT / "slurm" / "lib" / "common.sh").as_posix()
pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("git")), reason="needs bash and git"
)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "clone"
    r.mkdir()
    git(r, "init", "-q")
    for i in (1, 2):
        (r / "f.txt").write_text(str(i))
        git(r, "add", "f.txt")
        git(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", str(i))
    return r


def sh(repo, root, commit, body):
    script = (
        f'export USER=t CHECKOUT_ROOT="{root.as_posix()}" EXPECTED_COMMIT={commit}; '
        f'cd "{repo.as_posix()}"; source "{COMMON}"; {body}'
    )
    # Not the test gate's own job: its SLURM_SUBMIT_DIR would point create_commit_checkout at the
    # real clone, and its EXPECTED_COMMIT and GRIDS are not this test's.
    gate = ("EXPECTED_COMMIT", "GRIDS", "CHECKOUT_ROOT", "PYTHONPATH")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_") and k not in gate}
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)


def test_checkout_is_created_once_and_reused(repo, tmp_path):
    head, root = git(repo, "rev-parse", "HEAD"), tmp_path / "src"
    for _ in range(2):  # idempotent: the second call accepts the existing checkout
        r = sh(repo, root, head, "create_commit_checkout && echo ok")
        assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    r = sh(repo, root, head, 'use_commit_checkout && git rev-parse --show-toplevel && echo "$PYTHONPATH"')
    assert r.returncode == 0, r.stderr
    toplevel, pythonpath = r.stdout.split()
    assert git(toplevel, "rev-parse", "HEAD") == head  # entered the checkout
    assert pythonpath.endswith(f"{head}/src")
    assert "locked" in git(repo, "worktree", "list", "--porcelain")


def test_use_refuses_missing_modified_or_moved_checkouts(repo, tmp_path):
    head, root = git(repo, "rev-parse", "HEAD"), tmp_path / "src"
    assert "no checkout" in sh(repo, root, head, "use_commit_checkout").stderr
    assert sh(repo, root, head, "create_commit_checkout").returncode == 0
    (root / head / "f.txt").write_text("changed")
    r = sh(repo, root, head, "use_commit_checkout")
    assert r.returncode == 1 and "uncommitted changes" in r.stderr
    git(root / head, "checkout", "-q", "--", "f.txt")
    git(root / head, "checkout", "-q", "--detach", "HEAD~1")  # moved off the pinned commit
    r = sh(repo, root, head, "use_commit_checkout")
    assert r.returncode == 1 and "not " + head in r.stderr
