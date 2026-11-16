"""Git provenance guards shared by the scripts that write committed results."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def dirty_paths(root: Path = REPO_ROOT) -> list[str]:
    """Every uncommitted or untracked path. No exemptions: stale outputs or an edited input
    file would otherwise slip into a run that claims to be clean."""
    out = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=root
    )
    return [line[3:] for line in out.stdout.splitlines() if line.strip()]


def committed_bytes(path: Path, root: Path = REPO_ROOT) -> bytes:
    """The file exactly as committed at HEAD, i.e. the bytes a fresh clone gets.

    The disk copy can differ while `git status` says clean: .gitattributes normalizes line
    endings, so a CRLF file on disk matches an LF blob. Provenance must hash the blob.
    """
    rel = Path(path).resolve().relative_to(root).as_posix()
    return subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, check=True, cwd=root).stdout
