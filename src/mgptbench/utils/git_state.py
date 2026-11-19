import logging
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PATCH_FILENAME = "code.patch"
# Untracked files here are code, so they go into the patch; others (e.g. data/) are ignored
UNTRACKED_CODE_DIRS = ("src", "conf")


def _git(*args: str, cwd: Path) -> str:
    # -C, not the working directory: under submitit the job may start elsewhere
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    ).stdout


def _new_file_diff(root: Path, path: str) -> str:
    # --no-index exits with 1 when the files differ, which is always the case here
    return subprocess.run(
        ["git", "-C", str(root), "diff", "--no-index", "--binary", "/dev/null", path],
        capture_output=True,
        text=True,
    ).stdout


# D: 0003
def git_state(output_dir: str | Path) -> dict[str, Any]:
    """Commit, branch and uncommitted changes of the code that runs.

    Uncommitted changes, including new files in `src/` and `conf/`, are saved to
    `<output_dir>/code.patch`, so the code can be restored with
    `git checkout <commit> && git apply code.patch`.
    """
    try:
        root = Path(_git("rev-parse", "--show-toplevel", cwd=Path(__file__).parent).strip())
        commit = _git("rev-parse", "HEAD", cwd=root).strip()
        branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root).strip()
        untracked = _git(
            "ls-files", "--others", "--exclude-standard", "--", *UNTRACKED_CODE_DIRS, cwd=root
        ).split()
        patch = _git("diff", "HEAD", "--binary", cwd=root)
        patch += "".join(_new_file_diff(root, path) for path in untracked)
    except (OSError, subprocess.CalledProcessError) as error:
        log.warning("Git state not available, logging it as unknown: %s", error)
        return {"commit": "unknown", "branch": "unknown", "dirty": None, "patch": None}

    if not patch:
        return {"commit": commit, "branch": branch, "dirty": False, "patch": None}

    (Path(output_dir) / PATCH_FILENAME).write_text(patch)
    return {"commit": commit, "branch": branch, "dirty": True, "patch": PATCH_FILENAME}
