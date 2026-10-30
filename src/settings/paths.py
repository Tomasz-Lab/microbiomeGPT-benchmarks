import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def find_project_root(marker: str = "pyproject.toml") -> Path:
    start_dir = Path(__file__).resolve().parent

    for directory in (start_dir, *start_dir.parents):
        if (directory / marker).is_file():
            return directory

    raise FileNotFoundError(
        f"Could not find project root containing {marker!r} starting from {start_dir}"
    )


PROJECT_ROOT = find_project_root()

DATASET_DIR = (
    Path(
        os.environ.get(
            "DATASET_DIR",
            str(PROJECT_ROOT / "data"),
        )
    )
    .expanduser()
    .resolve()
)
