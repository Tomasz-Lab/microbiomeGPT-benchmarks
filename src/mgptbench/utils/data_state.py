import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from data.splits import SampleId, Split

_CHUNK_SIZE = 1 << 20
SPLIT_FILENAME = "split.json"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids: Iterable[Any]) -> str:
    """Hash of a set of sample IDs; sorted first, so their order doesn't matter."""
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def ids_summary(ids: Sequence[SampleId]) -> dict[str, Any]:
    return {"n_samples": len(ids), "ids_sha256": ids_sha256(ids)}


# D: 0003
def data_state(files: Sequence[str | Path], split: Split, output_dir: str | Path) -> dict[str, Any]:
    """Input file hashes and split summary of the data used in the run.

    Full sample IDs of each split stay local in `<output_dir>/split.json`, so two runs'
    splits can be compared exactly; only counts and hashes are logged.
    """
    paths = [Path(file).resolve() for file in files]
    root = Path(os.path.commonpath(paths)) if len(paths) > 1 else paths[0].parent
    file_hashes = {str(path.relative_to(root)): file_sha256(path) for path in paths}
    parts = {"train": split.train, "val": split.val, "test": split.test}
    # Sorted strings, like in ids_sha256, so the file matches the hash and diffs cleanly
    split_ids = {name: sorted(map(str, ids)) for name, ids in parts.items()}
    (Path(output_dir) / SPLIT_FILENAME).write_text(json.dumps(split_ids, indent=2))
    return {
        "root": str(root),
        "files": file_hashes,
        "n_samples": len({*split.train, *split.val, *split.test}),
        "splits": {name: ids_summary(ids) for name, ids in parts.items()},
        "split_ids": SPLIT_FILENAME,
    }
