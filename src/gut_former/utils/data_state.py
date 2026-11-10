import hashlib
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

_CHUNK_SIZE = 1 << 20


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids: Iterable[Any]) -> str:
    """Hash of a set of sample IDs; sorted first, so their order doesn't matter."""
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def data_state(files: Sequence[str | Path], splits: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    paths = [Path(file).resolve() for file in files]
    root = Path(os.path.commonpath(paths)) if len(paths) > 1 else paths[0].parent
    file_hashes = {str(path.relative_to(root)): file_sha256(path) for path in paths}
    return {
        "root": str(root),
        "files": file_hashes,
        "n_samples": len({sample for ids in splits.values() for sample in ids}),
        "splits": {
            split: {"n_samples": len(ids), "ids_sha256": ids_sha256(ids)}
            for split, ids in splits.items()
        },
    }
