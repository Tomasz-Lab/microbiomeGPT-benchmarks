import logging
import shutil
from pathlib import Path

import lightning.pytorch as L
import torch

log = logging.getLogger(__name__)


def prepare_resume_checkpoint(trainer: L.Trainer, resume_from: str | None) -> str | None:
    if resume_from is None:
        return None
    copied = copy_checkpoints_to_this_run(trainer, resume_from)
    log.info("Resuming from a copy of %s: %s", resume_from, copied)
    return copied


def copy_checkpoints_to_this_run(trainer: L.Trainer, resume_from: str) -> str:
    """Copy the resumed checkpoint and its best checkpoint into this run's checkpoint folder.

    The resumed run is only read, never written. ModelCheckpoint restores its best score only when
    the folder stored in the checkpoint is the one it saves to, so the copy stores this run's
    folder and paths. Then best.ckpt changes only when an epoch beats the resumed run's best.
    """
    this_run_dir = Path(trainer.checkpoint_callback.dirpath)
    if Path(resume_from).resolve().parent == this_run_dir.resolve():
        raise ValueError(
            f"task.resume_from is in this run's checkpoint folder ({this_run_dir}); resume in a new "
            "output folder, so the resumed run is not overwritten."
        )

    checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)

    for callback in trainer.checkpoint_callbacks:
        state = checkpoint["callbacks"][callback.state_key]
        new_dir = Path(callback.dirpath)
        new_dir.mkdir(parents=True, exist_ok=True)

        if callback.monitor is not None and state["best_model_path"]:
            shutil.copy2(state["best_model_path"], _in_folder(state["best_model_path"], new_dir))

        state["dirpath"] = callback.dirpath
        for key in ("best_model_path", "kth_best_model_path", "last_model_path"):
            state[key] = _in_folder(state[key], new_dir)
        state["best_k_models"] = {
            _in_folder(path, new_dir): score for path, score in state["best_k_models"].items()
        }

    copied = this_run_dir / Path(resume_from).name
    torch.save(checkpoint, copied)
    return str(copied)


def _in_folder(path: str, folder: Path) -> str:
    """The same file name in another folder; an empty path (no checkpoint yet) stays empty."""
    return str(folder / Path(path).name) if path else path
