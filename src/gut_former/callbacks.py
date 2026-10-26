import logging
from pathlib import Path

import lightning.pytorch as L
import pandas as pd

log = logging.getLogger(__name__)

# Logged metric name -> column in the training stats CSV (the legacy format used by notebooks)
STATS_COLUMNS = {
    "train_loss": "train_loss",
    "train_taxonomy_mse": "train_taxonomy_mse",
    "train_pathways_mse": "train_pathways_mse",
    "val_loss": "test_loss",
    "val_taxonomy_mse": "test_taxonomy_mse",
    "val_pathways_mse": "test_pathways_mse",
}


class EpochStatsWriter(L.Callback):
    """Write one row of epoch-level metrics per epoch, rewriting the CSV after each epoch.

    The history is stored in checkpoints, so a resumed run writes all epochs, not only new ones.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.history: list[dict] = []

    def state_dict(self) -> dict:
        return {"history": self.history}

    def load_state_dict(self, state_dict: dict):
        self.history = list(state_dict["history"])

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics
        row = {"epoch": trainer.current_epoch}
        for metric, column in STATS_COLUMNS.items():
            row[column] = metrics[metric].item() if metric in metrics else float("nan")
        self.history.append(row)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        stats = pd.DataFrame(self.history, columns=["epoch", *STATS_COLUMNS.values()])
        stats.to_csv(self.path, index=False)

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            log.info("Training stats saved: %s", self.path)
