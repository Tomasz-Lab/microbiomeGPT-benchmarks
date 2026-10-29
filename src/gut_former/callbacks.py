import logging
from pathlib import Path

import lightning.pytorch as L
import pandas as pd

log = logging.getLogger(__name__)


class EpochStatsWriter(L.Callback):
    """Write one row of the logged epoch-level metrics per epoch, rewriting the CSV each epoch.

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

        row = {"epoch": trainer.current_epoch}
        row.update({name: value.item() for name, value in sorted(trainer.callback_metrics.items())})
        self.history.append(row)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.history).to_csv(self.path, index=False)

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            log.info("Training stats saved: %s", self.path)
