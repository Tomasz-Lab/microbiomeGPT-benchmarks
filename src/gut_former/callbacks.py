import logging
from pathlib import Path

import lightning.pytorch as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import BasePredictionWriter

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


class PredictionWriter(BasePredictionWriter):
    """Write each configured prediction output to its own CSV, indexed by sample ID.

    Outputs named after a data modality get that modality's feature names as columns; others
    (e.g. the latent space) get `{column_prefix}{i}`.
    """

    def __init__(self, output_dir: str | Path, filenames: dict[str, str], column_prefix: str = "z"):
        super().__init__(write_interval="epoch")
        self.output_dir = Path(output_dir)
        self.filenames = dict(filenames)
        self.column_prefix = column_prefix

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        if not predictions:
            raise RuntimeError("Prediction returned no batches; is the inference dataset empty?")

        # Check every configured output first, so a typo doesn't leave partial results
        missing = sorted(set(self.filenames) - set(predictions[0]))
        if missing:
            raise ValueError(
                f"The model predicts no {missing} output(s); available: {sorted(predictions[0])}."
            )

        datamodule = trainer.datamodule
        feature_names = datamodule.data_schema().feature_names
        index = datamodule.predict_index

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for key, filename in self.filenames.items():
            values = torch.cat([batch[key] for batch in predictions]).cpu().numpy()
            columns = feature_names.get(key) or [
                f"{self.column_prefix}{i}" for i in range(values.shape[1])
            ]
            path = self.output_dir / filename
            pd.DataFrame(values, index=index, columns=list(columns)).to_csv(path)
            log.info("Saved %s predictions: %s", key, path)
