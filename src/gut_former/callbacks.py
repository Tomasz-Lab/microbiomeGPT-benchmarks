import logging
from pathlib import Path

import lightning.pytorch as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import BasePredictionWriter

log = logging.getLogger(__name__)


def _device_summary(trainer) -> str:
    accelerator = type(trainer.accelerator).__name__.removesuffix("Accelerator")
    return f"{accelerator}, {trainer.num_devices} device(s)"


def _num_samples(dataloaders) -> int:
    if dataloaders is None:
        return 0
    if not isinstance(dataloaders, list | tuple):
        dataloaders = [dataloaders]
    return sum(len(dataloader.dataset) for dataloader in dataloaders)


class RunLogger(L.Callback):
    """Log the start and end of training and inference runs."""

    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        resumed = f", resuming at epoch {trainer.current_epoch}" if trainer.current_epoch else ""
        log.info(
            "Training started: %d epochs on %s%s; %d train / %d validation samples",
            trainer.max_epochs,
            _device_summary(trainer),
            resumed,
            _num_samples(trainer.train_dataloader),
            _num_samples(trainer.val_dataloaders),
        )

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback is None:
            log.info("Training completed (no checkpoint callback configured).")
            return
        log.info(
            "Training completed.\nSaved checkpoints:\n  - Best: %s\n  - Last: %s",
            checkpoint_callback.best_model_path,
            checkpoint_callback.last_model_path,
        )

    def on_predict_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            log.info(
                "Inference started: %d samples on %s",
                _num_samples(trainer.predict_dataloaders),
                _device_summary(trainer),
            )

    def on_predict_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            log.info("Inference completed.")


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
