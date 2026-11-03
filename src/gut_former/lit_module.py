import logging
from pathlib import Path

import hydra
import lightning.pytorch as L
import torch
from torch import nn
from torchmetrics import MeanMetric, MeanSquaredError, MetricCollection

from gut_former.schema import DataSchema

log = logging.getLogger(__name__)


def _stage_metrics() -> MetricCollection:
    # Each metric gets its own inputs, so no shared compute groups
    return MetricCollection(
        {
            "loss": MeanMetric(),
            "taxonomy_mse": MeanSquaredError(),
            "pathways_mse": MeanSquaredError(),
        },
        compute_groups=False,
    )


class AutoEncoderModule(L.LightningModule):
    """Training, validation and prediction steps for a taxonomy/pathways autoencoder.

    `model_config` and `data_schema` are stored in the checkpoint so `load_module` can rebuild
    the network without the original data or Hydra config.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | None = None,
        optimizer=None,
        model_config: dict | None = None,
        data_schema: dict[str, list[str]] | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_factory = optimizer
        self.save_hyperparameters(ignore=["model", "loss_fn", "optimizer"])

        # Separate attributes, not ModuleDict keys: "train" clashes with nn.Module.train
        self.train_metrics = _stage_metrics()
        self.val_metrics = _stage_metrics()

    def forward(self, xt, xp):
        # Dataset yields (taxonomy, pathways); the model takes (pathways, taxonomy)
        return self.model(xp, xt)

    def _shared_step(self, batch, stage: str):
        if self.loss_fn is None:
            raise ValueError("A loss function is required for training and validation.")
        xt, xp = batch
        xt_pred, xp_pred, _ = self(xt, xp)
        loss = self.loss_fn(xt, xt_pred, xp, xp_pred)

        # Metric objects accumulate over the epoch and sync across devices themselves;
        # Lightning calls compute() and reset() at epoch end
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics["loss"].update(loss.detach(), weight=xt.size(0))
        metrics["taxonomy_mse"].update(xt_pred, xt)
        metrics["pathways_mse"].update(xp_pred, xp)
        self.log_dict(
            {f"{stage}_{name}": metric for name, metric in metrics.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=stage == "val",
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        xt, xp = batch
        xt_pred, xp_pred, latent = self(xt, xp)
        return {
            "taxonomy": xt_pred.squeeze(-1),
            "pathways": xp_pred.squeeze(-1),
            "latent": latent,
        }

    def configure_optimizers(self):
        if self.optimizer_factory is None:
            raise ValueError("An optimizer is required for training.")
        return self.optimizer_factory(self.parameters())


def resolve_accelerator(accelerator: str) -> str:
    """Map "auto" to CUDA or CPU; Lightning's own "auto" picks MPS, which lacks cdist backward."""
    if accelerator != "auto":
        return accelerator
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_lightning_checkpoint(path: str | Path) -> bool:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return "hyper_parameters" in checkpoint


def check_schema(expected: DataSchema, actual: DataSchema):
    for modality, names in expected.feature_names.items():
        actual_names = actual.feature_names.get(modality)
        if actual_names is None:
            raise ValueError(f"The data has no {modality} features, but the checkpoint needs them.")
        if actual_names != names:
            raise ValueError(
                f"{modality} features differ from the checkpoint: expected {len(names)} columns, "
                f"got {len(actual_names)} (names or order do not match)."
            )


def verify_checkpoint_schema(path: str | Path, data_schema: DataSchema):
    """Fail if the data columns differ from those the checkpoint was trained on."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = checkpoint.get("hyper_parameters", {}).get("data_schema")
    if expected is None:
        log.warning(
            "Checkpoint %s has no data schema (legacy format); "
            "feature names and order cannot be verified.",
            path,
        )
        return
    check_schema(DataSchema.from_dict(expected), data_schema)


def load_model_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Model weights from a Lightning checkpoint or a legacy bare state_dict file."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "hyper_parameters" not in checkpoint:
        state_dict: dict[str, torch.Tensor] = checkpoint
        return state_dict
    return {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }


def load_module(
    path: str | Path, loss_fn: nn.Module | None = None, map_location="cpu"
) -> AutoEncoderModule:
    """Rebuild the network from the schema stored in a Lightning checkpoint and load its weights."""
    hparams = torch.load(path, map_location=map_location, weights_only=True)["hyper_parameters"]
    model_builder = hydra.utils.instantiate(hparams["model_config"])
    model = model_builder(data_schema=DataSchema.from_dict(hparams["data_schema"]))
    return AutoEncoderModule.load_from_checkpoint(
        path, model=model, loss_fn=loss_fn, map_location=map_location
    )
