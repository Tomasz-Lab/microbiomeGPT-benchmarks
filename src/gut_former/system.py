import logging
from pathlib import Path

import hydra
import lightning.pytorch as L
import torch
from torch import nn
from torchmetrics import MeanMetric, MeanSquaredError, MetricCollection

from gut_former.schema import DataSchema
from gut_former.utils.hydra_utils import instantiate

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


class AutoEncoderSystem(L.LightningModule):
    """Training, validation and prediction steps for a taxonomy/pathways autoencoder.

    `model_config` and `data_schema` are stored in the checkpoint so `load_system` can rebuild
    the network without the original data or Hydra config.
    """

    def __init__(
        self,
        model_schema: dict,
        loss: dict | None = None,
        optimizer: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = instantiate(model_schema)
        self.loss_fn = instantiate(loss)
        self.optimizer_fn = instantiate(optimizer)

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
        if self.optimizer_fn is None:
            raise ValueError("An optimizer is required for training.")
        return self.optimizer_fn(self.parameters())


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


def build_model(model_config, data_schema: DataSchema) -> nn.Module:
    """Instantiate the model config with the data schema."""
    # Not instantiate(..., data_schema=...): Hydra would merge the dataclass into the config
    return hydra.utils.instantiate(model_config, _partial_=True)(data_schema=data_schema)


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


def load_system(
    path: str | Path,
    loss_fn: nn.Module | None = None,
    map_location="cpu",
    system_class: type[L.LightningModule] = AutoEncoderSystem,
) -> L.LightningModule:
    """Rebuild the network from the schema stored in a Lightning checkpoint and load its weights."""
    hparams = torch.load(path, map_location=map_location, weights_only=True)["hyper_parameters"]
    model = build_model(hparams["model_config"], DataSchema.from_dict(hparams["data_schema"]))
    return system_class.load_from_checkpoint(
        path, model=model, loss_fn=loss_fn, map_location=map_location
    )
