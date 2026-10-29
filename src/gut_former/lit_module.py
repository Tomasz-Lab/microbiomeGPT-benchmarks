import logging
from pathlib import Path

import hydra
import lightning.pytorch as L
import torch
from torch import nn
from torch.nn.functional import mse_loss

log = logging.getLogger(__name__)


class GutFormerModule(L.LightningModule):
    """Training, validation and prediction steps for a taxonomy/pathways autoencoder.

    `model_config` and `model_kwargs` are stored in the checkpoint so `load_module` can rebuild
    the network without the original data or Hydra config.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module | None = None,
        optimizer=None,
        model_config: dict | None = None,
        model_kwargs: dict | None = None,
        feature_names: dict[str, list[str]] | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_factory = optimizer
        self.save_hyperparameters(ignore=["model", "loss_fn", "optimizer"])

    def forward(self, xt, xp):
        # Dataset yields (taxonomy, pathways); the model takes (pathways, taxonomy)
        return self.model(xp, xt)

    def _shared_step(self, batch, stage: str):
        if self.loss_fn is None:
            raise ValueError("A loss function is required for training and validation.")
        xt, xp = batch
        xt_pred, xp_pred, _ = self(xt, xp)
        loss = self.loss_fn(xt, xt_pred, xp, xp_pred)
        self.log_dict(
            {
                f"{stage}_loss": loss,
                f"{stage}_taxonomy_mse": mse_loss(xt_pred, xt),
                f"{stage}_pathways_mse": mse_loss(xp_pred, xp),
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=xt.size(0),
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


def check_feature_names(expected: dict[str, list[str]], actual: dict[str, list[str]]):
    for kind, names in expected.items():
        if actual[kind] != names:
            raise ValueError(
                f"{kind} features differ from the checkpoint: expected {len(names)} columns, "
                f"got {len(actual[kind])} (names or order do not match)."
            )


def verify_checkpoint_features(path: str | Path, feature_names: dict[str, list[str]]):
    """Fail if the data columns differ from those the checkpoint was trained on."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = checkpoint.get("hyper_parameters", {}).get("feature_names")
    if expected is None:
        log.warning(
            "Checkpoint %s has no feature names (legacy format); "
            "feature names and order cannot be verified.",
            path,
        )
        return
    check_feature_names(expected, feature_names)


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
) -> GutFormerModule:
    """Rebuild the network from the shape stored in a Lightning checkpoint and load its weights."""
    hparams = torch.load(path, map_location=map_location, weights_only=True)["hyper_parameters"]
    model_builder = hydra.utils.instantiate(hparams["model_config"])
    model = model_builder(**hparams["model_kwargs"])
    return GutFormerModule.load_from_checkpoint(
        path, model=model, loss_fn=loss_fn, map_location=map_location
    )
