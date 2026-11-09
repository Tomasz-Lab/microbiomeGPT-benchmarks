from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


def cfg_to_plain_dict(cfg: DictConfig) -> dict[str, Any] | None:
    """Resolved copy of a config node as plain Python types."""
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(container, dict):
        raise TypeError(f"Expected a mapping config, got {type(container).__name__}.")
    return container


# this wraper is need only because later
# if we want to replace hydra with another config system,
# we can do it here easily
def instantiate(config: DictConfig | dict) -> Any:
    """Build the object described by a config with a `_target_` key."""
    return hydra.utils.instantiate(config)


def load_from_checkpoint(
    config: DictConfig | dict, checkpoint_path: str, map_location: str = "cpu"
) -> Any:
    """Restore the class named by a config's `_target_` from a checkpoint.

    The class rebuilds itself from the hyperparameters stored in the checkpoint, so only
    `_target_` is read from the config.
    """
    target_class = hydra.utils.get_class(config["_target_"])
    return target_class.load_from_checkpoint(checkpoint_path, map_location=map_location)
