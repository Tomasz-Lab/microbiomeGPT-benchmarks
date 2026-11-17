import logging
from typing import Any

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from settings.hydra import ALLOWED_TARGET_PREFIXES

log = logging.getLogger(__name__)


def cfg_to_plain_dict(cfg: DictConfig) -> dict[str, Any] | None:
    """Resolved copy of a config node as plain Python types."""
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(container, dict):
        raise TypeError(f"Expected a mapping config, got {type(container).__name__}.")
    return container


# this wraper is need only because later
# if we want to replace hydra with another config system,
# we can do it here easily ~Fasamik39
def instantiate(config: DictConfig | dict) -> Any:
    """Build the object described by a config with a `_target_` key."""
    _check_targets(config)
    return hydra.utils.instantiate(config)


def _check_targets(config: Any) -> None:
    if isinstance(config, dict | DictConfig):
        target = config.get("_target_")
        if target is not None and not str(target).startswith(ALLOWED_TARGET_PREFIXES):
            log.error("Model not loaded: _target_ %r is not in %s", target, ALLOWED_TARGET_PREFIXES)
            raise ValueError(f"_target_ {target!r} is not allowed.")
        children = config.values()
    elif isinstance(config, list | ListConfig):
        children = config
    else:
        return
    for child in children:
        _check_targets(child)


def load_from_checkpoint(
    config: DictConfig | dict, checkpoint_path: str, map_location: str = "cpu"
) -> Any:
    """Restore the class named by a config's `_target_` from a checkpoint.

    The class rebuilds itself from the hyperparameters stored in the checkpoint, so only
    `_target_` is read from the config.
    """
    target_class = hydra.utils.get_class(config["_target_"])
    return target_class.load_from_checkpoint(checkpoint_path, map_location=map_location)
