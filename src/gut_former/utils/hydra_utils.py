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
