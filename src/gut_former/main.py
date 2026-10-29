import hydra
from omegaconf import DictConfig

from settings.hydra import CONFIG_NAME, CONFIG_PATH, VERSION_BASE


@hydra.main(version_base=VERSION_BASE, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    task_func = hydra.utils.instantiate(cfg.task)
    task_func(cfg)


if __name__ == "__main__":
    main()
