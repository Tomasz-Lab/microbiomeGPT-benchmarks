import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

load_dotenv()


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    task_func = hydra.utils.instantiate(cfg.task)
    task_func(cfg)


if __name__ == "__main__":
    main()
