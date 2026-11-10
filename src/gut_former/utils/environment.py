import os
import platform
import socket
from typing import Any

import lightning
import torch


def environment_state(trainer: lightning.pytorch.Trainer) -> dict[str, Any]:
    env: dict[str, Any] = {
        "os": platform.platform(),
        "host": socket.gethostname(),
        "torch": str(torch.__version__),
        "lightning": lightning.__version__,
        "accelerator": type(trainer.accelerator).__name__.removesuffix("Accelerator"),
        "devices": trainer.num_devices,
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    for key, variable in (("cluster", "SLURM_CLUSTER_NAME"), ("slurm_job_id", "SLURM_JOB_ID")):
        if variable in os.environ:
            env[key] = os.environ[variable]
    return env
