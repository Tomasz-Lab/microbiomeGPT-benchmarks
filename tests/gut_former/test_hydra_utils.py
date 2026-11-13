import logging

import pytest

from gut_former.utils.hydra_utils import instantiate


def test_instantiate_builds_allowed_target():
    # When
    optimizer = instantiate({"_target_": "torch.optim.Adam", "_partial_": True, "lr": 0.1})

    # Then
    assert optimizer.keywords["lr"] == 0.1


@pytest.mark.parametrize(
    "config",
    [
        {"_target_": "os.system", "command": "echo hi"},
        {"_target_": "gut_former.loss.CompositeLoss", "inner": [{"_target_": "os.system"}]},
    ],
)
def test_instantiate_rejects_and_logs_disallowed_target(config, caplog):
    # When / Then
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="os.system"):
        instantiate(config)
    assert "Model not loaded" in caplog.text
