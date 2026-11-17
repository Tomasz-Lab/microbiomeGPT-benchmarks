import pytest

from gut_former.system import AutoEncoderSystem


def test_rejects_a_model_that_is_not_an_autoencoder():
    # Given
    model_schema = {"_target_": "torch.nn.Identity"}

    # When / Then
    with pytest.raises(TypeError, match="AutoEncoderModel"):
        AutoEncoderSystem(model_schema=model_schema)
