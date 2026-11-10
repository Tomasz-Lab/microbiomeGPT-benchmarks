"""[purbancz] The architecture for the planned frozen Gut-FORMer model must keep computing
what the original model computed.

`tests/data/gut_former_v1_reference.pt` was generated once from the original monolithic
`BacteriaModel` (the commit is stored in the file as `source_commit`). It holds a small model's
weights, a batch of inputs and the outputs that model produced in eval mode. The template
transformer layers (`encoder_layer`, `final_encoder_layer`, `decoder_layer`) are registered
in that class but never called in forward, so they were left out of the saved weights.
"""

from pathlib import Path

import pytest
import torch

from gut_former.model import BacteriaModel
from gut_former.schema import DataSchema

REFERENCE_PATH = Path(__file__).parents[1] / "data" / "gut_former_v1_reference.pt"
UNUSED_PREFIXES = ("encoder_layer.", "final_encoder_layer.", "decoder_layer.")


@pytest.fixture(scope="module")
def reference() -> dict:
    return torch.load(REFERENCE_PATH, map_location="cpu", weights_only=True)


@torch.no_grad()
def test_frozen_gutformer_model_reproduces_saved_reference_outputs(reference):
    # Given
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(
            42
        )  # [purbancz] The reference was trained with torch.manual_seed(0). Intentionally different
        model = BacteriaModel(
            DataSchema.from_dict(reference["feature_names"]), **reference["model_config"]
        ).eval()
    load_result = model.load_state_dict(reference["state_dict"], strict=False)
    inputs, expected = reference["inputs"], reference["outputs"]

    # When
    taxonomy, pathways, latent = model(inputs["pathways"], inputs["taxonomy"])

    # Then
    assert load_result.unexpected_keys == []
    assert all(key.startswith(UNUSED_PREFIXES) for key in load_result.missing_keys)
    torch.testing.assert_close(taxonomy, expected["taxonomy"])
    torch.testing.assert_close(pathways, expected["pathways"])
    torch.testing.assert_close(latent, expected["latent"])
