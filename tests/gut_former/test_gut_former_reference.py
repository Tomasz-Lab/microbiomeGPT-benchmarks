"""[purbancz] The recreated (and frozen) GutFormer components must keep computing what
the original Gut-FORMer model computed.

`tests/data/gut_former_reference.pt` was generated once from the original monolithic
`BacteriaModel` (the commit is stored in the file as `source_commit`). It holds a small model's
weights, a batch of inputs and the outputs that model produced in eval mode. The weights were
then renamed to the GutFormer component names; the parameters that the original class registered
but never used in forward (`encoder_layer`, `final_encoder_layer`, `decoder_layer`,
`latent_norm`) are not included.

Note also that "gut_former" can stay in the name of this file after the package name refactor.
"""

import copy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from gut_former.schema import DataSchema
from gut_former.utils.hydra_utils import cfg_to_plain_dict, instantiate
from settings.hydra import CONFIG_PATH

REFERENCE_PATH = Path(__file__).parents[1] / "data" / "gut_former_reference.pt"


@pytest.fixture(scope="module")
def reference() -> dict:
    return torch.load(REFERENCE_PATH, map_location="cpu", weights_only=True)


@pytest.fixture(scope="module")
def build_gut_former():
    """[purbancz] Builds GutFormer from conf/model/gut_former.yaml, as the training runner does,
    so the config is checked too. Sizes are added only to build a small model."""
    model_config = OmegaConf.load(Path(CONFIG_PATH) / "model" / "gut_former.yaml")

    def build(data_schema: DataSchema, seed: int = 0, **sizes: int):
        config = copy.deepcopy(model_config)
        for name, value in sizes.items():
            config[name] = value
        model_schema = cfg_to_plain_dict(config)
        model_schema["data_schema"] = data_schema.to_config()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return instantiate(model_schema).eval()

    return build


@torch.no_grad()
def test_frozen_gutformer_model_reproduces_saved_reference_outputs(reference, build_gut_former):
    # Given
    # [purbancz] The reference was generated with torch.manual_seed(0). Intentionally different
    model = build_gut_former(
        DataSchema.from_dict(reference["feature_names"]), **reference["model_config"], seed=42
    )
    model.load_state_dict(reference["state_dict"])  # [purbancz] strict: every weight
    inputs, expected = reference["inputs"], reference["outputs"]

    # When
    output = model(inputs)

    # Then
    torch.testing.assert_close(output.reconstructions["taxonomy"], expected["taxonomy"])
    torch.testing.assert_close(output.reconstructions["pathways"], expected["pathways"])
    torch.testing.assert_close(output.latent, expected["latent"])


@torch.no_grad()
def test_frozen_gutformer_model_has_the_original_sizes(reference, build_gut_former):
    # Given
    # [purbancz] The reference uses small sizes, so the frozen ones are checked separately
    model = build_gut_former(DataSchema.from_dict(reference["feature_names"]))

    # When
    output = model(reference["inputs"])

    # Then
    assert model.tokenizer.feature_embeddings("taxonomy").shape[1] == 128
    assert output.latent.shape[1] == 64
