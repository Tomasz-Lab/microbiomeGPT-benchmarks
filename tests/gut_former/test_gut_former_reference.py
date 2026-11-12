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

from functools import partial
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from gut_former.models.autoencoder import (
    ComposableAutoEncoder,
    GutFormer,
    GutFormerDecoder,
    GutFormerEncoder,
    GutFormerHead,
    GutFormerLatent,
    GutFormerTokenizer,
)
from gut_former.schema import DataSchema
from gut_former.utils.hydra_utils import cfg_to_plain_dict, instantiate
from settings.hydra import CONFIG_PATH

REFERENCE_PATH = Path(__file__).parents[1] / "data" / "gut_former_reference.pt"


@pytest.fixture(scope="module")
def reference() -> dict:
    return torch.load(REFERENCE_PATH, map_location="cpu", weights_only=True)


def build_small_gut_former(data_schema: DataSchema, embedding_dim: int, latent_size: int):
    """[purbancz] GutFormer's components at the reference's small sizes; GutFormer itself
    has its sizes frozen."""
    return ComposableAutoEncoder(
        data_schema=data_schema,
        tokenizer=partial(GutFormerTokenizer, embedding_dim=embedding_dim),
        encoder=GutFormerEncoder(embedding_dim),
        latent=GutFormerLatent(embedding_dim, latent_size),
        decoder=GutFormerDecoder(embedding_dim),
        head=GutFormerHead(embedding_dim),
    )


@torch.no_grad()
def test_frozen_gutformer_components_reproduce_saved_reference_outputs(reference):
    # Given
    with torch.random.fork_rng(devices=[]):
        # [purbancz] The reference was generated with torch.manual_seed(0). Intentionally different
        torch.manual_seed(42)
        model = build_small_gut_former(
            DataSchema.from_dict(reference["feature_names"]), **reference["model_config"]
        ).eval()
    model.load_state_dict(reference["state_dict"])  # [purbancz] strict: every weight
    inputs, expected = reference["inputs"], reference["outputs"]

    # When
    output = model(inputs)

    # Then
    torch.testing.assert_close(output.reconstructions["taxonomy"], expected["taxonomy"])
    torch.testing.assert_close(output.reconstructions["pathways"], expected["pathways"])
    torch.testing.assert_close(output.latent, expected["latent"])


@torch.no_grad()
def test_gut_former_config_builds_the_frozen_model_at_original_sizes(reference):
    # Given
    # [purbancz] Built from conf/model/gut_former.yaml as the training runner does
    model_schema = cfg_to_plain_dict(
        OmegaConf.load(Path(CONFIG_PATH) / "model" / "gut_former.yaml")
    )
    model_schema["data_schema"] = DataSchema.from_dict(reference["feature_names"]).to_config()

    # When
    model = instantiate(model_schema).eval()
    output = model(reference["inputs"])

    # Then
    assert isinstance(model, GutFormer)
    assert [type(component) for component in model.children()] == [
        GutFormerTokenizer,
        GutFormerEncoder,
        GutFormerLatent,
        GutFormerDecoder,
        GutFormerHead,
    ]
    assert model.tokenizer.feature_embeddings("taxonomy").shape[1] == 128
    assert output.latent.shape[1] == 64
