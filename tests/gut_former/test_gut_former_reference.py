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

from gut_former.models.autoencoder import (
    ComposableAutoEncoder,
    GutFormerDecoder,
    GutFormerEncoder,
    GutFormerHead,
    GutFormerLatent,
    GutFormerTokenizer,
)
from gut_former.schema import DataSchema

REFERENCE_PATH = Path(__file__).parents[1] / "data" / "gut_former_reference.pt"


@pytest.fixture(scope="module")
def reference() -> dict:
    return torch.load(REFERENCE_PATH, map_location="cpu", weights_only=True)


def build_gut_former(data_schema: DataSchema, embedding_dim: int, latent_size: int):
    return ComposableAutoEncoder(
        data_schema=data_schema,
        tokenizer=partial(GutFormerTokenizer, embedding_dim=embedding_dim),
        encoder=GutFormerEncoder(embedding_dim),
        latent=GutFormerLatent(embedding_dim, latent_size),
        decoder=GutFormerDecoder(embedding_dim),
        head=GutFormerHead(embedding_dim),
    )


@torch.no_grad()
def test_frozen_gutformer_model_reproduces_saved_reference_outputs(reference):
    # Given
    with torch.random.fork_rng(devices=[]):
        # [purbancz] The reference was generated with torch.manual_seed(0). Intentionally different
        torch.manual_seed(42)
        model = build_gut_former(
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
