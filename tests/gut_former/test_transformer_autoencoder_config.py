import pytest
import torch
from hydra import compose, initialize_config_dir

from gut_former.models.autoencoder import ComposableAutoEncoder, GutFormer
from gut_former.models.components import (
    FeatureEmbeddingTokenizer,
    JointTransformerEncoder,
    LinearHead,
    MeanPooling,
    PooledLinearLatent,
    SharedTransformerDecoder,
)
from gut_former.schema import DataSchema
from gut_former.utils.hydra_utils import cfg_to_plain_dict, instantiate
from settings.hydra import CONFIG_NAME, CONFIG_PATH, VERSION_BASE

BATCH_SIZE = 2


@pytest.fixture
def data_schema() -> DataSchema:
    return DataSchema.from_dict(
        {"taxonomy": ["taxon_0", "taxon_1"], "pathways": ["pathway_0", "pathway_1", "pathway_2"]}
    )


def build_model(data_schema: DataSchema, *overrides: str) -> ComposableAutoEncoder:
    with initialize_config_dir(config_dir=CONFIG_PATH, version_base=VERSION_BASE):
        cfg = compose(
            config_name=CONFIG_NAME,
            overrides=[
                "experiment=gut_former_training",
                "model=transformer_autoencoder",
                *overrides,
            ],
        )
    model_schema = cfg_to_plain_dict(cfg.model) | {"data_schema": data_schema.to_config()}
    model: ComposableAutoEncoder = instantiate(model_schema)
    return model


def stack_settings(stack) -> list[tuple[int, float, int]]:
    return [
        (layer.self_attn.num_heads, layer.dropout.p, layer.linear1.out_features)
        for layer in stack.layers
    ]


def test_config_builds_the_components_at_gutformer_values(data_schema):
    # Given / When
    model = build_model(data_schema)

    # Then
    assert [type(child) for child in model.children()] == [
        FeatureEmbeddingTokenizer,
        JointTransformerEncoder,
        PooledLinearLatent,
        SharedTransformerDecoder,
        LinearHead,
    ]
    assert isinstance(model.latent.pooling, MeanPooling)
    assert model.tokenizer.embedding_dim == 128
    assert model.latent.latent_size == 64
    assert [stack_settings(stack) for stack in model.encoder.stacks] == [
        [(2, 0.05, 2048)],
        [(128, 0.1, 2048)],
    ]
    assert stack_settings(model.decoder.stack) == [(2, 0.1, 2048)]


def test_config_has_as_many_parameters_as_gutformer(data_schema):
    # Given / When
    model = build_model(data_schema)

    # Then
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == sum(parameter.numel() for parameter in GutFormer(data_schema).parameters())


def test_overrides_reach_the_components(data_schema):
    # Given / When
    model = build_model(
        data_schema,
        "model.tokenizer.embedding_dim=16",
        "model.encoder.stacks.0.num_layers=2",
        "model.decoder.stack.num_heads=4",
        "model.latent.latent_size=8",
    )

    # Then
    assert {module.embedding_dim for module in model.children()} == {16}
    assert [stack_settings(stack) for stack in model.encoder.stacks] == [
        [(2, 0.05, 2048), (2, 0.05, 2048)],
        [(16, 0.1, 2048)],
    ]
    assert stack_settings(model.decoder.stack) == [(4, 0.1, 2048)]
    inputs = {
        modality: torch.rand(BATCH_SIZE, len(names), 1)
        for modality, names in data_schema.feature_names.items()
    }
    assert model(inputs).latent.shape == (BATCH_SIZE, 8)
