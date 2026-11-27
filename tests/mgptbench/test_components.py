"""[purbancz] Unit tests of the reusable component implementations."""

from typing import TypedDict

import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn

from mgptbench.models.autoencoder import ComposableAutoEncoder
from mgptbench.models.components import (
    FeatureEmbeddingTokenizer,
    JointTransformerEncoder,
    LinearHead,
    MeanPooling,
    PooledLinearLatent,
    Pooling,
    SharedTransformerDecoder,
)
from mgptbench.models.components._transformer_stack import TransformerStack
from mgptbench.schema import DataSchema

EMBEDDING_DIM = 8
LATENT_SIZE = 4
BATCH_SIZE = 3


class StackSettings(TypedDict):
    num_layers: int
    num_heads: int
    dropout: float
    feedforward_dim: int


STACK: StackSettings = {"num_layers": 1, "num_heads": 2, "dropout": 0.0, "feedforward_dim": 16}


class FirstTokenPooling(Pooling):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states[:, 0]


class LinearPooling(Pooling):
    """[purbancz] A pooling with its own weights, initialized to ones."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        nn.init.ones_(self.projection.weight)
        nn.init.ones_(self.projection.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        pooled: Tensor = self.projection(hidden_states).mean(dim=1)
        return pooled


@pytest.fixture
def data_schema() -> DataSchema:
    return DataSchema.from_dict(
        {"taxonomy": ["taxon_0", "taxon_1"], "pathways": ["pathway_0", "pathway_1", "pathway_2"]}
    )


@pytest.fixture
def inputs(data_schema) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        modality: torch.rand(BATCH_SIZE, len(names), 1, generator=generator)
        for modality, names in data_schema.feature_names.items()
    }


def hidden_states(sequence_length: int) -> Tensor:
    generator = torch.Generator().manual_seed(1)
    return torch.rand(BATCH_SIZE, sequence_length, EMBEDDING_DIM, generator=generator)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("num_layers", 0),
        ("num_heads", 0),
        ("feedforward_dim", 0),
        ("dropout", -0.1),
    ],
)
def test_stack_rejects_invalid_settings(setting, value):
    # When / Then
    with pytest.raises(ValueError, match=setting):
        JointTransformerEncoder(EMBEDDING_DIM, stacks=[STACK | {setting: value}])


def test_stack_rejects_unknown_settings():
    # Given: a misspelled key, as it could come from yaml
    stack = {key: value for key, value in STACK.items() if key != "num_heads"} | {"num_head": 2}

    # When / Then
    with pytest.raises(TypeError, match="num_head"):
        SharedTransformerDecoder(EMBEDDING_DIM, stack=stack)


def test_stack_rejects_num_heads_that_do_not_divide_embedding_dim():
    # When / Then
    with pytest.raises(ValueError, match="divisible"):
        JointTransformerEncoder(EMBEDDING_DIM, stacks=[STACK | {"num_heads": 3}])


def test_tokenizer_token_is_value_projection_plus_feature_embedding(data_schema, inputs):
    # Given
    tokenizer = FeatureEmbeddingTokenizer(data_schema, EMBEDDING_DIM)

    # When
    tokens = tokenizer(inputs)

    # Then
    for modality, names in data_schema.feature_names.items():
        embeddings = tokenizer.feature_embeddings(modality)
        assert embeddings.shape == (len(names), EMBEDDING_DIM)
        expected = tokenizer.value_projection(inputs[modality]) + embeddings
        torch.testing.assert_close(tokens[modality], expected)


def test_encoder_concatenates_modalities_into_one_sequence(data_schema, inputs):
    # Given
    tokens = FeatureEmbeddingTokenizer(data_schema, EMBEDDING_DIM)(inputs)
    encoder = JointTransformerEncoder(EMBEDDING_DIM, stacks=[STACK])

    # When
    encoded = encoder(tokens)

    # Then
    assert encoded.shape == (BATCH_SIZE, 2 + 3, EMBEDDING_DIM)


def first_layer_settings(stack) -> tuple[int, int, float, int]:
    """[purbancz] Layer count and first-layer settings"""
    layer = stack.layers[0]
    return (
        len(stack.layers),
        layer.self_attn.num_heads,
        layer.dropout.p,
        layer.linear1.out_features,
    )


def test_encoder_builds_each_stack_with_its_own_settings():
    # Given: stack settings as Hydra passes them from yaml
    stacks = OmegaConf.create(
        [
            {"num_layers": 2, "num_heads": 2, "dropout": 0.05, "feedforward_dim": 16},
            {"num_layers": 1, "num_heads": 8, "dropout": 0.1, "feedforward_dim": 32},
        ]
    )

    # When
    encoder = JointTransformerEncoder(EMBEDDING_DIM, stacks=stacks)

    # Then
    built = [first_layer_settings(stack) for stack in encoder.stacks]
    assert built == [(2, 2, 0.05, 16), (1, 8, 0.1, 32)]


def test_encoder_needs_at_least_one_stack():
    # When / Then
    with pytest.raises(ValueError, match="at least one stack"):
        JointTransformerEncoder(EMBEDDING_DIM, stacks=[])


def test_stack_layers_start_with_their_own_weights():
    # Given
    stack = TransformerStack(EMBEDDING_DIM, **(STACK | {"num_layers": 3}))

    # When
    first, *others = [dict(layer.named_parameters()) for layer in stack.layers]

    # Then: every randomly initialized weight differs between layers
    for name in ("self_attn.in_proj_weight", "self_attn.out_proj.weight", "linear1.weight"):
        assert all(not torch.equal(first[name], other[name]) for other in others), name


def test_stack_runs_its_layers_in_order():
    # Given
    stack = TransformerStack(EMBEDDING_DIM, **(STACK | {"num_layers": 2})).eval()
    states = hidden_states(sequence_length=5)

    # When
    output = stack(states)

    # Then
    torch.testing.assert_close(output, stack.layers[1](stack.layers[0](states)))


def test_mean_pooling_averages_over_the_sequence():
    # Given
    states = hidden_states(sequence_length=5)

    # When
    pooled = MeanPooling()(states)

    # Then
    torch.testing.assert_close(pooled, states.mean(dim=1))


def test_latent_compresses_the_pooled_sequence():
    # Given
    latent = PooledLinearLatent(EMBEDDING_DIM, LATENT_SIZE, pooling=FirstTokenPooling())
    states = hidden_states(sequence_length=5)

    # When
    compressed = latent.to_latent(states)

    # Then
    assert compressed.shape == (BATCH_SIZE, LATENT_SIZE)
    torch.testing.assert_close(compressed, latent.compress(states[:, 0]))


def test_latent_expands_back_to_embedding_dim():
    # Given
    latent = PooledLinearLatent(EMBEDDING_DIM, LATENT_SIZE, pooling=MeanPooling())

    # When
    decoded = latent.from_latent(torch.rand(BATCH_SIZE, LATENT_SIZE))

    # Then
    assert decoded.shape == (BATCH_SIZE, EMBEDDING_DIM)


def test_latent_keeps_the_pooling_initialization():
    # When
    pooling = LinearPooling()
    latent = PooledLinearLatent(EMBEDDING_DIM, LATENT_SIZE, pooling=pooling)

    # Then
    assert latent.pooling is pooling
    projection = pooling.projection
    assert torch.all(projection.weight == 1)
    assert torch.all(projection.bias == 1)


def test_decoder_applies_one_shared_stack_to_each_modality_separately():
    # Given
    decoder = SharedTransformerDecoder(EMBEDDING_DIM, stack=STACK).eval()
    queries = {"taxonomy": hidden_states(2), "pathways": hidden_states(3)}

    # When
    decoded = decoder(queries)

    # Then
    for modality, query in queries.items():
        torch.testing.assert_close(decoded[modality], decoder({modality: query})[modality])
        assert decoded[modality].shape == query.shape


def test_head_projects_each_feature_to_one_value():
    # When
    reconstruction = LinearHead(EMBEDDING_DIM)(hidden_states(sequence_length=5))

    # Then
    assert reconstruction.shape == (BATCH_SIZE, 5, 1)


def test_linear_layers_start_with_zero_bias(data_schema):
    # Given
    components = [
        FeatureEmbeddingTokenizer(data_schema, EMBEDDING_DIM),
        JointTransformerEncoder(EMBEDDING_DIM, stacks=[STACK, STACK]),
        PooledLinearLatent(EMBEDDING_DIM, LATENT_SIZE, pooling=MeanPooling()),
        SharedTransformerDecoder(EMBEDDING_DIM, stack=STACK),
        LinearHead(EMBEDDING_DIM),
    ]

    # When
    linear_layers = [
        module
        for component in components
        for module in component.modules()
        if isinstance(module, nn.Linear)
    ]

    # Then
    assert linear_layers
    assert all(torch.all(layer.bias == 0) for layer in linear_layers)


def test_components_compose_into_an_autoencoder(data_schema, inputs):
    # Given
    model = ComposableAutoEncoder(
        data_schema=data_schema,
        tokenizer=lambda schema: FeatureEmbeddingTokenizer(schema, EMBEDDING_DIM),
        encoder=JointTransformerEncoder(EMBEDDING_DIM, stacks=[STACK, STACK]),
        latent=PooledLinearLatent(EMBEDDING_DIM, LATENT_SIZE, pooling=MeanPooling()),
        decoder=SharedTransformerDecoder(EMBEDDING_DIM, stack=STACK),
        head=LinearHead(EMBEDDING_DIM),
    )

    # When
    output = model(inputs)

    # Then
    for modality, names in data_schema.feature_names.items():
        assert output.reconstructions[modality].shape == (BATCH_SIZE, len(names), 1)
    assert output.latent.shape == (BATCH_SIZE, LATENT_SIZE)
