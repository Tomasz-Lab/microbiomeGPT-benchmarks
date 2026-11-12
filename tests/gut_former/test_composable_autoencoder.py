import pytest
import torch
from torch import Tensor

from gut_former.models import Decoder, Encoder, Head, Latent, Tokenizer
from gut_former.models.autoencoder import ComposableAutoEncoder
from gut_former.schema import DataSchema

EMBEDDING_DIM = 3
LATENT_SIZE = 2
BATCH_SIZE = 4


class StubTokenizer(Tokenizer):
    def __init__(self, data_schema: DataSchema):
        super().__init__()
        self.received_modalities: list[str] = []
        self._feature_embeddings = {
            modality: torch.arange(len(names) * EMBEDDING_DIM, dtype=torch.float32).view(
                len(names), EMBEDDING_DIM
            )
            * (index + 1)
            for index, (modality, names) in enumerate(data_schema.feature_names.items())
        }

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        self.received_modalities = list(inputs)
        return {
            modality: values.expand(-1, -1, EMBEDDING_DIM) + self.feature_embeddings(modality)
            for modality, values in inputs.items()
        }

    def feature_embeddings(self, modality: str) -> Tensor:
        return self._feature_embeddings[modality]


class StubEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.received_modalities: list[str] = []

    def forward(self, tokens: dict[str, Tensor]) -> Tensor:
        self.received_modalities = list(tokens)
        return torch.cat(list(tokens.values()), dim=1)


class StubLatent(Latent):
    def to_latent(self, hidden_states: Tensor) -> Tensor:
        return hidden_states.mean(dim=1)[:, :LATENT_SIZE]

    def from_latent(self, latent: Tensor) -> Tensor:
        return latent.sum(dim=1, keepdim=True).expand(-1, EMBEDDING_DIM)


class StubDecoder(Decoder):
    def __init__(self):
        super().__init__()
        self.received_queries: dict[str, Tensor] = {}

    def forward(self, queries: dict[str, Tensor]) -> dict[str, Tensor]:
        self.received_queries = queries
        return queries


class StubHead(Head):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states.sum(dim=-1, keepdim=True)


@pytest.fixture
def data_schema() -> DataSchema:
    return DataSchema.from_dict(
        {"taxonomy": ["taxon_0", "taxon_1"], "pathways": ["pathway_0", "pathway_1", "pathway_2"]}
    )


@pytest.fixture
def model(data_schema) -> ComposableAutoEncoder:
    return ComposableAutoEncoder(
        data_schema=data_schema,
        tokenizer=StubTokenizer,
        encoder=StubEncoder(),
        latent=StubLatent(),
        decoder=StubDecoder(),
        head=StubHead(),
    )


@pytest.fixture
def inputs(data_schema) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        modality: torch.rand(BATCH_SIZE, len(names), 1, generator=generator)
        for modality, names in data_schema.feature_names.items()
    }


@pytest.mark.parametrize(
    "modalities",
    [["taxonomy"], ["taxonomy", "pathways", "phylogeny"]],
    ids=["missing", "extra"],
)
def test_rejects_inputs_with_other_modalities(model, inputs, modalities):
    # Given
    other_inputs = {
        modality: inputs.get(modality, torch.zeros(BATCH_SIZE, 1, 1)) for modality in modalities
    }

    # When / Then
    with pytest.raises(ValueError, match="modalities"):
        model(other_inputs)


def test_processes_modalities_in_data_schema_order(model, inputs):
    # Given
    reversed_inputs = dict(reversed(list(inputs.items())))

    # When
    model(reversed_inputs)

    # Then
    assert model.tokenizer.received_modalities == ["taxonomy", "pathways"]
    assert model.encoder.received_modalities == ["taxonomy", "pathways"]


def test_decoder_queries_are_decoded_latent_plus_feature_embeddings(model, inputs):
    # Given
    tokenizer, latent = model.tokenizer, model.latent

    # When
    output = model(inputs)

    # Then
    decoded_latent = latent.from_latent(output.latent).unsqueeze(1)
    for modality in ("taxonomy", "pathways"):
        expected = decoded_latent + tokenizer.feature_embeddings(modality)
        torch.testing.assert_close(model.decoder.received_queries[modality], expected)


def test_returns_one_reconstruction_per_modality_and_the_latent(model, inputs, data_schema):
    # Given / When
    output = model(inputs)

    # Then
    assert list(output.reconstructions) == ["taxonomy", "pathways"]
    for modality, names in data_schema.feature_names.items():
        assert output.reconstructions[modality].shape == (BATCH_SIZE, len(names), 1)
    assert output.latent.shape == (BATCH_SIZE, LATENT_SIZE)
    assert output.aux == {}
