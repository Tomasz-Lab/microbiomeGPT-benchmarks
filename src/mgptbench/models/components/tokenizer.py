from abc import ABC, abstractmethod

from torch import Tensor, nn

from mgptbench.models.components._initialization import xavier_init_
from mgptbench.schema import DataSchema


class Tokenizer(nn.Module, ABC):
    """[purbancz] Turns the input values of each modality into token embeddings.

    It also holds one identity embedding per feature, so the decoder side can ask for
    a specific feature to be reconstructed.
    """

    @abstractmethod
    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """[purbancz] modality name -> (batch_size, num_features, 1) becomes
        modality name -> (batch_size, num_features, embedding_dim)."""

    @abstractmethod
    def feature_embeddings(self, modality: str) -> Tensor:
        """[purbancz] Identity embeddings of one modality's features,
        shape (num_features, embedding_dim)."""


class FeatureEmbeddingTokenizer(Tokenizer):
    """[purbancz] Token = shared projection of the value + identity embedding of the feature."""

    def __init__(self, data_schema: DataSchema, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.value_projection = nn.Linear(1, embedding_dim)
        self.embeddings = nn.ModuleDict(
            {
                modality: nn.Embedding(len(names), embedding_dim)
                for modality, names in data_schema.feature_names.items()
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # [purbancz] The embeddings keep PyTorch's default, normal(0, 1)
        xavier_init_(self.value_projection)

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            modality: self.value_projection(values) + self.feature_embeddings(modality)
            for modality, values in inputs.items()
        }

    def feature_embeddings(self, modality: str) -> Tensor:
        embedding = self.embeddings[modality]
        assert isinstance(embedding, nn.Embedding)
        return embedding.weight
