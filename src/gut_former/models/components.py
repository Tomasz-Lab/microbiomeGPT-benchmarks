"""[purbancz] Component-level interfaces - ADR 0006."""

from abc import ABC, abstractmethod

from torch import Tensor, nn


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


class Encoder(nn.Module, ABC):
    """[purbancz] Fuses the tokens of all modalities into one sequence.

    How they are fused (jointly from the start, or encoded separately and mixed later)
    is up to the implementation.
    """

    @abstractmethod
    def forward(self, tokens: dict[str, Tensor]) -> Tensor:
        """[purbancz] modality name -> (batch_size, num_features, embedding_dim) becomes
        (batch_size, sequence_length, embedding_dim); sequence_length is set by the encoder."""


class Latent(nn.Module, ABC):
    """[purbancz] Maps the encoded sequence to one latent vector per sample and back."""

    @abstractmethod
    def to_latent(self, hidden_states: Tensor) -> Tensor:
        """[purbancz] (batch_size, sequence_length, embedding_dim) to (batch_size, latent_size)."""

    @abstractmethod
    def from_latent(self, latent: Tensor) -> Tensor:
        """[purbancz] (batch_size, latent_size) to (batch_size, embedding_dim)."""


class Decoder(nn.Module, ABC):
    """[purbancz] Turns the per-feature queries of each modality into hidden states.

    Modalities may be decoded separately or attend to each other; that is up to
    the implementation.
    """

    @abstractmethod
    def forward(self, queries: dict[str, Tensor]) -> dict[str, Tensor]:
        """[purbancz] modality name -> (batch_size, num_features, embedding_dim),
        same keys and shapes out."""


class Head(nn.Module, ABC):
    """[purbancz] Projects hidden states to one reconstructed value per feature."""

    @abstractmethod
    def forward(self, hidden_states: Tensor) -> Tensor:
        """[purbancz] (batch_size, num_features, embedding_dim) to (batch_size, num_features, 1)."""
