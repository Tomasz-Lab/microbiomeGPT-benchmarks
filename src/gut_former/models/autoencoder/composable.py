"""[purbancz] Generic container chaining the autoencoder components - ADR 0006."""

from collections.abc import Callable

from torch import Tensor

from gut_former.models.base import MicrobiomeModel, ModelOutput
from gut_former.models.components import Decoder, Encoder, Head, Latent, Tokenizer
from gut_former.schema import DataSchema


class ComposableAutoEncoder(MicrobiomeModel):
    """[purbancz] Architecture: Tokenizer -> Encoder -> Latent -> Decoder -> Head"""

    def __init__(
        self,
        data_schema: DataSchema,
        tokenizer: Callable[[DataSchema], Tokenizer],
        encoder: Encoder,
        latent: Latent,
        decoder: Decoder,
        head: Head,
    ):
        # [purbancz] tokenizer is a factory, because its size will depend on the data schema.
        super().__init__()
        self.modalities = tuple(data_schema.feature_names)
        self.tokenizer = tokenizer(data_schema)
        self.encoder = encoder
        self.latent = latent
        self.decoder = decoder
        self.head = head

    def forward(self, inputs: dict[str, Tensor]) -> ModelOutput:
        if set(inputs) != set(self.modalities):
            raise ValueError(
                f"Expected inputs for modalities {sorted(self.modalities)}, got {sorted(inputs)}."
            )
        ordered_inputs = {modality: inputs[modality] for modality in self.modalities}

        latent = self.latent.to_latent(self.encoder(self.tokenizer(ordered_inputs)))
        decoded_latent = self.latent.from_latent(latent).unsqueeze(1)
        queries = {
            modality: decoded_latent + self.tokenizer.feature_embeddings(modality)
            for modality in self.modalities
        }
        reconstructions = {
            modality: self.head(hidden_states)
            for modality, hidden_states in self.decoder(queries).items()
        }
        return ModelOutput(reconstructions=reconstructions, latent=latent)
