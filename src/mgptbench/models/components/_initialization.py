"""[purbancz] Weight initialization shared by the components.

Each component calls these from its reset_parameters(), passing only the layers it owns,
so nested components (e.g. a pooling inside a latent) keep their own initialization.
"""

from torch import nn


def xavier_init_(*layers: nn.Linear) -> None:
    """[purbancz] Xavier-uniform weights and zero biases, as in GutFormer."""
    for layer in layers:
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
