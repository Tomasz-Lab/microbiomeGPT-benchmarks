from data.modalities.base import ModalitySpec
from data.modalities.specs import ALL_SPECS

type ModalityName = str
"""Name of a registered modality (e.g. "taxonomy_metaphlan")."""


class ModalityRegistry:
    """Modality specs by name, built once from a list of specs."""

    def __init__(self, specs: list[ModalitySpec]) -> None:
        self._specs: dict[ModalityName, ModalitySpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise ValueError(f"Duplicate modality '{spec.name}'")
            self._specs[spec.name] = spec

    def get(self, name: ModalityName) -> ModalitySpec:
        """Return the spec registered under `name`; raise KeyError listing registered names if absent."""
        if name not in self._specs:
            raise KeyError(f"Unknown modality '{name}'; registered: {sorted(self._specs)}")
        return self._specs[name]


MODALITIES = ModalityRegistry(ALL_SPECS)
