from dataclasses import dataclass


@dataclass(frozen=True)
class DataSchema:
    """Feature names per modality, exposed by a DataModule after setup and interpreted by models."""

    feature_names: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, list[str]]:
        return {modality: list(names) for modality, names in self.feature_names.items()}

    @classmethod
    def from_dict(cls, feature_names: dict[str, list[str]]) -> "DataSchema":
        return cls({modality: tuple(names) for modality, names in feature_names.items()})
