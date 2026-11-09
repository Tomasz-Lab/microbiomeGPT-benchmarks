from dataclasses import dataclass


@dataclass(frozen=True)
class DataSchema:
    """Feature names per modality, exposed by a DataModule after setup and interpreted by models."""

    feature_names: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, list[str]]:
        return {modality: list(names) for modality, names in self.feature_names.items()}

    def to_config(self) -> dict:
        """Plain instantiable config (`_target_`), safe for Hydra and checkpoint hyperparameters."""
        return {
            "_target_": "gut_former.schema.DataSchema.from_dict",
            "feature_names": self.to_dict(),
        }

    @classmethod
    def from_dict(cls, feature_names: dict[str, list[str]]) -> "DataSchema":
        return cls({modality: tuple(names) for modality, names in feature_names.items()})
