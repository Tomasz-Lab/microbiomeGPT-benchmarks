from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal


class ModalityKind(ABC):
    @abstractmethod
    def validate(self, data: Any, name: str) -> Any:
        pass

    @abstractmethod
    def sample_ids(self, data: Any) -> frozenset[str]:
        pass

    @abstractmethod
    def select_samples(self, data: Any, ids: Sequence[str]) -> Any:
        pass


type Scope = Literal["per_sample", "reference"]


@dataclass(frozen=True)
class ModalitySpec:
    """Definition of one modality (e.g. taxonomy_species).

    Attributes:
        name: Modality name
        kind: Modality kind
        scope: Whether the modality is per-sample (e.g. species abundances) or reference (e.g. a reference genome) or something else.
        description(optional): short description of what the values mean.
    """

    name: str
    kind: ModalityKind
    scope: Scope
    description: str = ""

    def __post_init__(self) -> None:
        # becasue we later plan that e.g othre code parts of the system will use the name as a Python identifier
        # (e.g self.dataset.<modality_name>), we require that the name is a valid identifier, that's why we check it here
        # and raise an error if it's not.
        if not self.name.isidentifier():
            raise ValueError(
                f"Modality name must be a valid Python identifier (return True for str.isidentifier()), got '{self.name}'"
            )

    def validate(self, data: Any) -> Any:
        return self.kind.validate(data, self.name)


@dataclass(frozen=True)
class Modality:
    """One modality's data and its definition.
    Attributes:
        spec: Which modality this is.
        data: The values, e.g. a DataFrame for a `CompositionalTable`.
    """

    spec: ModalitySpec
    data: Any

    def select_samples(self, ids: Sequence[str]) -> "Modality":
        return Modality(spec=self.spec, data=self.spec.kind.select_samples(self.data, ids))
