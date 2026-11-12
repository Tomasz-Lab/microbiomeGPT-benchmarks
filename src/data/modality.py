from dataclasses import dataclass
from typing import Any, Literal

from data.kinds.base import ModalityKind

type Scope = Literal["per_sample", "reference"]


@dataclass(frozen=True)
class ModalitySpec:
    """Modality specification, describing the structure of a data modality.

    Attributes:
        name: Modality name.
        kind: Modality kind.
        scope: Whether the modality is per-sample (e.g. species abundances) or reference (e.g. a reference genome) or something else.
        description(optional): short description of what the values mean and where they come from.
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
