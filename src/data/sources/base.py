from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceInfo:
    """Description of where a source reads its data from

    Attributes:
        location: Where the data lives, e.g. an absolute file path, "s3://bucket/key"
            or a database table.
    """

    location: str


class Source[T](ABC):
    @abstractmethod
    def read(self) -> T:
        pass

    @abstractmethod
    def describe(self) -> SourceInfo:
        pass
