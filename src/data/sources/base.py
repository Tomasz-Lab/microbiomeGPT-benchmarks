from abc import ABC, abstractmethod


class Source[T](ABC):
    @abstractmethod
    def read(self) -> T:
        pass
