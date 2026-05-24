from abc import ABC, abstractmethod
from typing import List
from models.document import Chunk


class Generator(ABC):
    @abstractmethod
    def generate(self, query: str, context: List[Chunk]) -> str:
        pass
