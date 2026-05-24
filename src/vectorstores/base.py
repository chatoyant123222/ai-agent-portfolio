from abc import ABC, abstractmethod
from typing import List, Tuple
from models.document import Chunk


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[Chunk, float]]:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass
