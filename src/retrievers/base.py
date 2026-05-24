from abc import ABC, abstractmethod
from typing import List
from models.document import Chunk


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Chunk]:
        pass
