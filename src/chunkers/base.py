from abc import ABC, abstractmethod
from typing import List
from models.document import Document, Chunk


class Chunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        pass
