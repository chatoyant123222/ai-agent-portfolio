import random
from typing import List
from .base import Embedding


class FakeEmbedding(Embedding):
    def __init__(self, dimension: int = 1536):
        self._dimension = dimension

    def embed(self, text: str) -> List[float]:
        seed = hash(text) % (2**31)
        rng = random.Random(seed)
        return [rng.uniform(-1, 1) for _ in range(self._dimension)]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension
