import numpy as np
from typing import List, Tuple
from models.document import Chunk
from .base import VectorStore


class MemoryVectorStore(VectorStore):
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.vectors: np.ndarray = None

    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        if not chunks or not vectors:
            return
        new_vectors = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        new_vectors = new_vectors / norms

        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

        self.chunks.extend(chunks)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.vectors is None or len(self.chunks) == 0:
            return []

        query = np.array(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        similarities = np.dot(self.vectors, query)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0:
                continue
            results.append((self.chunks[idx], score))

        return results

    def clear(self) -> None:
        self.chunks = []
        self.vectors = None
