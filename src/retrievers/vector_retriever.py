from typing import List
from models.document import Chunk
from embeddings.base import Embedding
from vectorstores.base import VectorStore
from .base import Retriever


class VectorRetriever(Retriever):
    def __init__(self, embedding: Embedding, vector_store: VectorStore, top_k: int = 5):
        self.embedding = embedding
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[Chunk]:
        k = top_k or self.top_k
        query_vector = self.embedding.embed(query)
        results = self.vector_store.search(query_vector, top_k=k)
        return [chunk for chunk, score in results]
