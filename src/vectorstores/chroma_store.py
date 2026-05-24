from typing import List, Tuple
import chromadb
from chromadb.config import Settings
from models.document import Chunk
from .base import VectorStore


class ChromaVectorStore(VectorStore):
    """基于 ChromaDB 的持久化向量存储"""
    
    def __init__(self, collection_name: str = "rag_docs", persist_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        if not chunks:
            return
        
        ids = [c.id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [{
            "document_id": c.document_id,
            "index": c.index,
            **c.metadata
        } for c in chunks]
        
        self.collection.add(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas
        )
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[Chunk, float]]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        output = []
        if not results["ids"] or not results["ids"][0]:
            return output
        
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            chunk = Chunk(
                id=results["ids"][0][i],
                document_id=meta.get("document_id", ""),
                content=results["documents"][0][i],
                index=meta.get("index", 0),
                metadata={k: v for k, v in meta.items() if k not in ["document_id", "index"]}
            )
            # Chroma 返回的是距离（cosine distance = 1 - cosine similarity）
            distance = results["distances"][0][i]
            similarity = 1 - distance
            output.append((chunk, similarity))
        
        return output
    
    def clear(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )