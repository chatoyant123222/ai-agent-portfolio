from typing import List
from models.document import Document, Chunk
from loaders.pdf_loader import PDFLoader
from chunkers.base import Chunker
from embeddings.base import Embedding
from vectorstores.base import VectorStore
from retrievers.base import Retriever
from generators.base import Generator


class RAGPipeline:
    """
    完整的 RAG 流水线。
    
    使用流程：
        1. 调用 ingest(document) 将文档入库
        2. 调用 query(question) 获取回答
    """
    
    def __init__(
        self,
        loader: PDFLoader,
        chunker: Chunker,
        embedding: Embedding,
        vector_store: VectorStore,
        retriever: Retriever,
        generator: Generator
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedding = embedding
        self.vector_store = vector_store
        self.retriever = retriever
        self.generator = generator
    
    def ingest(self, source: str) -> Document:
        """
        文档入库流程：加载 → 分块 → 向量化 → 存储
        
        Args:
            source: 文档路径（如 PDF 文件路径）
            
        Returns:
            Document: 加载后的文档对象
        """
        # 1. 加载
        document = self.loader.load(source)
        print(f"📄 加载文档: {document.source} ({document.metadata.get('total_pages', '?')} 页)")
        
        # 2. 分块
        chunks = self.chunker.chunk(document)
        print(f"✂️ 分块完成: {len(chunks)} 个片段")
        
        # 3. 向量化
        texts = [c.content for c in chunks]
        vectors = self.embedding.embed_batch(texts)
        print(f"🔢 向量化完成: {len(vectors)} 个向量 (维度: {self.embedding.dimension})")
        
        # 4. 存储
        self.vector_store.add(chunks, vectors)
        print(f"💾 存入向量库成功")
        
        return document
    
    def query(self, question: str) -> str:
        """
        问答流程：检索 → 生成
        
        Args:
            question: 用户问题
            
        Returns:
            str: 生成的回答
        """
        # 1. 检索
        relevant_chunks = self.retriever.retrieve(question)
        print(f"🔍 检索到 {len(relevant_chunks)} 个相关片段")
        
        # 2. 生成
        answer = self.generator.generate(question, relevant_chunks)
        return answer
    
    def reset(self):
        """清空向量库"""
        self.vector_store.clear()
        print("🗑️ 向量库已清空")