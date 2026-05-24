import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("错误：未找到 OPENAI_API_KEY 环境变量")
    print("PowerShell 临时设置：sk-4hwedL1g9fWCiQyuqFlt1H7Zg6Xf9I38D1r9Unq91VjN33Iz='sk-xxxxxxxx'")
    sys.exit(1)

from models.document import Document
from chunkers.text_chunker import TextChunker
from embeddings.openai_embedding import OpenAIEmbedding
from vectorstores.memory_store import MemoryVectorStore
from retrievers.vector_retriever import VectorRetriever
from generators.openai_generator import OpenAIGenerator

print("=" * 60)
print("OpenAI 完整 RAG 测试")
print("=" * 60)

print("[1/4] 初始化组件...")
embedder = OpenAIEmbedding(api_key=api_key)
chunker = TextChunker(chunk_size=400, chunk_overlap=80)
store = MemoryVectorStore()
retriever = VectorRetriever(embedder, store, top_k=3)
generator = OpenAIGenerator(model="gpt-4o-mini", api_key=api_key)
print(f"   Embedding: {embedder.model}, 维度={embedder.dimension}")
print(f"   Generator: {generator.model}")

print("[2/4] 加载文档...")
long_text = '''人工智能是计算机科学的一个分支，致力于创造能够执行通常需要人类智能的任务的机器。
这些任务包括学习、推理、问题解决、感知和语言理解。

机器学习是人工智能的一个子集，使用统计技术使计算机系统能够从数据中学习和改进，而无需明确编程。
深度学习是机器学习的一个子集，使用多层神经网络进行建模。

自然语言处理是人工智能的另一个重要领域，研究计算机与人类语言之间的交互。
大型语言模型是自然语言处理的最新进展，能够理解和生成人类语言。

检索增强生成（RAG）是一种结合检索系统和生成模型的技术，用于提高回答的准确性和减少幻觉。
它先从外部知识库检索相关信息，然后基于这些信息生成回答。
向量数据库是检索增强生成系统的关键组件，用于存储和搜索文本向量。'''

doc = Document(id="openai-test-001", content=long_text, metadata={"topic": "AI科普"}, source="memory://demo")
print(f"   文档长度: {len(doc.content)} 字符")

print("[3/4] 分块 & 向量化 & 存储...")
chunks = chunker.chunk(doc)
print(f"   分块: {len(chunks)} 个片段")
vectors = embedder.embed_batch([c.content for c in chunks])
print(f"   向量化: {len(vectors)} 个向量")
store.add(chunks, vectors)
print("   存储完成")

print("[4/4] 检索 & 生成...")
question = "什么是检索增强生成（RAG）？它和向量数据库有什么关系？"
print(f"   问题: {question}")
retrieved = retriever.retrieve(question)
print(f"   检索到 {len(retrieved)} 个片段")

print("=" * 60)
print("LLM 生成回答：")
print("=" * 60)
answer = generator.generate(question, retrieved)
print(answer)
print("=" * 60)
print("OpenAI RAG 全流程测试完成！")
