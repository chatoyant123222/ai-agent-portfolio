import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.document import Document
from chunkers.text_chunker import TextChunker
from embeddings.fake_embedding import FakeEmbedding
from vectorstores.memory_store import MemoryVectorStore
from retrievers.vector_retriever import VectorRetriever
from generators.echo_generator import EchoGenerator

print("=" * 60)
print("🚀 RAG 纯本地架构验证测试")
print("=" * 60)

# ========== 1. 构造模拟文档（模拟 PDFLoader 的输出） ==========
long_text = """人工智能是计算机科学的一个分支，致力于创造能够执行通常需要人类智能的任务的机器。
这些任务包括学习、推理、问题解决、感知和语言理解。

机器学习是人工智能的一个子集，使用统计技术使计算机系统能够从数据中学习和改进，而无需明确编程。
深度学习是机器学习的一个子集，使用多层神经网络进行建模。

自然语言处理是人工智能的另一个重要领域，研究计算机与人类语言之间的交互。
大型语言模型是自然语言处理的最新进展，能够理解和生成人类语言。

检索增强生成是一种结合检索系统和生成模型的技术，用于提高回答的准确性和减少幻觉。
它先从外部知识库检索相关信息，然后基于这些信息生成回答。
向量数据库是检索增强生成系统的关键组件，用于存储和搜索文本向量。
"""

doc = Document(
    id="local-test-001",
    content=long_text,
    metadata={"test": True, "pages": 1},
    source="memory://demo"
)
print(f"\n✅ [1/6] 文档创建: id={doc.id}, 长度={len(doc.content)} 字符")

# ========== 2. 分块 ==========
chunker = TextChunker(chunk_size=300, chunk_overlap=50)
chunks = chunker.chunk(doc)
print(f"\n✅ [2/6] 分块完成: {len(chunks)} 个片段")
for i, c in enumerate(chunks):
    preview = c.content[:60].replace("\n", " ")
    print(f"    块 {i}: id={c.id}, 长度={len(c.content)}, 预览: {preview}...")

# ========== 3. 向量化 ==========
embedder = FakeEmbedding(dimension=128)  # 小维度，本地快速测试
vectors = embedder.embed_batch([c.content for c in chunks])
print(f"\n✅ [3/6] 向量化完成: {len(vectors)} 个向量, 维度={embedder.dimension}")
print(f"    向量示例 (前5个值): {vectors[0][:5]}")

# ========== 4. 存储 ==========
store = MemoryVectorStore()
store.add(chunks, vectors)
print(f"\n✅ [4/6] 存入 MemoryVectorStore: 共 {len(store.chunks)} 条记录")

# ========== 5. 检索 ==========
query = "什么是检索增强生成技术？"
retriever = VectorRetriever(embedder, store, top_k=3)
retrieved = retriever.retrieve(query)
print(f"\n✅ [5/6] 检索完成: 查询='{query}'")
print(f"    返回 {len(retrieved)} 个结果")
for i, c in enumerate(retrieved):
    preview = c.content[:80].replace("\n", " ")
    print(f"    结果 {i}: 块索引={c.index}, 预览={preview}...")

# 额外验证：用某个 chunk 的原始向量查询自身，确保存储/检索逻辑正确
if chunks:
    exact_vector = embedder.embed(chunks[0].content)
    exact_results = store.search(exact_vector, top_k=1)
    if exact_results:
        _, score = exact_results[0]
        print(f"\n    💡 精确匹配验证: 用块0的向量查询自身，相似度={score:.4f} (应为 1.0000)")

# ========== 6. 生成（本地 Echo，不调用 API） ==========
generator = EchoGenerator()
answer = generator.generate(query, retrieved)
print(f"\n✅ [6/6] 生成完成 (Echo模式，未调用LLM)")
print("-" * 60)
print(answer)
print("-" * 60)

print("\n🎉 纯本地架构验证全部通过！")
print("   说明：FakeEmbedding 使用伪随机向量，语义检索结果可能不精确，")
print("   但所有组件接口已验证连通。下一步可替换为 OpenAIEmbedding + OpenAIGenerator。")