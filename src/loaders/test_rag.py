import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=== 测试 Document 和 Chunk ===")
from models.document import Document, Chunk

doc = Document(
    id="doc-001",
    content="测试内容",
    metadata={"author": "test"},
    source="test.txt"
)
print(f"✅ Document: {doc.id}")

chunk = Chunk(id="c1", document_id="doc-001", content="分块", index=0, metadata={})
print(f"✅ Chunk: {chunk.id}")

print("\n=== 测试抽象类保护 ===")
from models.document import DocumentLoader
try:
    DocumentLoader()
except TypeError as e:
    print(f"✅ 抽象类正常: {e}")

print("\n=== 测试 PDFLoader ===")
from loaders.pdf_loader import PDFLoader
loader = PDFLoader()
print(f"✅ PDFLoader 实例化成功")

print("\n=== 测试异常 ===")
try:
    loader.load("不存在.pdf")
except FileNotFoundError:
    print("✅ 异常处理正常")

print("\n🎉 全部通过，可以开始下一步了")