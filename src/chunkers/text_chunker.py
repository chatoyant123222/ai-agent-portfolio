import uuid
from typing import List
from models.document import Document, Chunk
from .base import Chunker


class TextChunker(Chunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> List[Chunk]:
        text = document.content
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunk_text = text[start:]
            else:
                chunk_text = text[start:end]
                for sep in ["\n\n", "\n", ". ", " ", ""]:
                    idx = chunk_text.rfind(sep)
                    if idx != -1 and idx > self.chunk_size * 0.5:
                        chunk_text = chunk_text[:idx + len(sep)]
                        break

            chunk = Chunk(
                id=f"{document.id}_chunk_{index}",
                document_id=document.id,
                content=chunk_text.strip(),
                index=index,
                metadata={"start_char": start, "end_char": start + len(chunk_text), **document.metadata}
            )
            chunks.append(chunk)

            start += len(chunk_text)
            if len(chunk_text) == 0:
                break
            index += 1

        return chunks
