import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.document import Document, DocumentLoader


class PDFLoader(DocumentLoader):
    
    def __init__(self, extract_metadata: bool = True):
        self.extract_metadata = extract_metadata
    
    def load(self, source: str) -> Document:
        if not os.path.exists(source):
            raise FileNotFoundError(f"PDF 文件不存在: {source}")
        
        try:
            reader = PdfReader(source)
            return self.parse(reader, source_path=source)
        except Exception as e:
            raise ValueError(f"读取 PDF 文件失败: {e}")
    
    def parse(self, raw_data: Any, source_path: Optional[str] = None) -> Document:
        if isinstance(raw_data, str):
            reader = PdfReader(raw_data)
            source_path = raw_data
        else:
            reader = raw_data
        
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        full_content = "\n\n".join(text_parts)
        
        metadata: Dict[str, Any] = {"total_pages": len(reader.pages)}
        
        if self.extract_metadata and reader.metadata:
            pdf_meta = reader.metadata
            metadata.update({
                "title": str(pdf_meta.get("/Title", "")),
                "author": str(pdf_meta.get("/Author", "")),
                "subject": str(pdf_meta.get("/Subject", "")),
            })
        
        return Document(
            id=str(uuid.uuid4()),
            content=full_content,
            metadata=metadata,
            source=source_path or "unknown"
        )