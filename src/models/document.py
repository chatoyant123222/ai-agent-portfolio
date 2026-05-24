from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict


class Document(BaseModel):
    id: str = Field(..., description="文档唯一标识")
    content: str = Field(..., description="文档完整文本内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    source: str = Field(..., description="文档来源路径或URL")


class Chunk(BaseModel):
    id: str = Field(..., description="分块唯一标识")
    document_id: str = Field(..., description="所属文档ID")
    content: str = Field(..., description="分块文本内容")
    index: int = Field(..., description="分块在文档中的序号")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="分块元数据")


class DocumentLoader(ABC):
    
    @abstractmethod
    def load(self, source: str) -> Document:
        pass
    
    @abstractmethod
    def parse(self, raw_data: Any) -> Document:
        pass