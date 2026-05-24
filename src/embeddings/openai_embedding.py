import os
from typing import List
import openai
from .base import Embedding


class OpenAIEmbedding(Embedding):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None, base_url: str = None):
        self.model = model
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        self._dimension = 1536

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=valid_texts)
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dimension
