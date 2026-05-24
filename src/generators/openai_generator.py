import os
from typing import List
import openai
from models.document import Chunk
from .base import Generator


class OpenAIGenerator(Generator):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = None,
        base_url: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        system_prompt: str = None
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        self.system_prompt = system_prompt or (
            "你是一个专业的文档问答助手。请严格基于提供的参考资料回答问题。"
            "如果资料中没有相关信息，请明确说明'根据提供的资料，我无法回答这个问题'。"
            "不要编造信息。"
        )

    def generate(self, query: str, context: List[Chunk]) -> str:
        context_text = "\n\n".join([
            f"[参考片段 {i+1}]:\n{chunk.content}"
            for i, chunk in enumerate(context)
        ])
        user_prompt = (
            f"用户问题：{query}\n\n"
            f"参考资料：\n{context_text}\n\n"
            f"请基于以上参考资料回答用户问题。"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()
