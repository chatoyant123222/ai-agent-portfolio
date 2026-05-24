from typing import List
from models.document import Chunk
from .base import Generator


class EchoGenerator(Generator):
    def __init__(self, max_context_length: int = 2000):
        self.max_context_length = max_context_length

    def generate(self, query: str, context: List[Chunk]) -> str:
        if not context:
            return f"（本地测试模式）未检索到与问题相关的片段。\n问题：{query}"

        lines = [
            f"（本地测试模式 - 未调用 LLM）",
            f"",
            f"用户问题：{query}",
            f"",
            f"检索到 {len(context)} 个相关片段：",
            ""
        ]

        total_len = 0
        for i, chunk in enumerate(context, 1):
            header = f"--- 片段 {i} (文档: {chunk.document_id}, 索引: {chunk.index}) ---"
            content = chunk.content.strip()
            remaining = self.max_context_length - total_len
            if remaining <= 0:
                break
            if len(content) > remaining:
                content = content[:remaining] + "..."
            lines.extend([header, content, ""])
            total_len += len(content)

        lines.append("（以上为检索到的原始资料，未经过 LLM 生成回答）")
        return "\n".join(lines)
