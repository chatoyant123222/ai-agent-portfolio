"""
DeepSeek API 测试脚本
功能：验证环境变量加载并发送一条简单的对话请求
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


def load_env():
    """从项目根目录加载 .env 文件"""
    # 计算项目根目录（backend 的父目录）
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"✅ 已加载环境变量: {env_path}")
    else:
        load_dotenv()  # 尝试系统默认方式
        print(f"⚠️ 未找到 {env_path}，尝试默认加载")


def test_api():
    """测试 DeepSeek API 连通性"""
    load_env()
    
    # 读取配置
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # 检查 API Key
    if not api_key:
        print("\n❌ 错误: DEEPSEEK_API_KEY 未设置")
        print("请确认项目根目录的 .env 文件包含：")
        print("  DEEPSEEK_API_KEY=sk-你的密钥")
        sys.exit(1)
    
    print(f"\n🔑 API Key: {api_key[:10]}...")
    print(f"🔗 Base URL: {base_url}")
    print(f"🤖 Model: {model}")
    
    # 初始化客户端
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 构造对话
    messages = [
        {"role": "system", "content": "你是一个简洁的技术助手。"},
        {"role": "user", "content": "用一句话介绍 DeepSeek。"}
    ]
    
    print("\n📡 正在发送请求...")
    
    try:
        # 发送非流式请求（适合新手调试）
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        
        # 解析结果
        content = response.choices[0].message.content
        usage = response.usage
        
        print("\n" + "=" * 40)
        print("✅ 请求成功！")
        print("=" * 40)
        print(f"\n💬 AI 回复：\n{content}")
        
        if usage:
            print(f"\n📊 Token 消耗：")
            print(f"   输入: {usage.prompt_tokens}")
            print(f"   输出: {usage.completion_tokens}")
            print(f"   总计: {usage.total_tokens}")
            
    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_api()