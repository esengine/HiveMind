#!/usr/bin/env python
"""
HiveMind 交互对话脚本

使用方法:
    python scripts/chat.py
    python scripts/chat.py --adapter ./adapters/my_adapter
    python scripts/chat.py --model Qwen/Qwen2-7B --no-4bit
"""

import argparse
from pathlib import Path

from hivemind.client.inference import HiveMindChat
from hivemind.config import model_config


def main():
    parser = argparse.ArgumentParser(description="HiveMind Chat")

    parser.add_argument(
        "--model",
        type=str,
        default=model_config.name,
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="你是 HiveMind，一个可以自我进化的个性化 AI 助手。你友好、有帮助，并且会记住用户的偏好。",
        help="System prompt",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )

    args = parser.parse_args()

    # 创建聊天实例
    adapter_path = Path(args.adapter) if args.adapter else None

    chat = HiveMindChat(
        model_name=args.model,
        adapter_path=adapter_path,
        use_4bit=not args.no_4bit,
    )

    # 开始对话
    chat.chat(system_prompt=args.system)


if __name__ == "__main__":
    main()
