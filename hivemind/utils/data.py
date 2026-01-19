"""
数据处理工具
"""

import json
from pathlib import Path
from typing import Optional


def load_json_data(path: Path) -> list:
    """加载 JSON 数据文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_data(data: list, path: Path):
    """保存 JSON 数据文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_sample_dataset(output_path: Optional[Path] = None) -> Path:
    """
    创建示例训练数据集

    Args:
        output_path: 输出路径

    Returns:
        数据集路径
    """
    from hivemind.config import DATA_DIR

    output_path = output_path or (DATA_DIR / "sample_dataset.json")

    sample_data = [
        {
            "instruction": "你是谁？",
            "input": "",
            "output": "我是 HiveMind，一个可以自我进化的个性化 AI 助手。我通过学习用户的偏好和知识来不断成长，同时汇聚所有用户的智慧变得更加强大。",
        },
        {
            "instruction": "介绍一下你的特点",
            "input": "",
            "output": "我有以下特点：\n1. 个性化学习：我会根据你的对话和偏好进行学习，变得更懂你\n2. 持续进化：通过联邦学习，我可以从所有用户那里学习，不断变强\n3. 隐私保护：你的数据只在本地，只有学习到的知识会被分享\n4. 独特个性：每个用户的我都是独一无二的",
        },
        {
            "instruction": "解释什么是联邦学习",
            "input": "",
            "output": "联邦学习是一种分布式机器学习方法。简单来说：\n\n1. 每个用户在本地训练模型\n2. 只上传模型参数（而非原始数据）到服务器\n3. 服务器聚合所有用户的参数\n4. 将聚合后的模型分发给所有用户\n\n这样既能利用大家的数据改进模型，又保护了隐私。",
        },
        {
            "instruction": "写一个 Python 函数计算斐波那契数列",
            "input": "第 n 个斐波那契数",
            "output": '```python\ndef fibonacci(n: int) -> int:\n    """计算第 n 个斐波那契数"""\n    if n <= 0:\n        return 0\n    elif n == 1:\n        return 1\n    \n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n# 示例\nfor i in range(10):\n    print(f"F({i}) = {fibonacci(i)}")\n```',
        },
        {
            "instruction": "用简单的话解释什么是 LoRA",
            "input": "",
            "output": "LoRA (Low-Rank Adaptation) 是一种高效的模型微调方法。\n\n想象一下：\n- 原始大模型就像一本百科全书，有几十亿个参数\n- 传统微调需要修改整本书，成本很高\n- LoRA 只是在书页边上贴便签纸，记录你的个人笔记\n\n这些便签纸（LoRA adapter）只有原模型的 1% 大小，但能有效地让模型学会新知识。训练快、存储小、效果好！",
        },
        {
            "instruction": "如何保持学习的动力？",
            "input": "",
            "output": "保持学习动力的几个建议：\n\n1. **设定小目标**：把大目标拆成小步骤，完成每一步都有成就感\n2. **找到乐趣**：尝试用有趣的方式学习，比如做项目、参加比赛\n3. **结伴学习**：和朋友一起学，互相督促和讨论\n4. **及时奖励**：完成目标后给自己小奖励\n5. **回顾进步**：定期看看自己学了多少，会更有信心\n\n记住：学习不是冲刺，而是马拉松。保持节奏，享受过程。",
        },
    ]

    save_json_data(sample_data, output_path)
    print(f"Sample dataset created at: {output_path}")

    return output_path


def convert_sharegpt_to_alpaca(sharegpt_data: list) -> list:
    """
    将 ShareGPT 格式转换为 Alpaca 格式

    ShareGPT: [{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}]
    Alpaca: [{"instruction": "...", "input": "", "output": "..."}]
    """
    alpaca_data = []

    for item in sharegpt_data:
        conversations = item.get("conversations", [])

        for i in range(0, len(conversations) - 1, 2):
            if conversations[i]["from"] == "human" and conversations[i + 1]["from"] in [
                "gpt",
                "assistant",
            ]:
                alpaca_data.append(
                    {
                        "instruction": conversations[i]["value"],
                        "input": "",
                        "output": conversations[i + 1]["value"],
                    }
                )

    return alpaca_data
