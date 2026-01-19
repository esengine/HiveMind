"""
LoRA 配置和工具
"""

from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from hivemind.config import lora_config as default_lora_config


def create_lora_config(
    r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    target_modules: Optional[list[str]] = None,
) -> LoraConfig:
    """
    创建 LoRA 配置

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: Dropout 率
        target_modules: 目标模块

    Returns:
        LoraConfig
    """
    return LoraConfig(
        r=r or default_lora_config.r,
        lora_alpha=lora_alpha or default_lora_config.lora_alpha,
        lora_dropout=lora_dropout or default_lora_config.lora_dropout,
        target_modules=target_modules or default_lora_config.target_modules,
        bias=default_lora_config.bias,
        task_type=default_lora_config.task_type,
    )


def apply_lora(model, lora_config: Optional[LoraConfig] = None):
    """
    应用 LoRA 到模型

    Args:
        model: 基础模型
        lora_config: LoRA 配置

    Returns:
        PeftModel
    """
    if lora_config is None:
        lora_config = create_lora_config()

    return get_peft_model(model, lora_config)


def load_lora_adapter(model, adapter_path: Path) -> PeftModel:
    """
    加载 LoRA adapter

    Args:
        model: 基础模型
        adapter_path: adapter 路径

    Returns:
        PeftModel
    """
    return PeftModel.from_pretrained(model, adapter_path)


def merge_lora_weights(model: PeftModel, output_path: Optional[Path] = None):
    """
    将 LoRA 权重合并到基础模型

    注意：合并后模型会变大，但推理更快

    Args:
        model: PeftModel
        output_path: 输出路径 (可选)

    Returns:
        合并后的模型
    """
    merged_model = model.merge_and_unload()

    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(output_path)
        print(f"Merged model saved to: {output_path}")

    return merged_model
