"""
基础模型加载工具
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from hivemind.config import model_config


def load_tokenizer(
    model_name: Optional[str] = None,
    padding_side: str = "right",
) -> AutoTokenizer:
    """
    加载 tokenizer

    Args:
        model_name: 模型名称或路径
        padding_side: padding 方向

    Returns:
        AutoTokenizer
    """
    model_name = model_name or model_config.name

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.trust_remote_code,
        padding_side=padding_side,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_base_model(
    model_name: Optional[str] = None,
    use_4bit: bool = True,
    use_8bit: bool = False,
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """
    加载基础模型

    Args:
        model_name: 模型名称或路径
        use_4bit: 是否使用 4bit 量化
        use_8bit: 是否使用 8bit 量化
        device_map: 设备映射

    Returns:
        AutoModelForCausalLM
    """
    model_name = model_name or model_config.name

    # 量化配置
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.float16 if bnb_config is None else None,
    )

    return model
