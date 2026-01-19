"""
HiveMind 全局配置
"""

import os
from pathlib import Path
from pydantic import BaseModel


# 项目根目录 (支持环境变量覆盖)
ROOT_DIR = Path(__file__).parent.parent
_DATA_BASE = os.environ.get("HIVEMIND_DATA_DIR")
if _DATA_BASE:
    DATA_DIR = Path(_DATA_BASE) / "data"
    ADAPTERS_DIR = Path(_DATA_BASE) / "adapters"
    MODELS_DIR = Path(_DATA_BASE) / "models"
else:
    DATA_DIR = ROOT_DIR / "data"
    ADAPTERS_DIR = ROOT_DIR / "adapters"
    MODELS_DIR = ROOT_DIR / "models"


# 服务器地址 (用户可配置)
DEFAULT_SERVER_URL = os.environ.get(
    "HIVEMIND_SERVER_URL",
    "https://hivemind.earthonline-game.cn"
)


class ModelConfig(BaseModel):
    """基础模型配置"""

    name: str = "Qwen/Qwen2-7B"
    cache_dir: Path = MODELS_DIR
    torch_dtype: str = "auto"  # auto, float16, bfloat16
    device_map: str = "auto"
    trust_remote_code: bool = True


class LoRAConfig(BaseModel):
    """LoRA 微调配置"""

    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha
    lora_dropout: float = 0.05
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainingConfig(BaseModel):
    """训练配置"""

    output_dir: Path = ADAPTERS_DIR / "my_adapter"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 100

    # 量化配置 (QLoRA)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"


class ServerConfig(BaseModel):
    """服务端配置"""

    host: str = "0.0.0.0"
    port: int = 8000
    adapters_storage: Path = ADAPTERS_DIR / "server"
    min_adapters_for_aggregation: int = 2  # 最少需要多少个 adapter 才能聚合


class ClientConfig(BaseModel):
    """客户端配置"""

    server_url: str = DEFAULT_SERVER_URL
    user_id: str = "default_user"
    adapter_dir: Path = ADAPTERS_DIR / "local"


# 默认配置实例
model_config = ModelConfig()
lora_config = LoRAConfig()
training_config = TrainingConfig()
server_config = ServerConfig()
client_config = ClientConfig()


def ensure_dirs():
    """确保必要的目录存在"""
    for dir_path in [DATA_DIR, ADAPTERS_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
