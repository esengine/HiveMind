# HiveMind

一个可自我进化、个性化学习的分布式 AI 系统。

[English](./README.md)

## 概述

HiveMind 让每个用户拥有独特的个性化 AI，同时通过联邦学习将所有用户的智慧汇聚，持续进化出更强大的基础模型。

### 核心理念

- **个性化**: 每个用户通过本地训练拥有独特的 AI 助手
- **隐私保护**: 原始数据不离开本地，只上传模型参数
- **集体进化**: 通过联邦聚合，所有用户的知识共同提升模型能力

### 技术架构

```
用户端                                服务端
┌─────────────────┐                ┌─────────────────┐
│  Base Model     │                │  Aggregator     │
│  (Qwen2-7B)     │  ──上传LoRA──► │  (FedAvg)       │
│       +         │                │                 │
│  Personal LoRA  │  ◄──下载更新── │  Model Storage  │
└─────────────────┘                └─────────────────┘
```

## 快速开始

```bash
# 1. 克隆并进入项目
git clone https://github.com/esengine/HiveMind.git
cd HiveMind

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 3. 安装依赖
pip install -e .

# 4. 创建示例数据集
python scripts/train.py --create-sample

# 5. 训练个性化模型 (需要 GPU)
python scripts/train.py --data ./data/sample_dataset.json

# 6. 测试对话
python scripts/chat.py --adapter ./adapters/my_adapter

# 7. 启动服务端 (另开终端)
python scripts/serve.py

# 8. 上传 adapter 到服务器
python scripts/upload.py --adapter ./adapters/my_adapter
```

## 环境要求

- **Python 3.10 - 3.12** (3.13+ 暂不支持，PyTorch 尚未适配)
- NVIDIA 显卡 + CUDA 11.8+ (推荐)
- 显存: 8GB+ (QLoRA) / 16GB+ (LoRA)
- 磁盘: 20GB+ (用于存储基础模型)

## 安装

### Windows (PowerShell)

```powershell
# 克隆仓库
git clone https://github.com/esengine/HiveMind.git
cd HiveMind

# 创建虚拟环境 (指定 Python 3.11)
py -3.11 -m venv venv
venv\Scripts\activate

# 安装 PyTorch GPU 版本
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 安装 HiveMind
pip install -e .

# 验证 GPU 是否识别
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Linux/macOS

```bash
# 克隆仓库
git clone https://github.com/esengine/HiveMind.git
cd HiveMind

# 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate

# 安装 PyTorch GPU 版本 (Linux + NVIDIA 显卡)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 或 CPU 版本 (macOS / 无显卡)
pip install torch

# 安装 HiveMind
pip install -e .
```

## 详细说明

### 1. 准备训练数据

创建 JSON 文件，支持 Alpaca 格式:

```json
[
  {
    "instruction": "你的问题或指令",
    "input": "可选的额外输入",
    "output": "期望的回答"
  }
]
```

或使用内置命令生成示例数据:

```bash
python scripts/train.py --create-sample
```

### 2. 本地训练

```bash
# 基础训练
python scripts/train.py --data ./data/sample_dataset.json

# 自定义参数
python scripts/train.py \
  --data ./data/my_data.json \
  --output ./adapters/my_adapter \
  --epochs 3 \
  --batch-size 4
```

训练参数说明:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | 必填 | 训练数据路径 |
| `--output` | `./adapters/my_adapter` | Adapter 输出目录 |
| `--model` | `Qwen/Qwen2-7B` | 基础模型 |
| `--epochs` | 3 | 训练轮数 |
| `--batch-size` | 4 | 批次大小 |
| `--no-4bit` | False | 禁用 4bit 量化 |

### 3. 对话测试

```bash
# 使用训练好的 adapter
python scripts/chat.py --adapter ./adapters/my_adapter

# 仅使用基础模型
python scripts/chat.py

# 自定义系统提示
python scripts/chat.py --adapter ./adapters/my_adapter --system "你是一个专业的编程助手"
```

对话命令:
- 输入文字后回车发送
- `clear` - 清除对话历史
- `exit` - 退出

### 4. 启动服务端

```bash
# 默认配置
python scripts/serve.py

# 自定义地址
python scripts/serve.py --host 0.0.0.0 --port 8000

# 开发模式 (自动重载)
python scripts/serve.py --reload
```

服务启动后访问 `http://localhost:8000/docs` 查看 API 文档。

### 5. 上传 Adapter

```bash
# 上传到服务器
python scripts/upload.py --adapter ./adapters/my_adapter

# 指定服务器地址
python scripts/upload.py \
  --adapter ./adapters/my_adapter \
  --server http://your-server:8000 \
  --user-id your_username

# 查看服务器状态
python scripts/upload.py --status

# 下载最新聚合模型
python scripts/upload.py --download ./adapters/latest
```

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/status` | GET | 获取服务状态 |
| `/api/v1/adapter/upload` | POST | 上传 Adapter |
| `/api/v1/adapter/list` | GET | 列出所有 Adapter |
| `/api/v1/adapter/download/latest` | GET | 下载最新聚合版本 |
| `/api/v1/aggregate` | POST | 触发聚合 |

## 项目结构

```
HiveMind/
├── hivemind/
│   ├── config.py          # 全局配置
│   ├── client/
│   │   ├── trainer.py     # LoRA 训练
│   │   ├── inference.py   # 推理对话
│   │   └── uploader.py    # 上传模块
│   ├── server/
│   │   ├── app.py         # FastAPI 应用
│   │   ├── aggregator.py  # FedAvg 聚合
│   │   └── storage.py     # 存储管理
│   ├── models/
│   │   ├── base.py        # 模型加载
│   │   └── lora.py        # LoRA 工具
│   └── utils/
│       └── data.py        # 数据处理
├── scripts/
│   ├── train.py           # 训练脚本
│   ├── chat.py            # 对话脚本
│   ├── serve.py           # 服务脚本
│   └── upload.py          # 上传脚本
├── data/                  # 训练数据目录
├── adapters/              # Adapter 存储目录
└── tests/                 # 测试
```

## 配置说明

主要配置位于 `hivemind/config.py`:

```python
# 模型配置
BASE_MODEL = "Qwen/Qwen2-7B"

# LoRA 配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# 训练配置
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

# 服务配置
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
```

## 常见问题

**Q: 显存不足怎么办?**

启用 4bit 量化 (默认已启用)，可将显存需求降至 8GB。如仍不足，可尝试减小 `batch_size` 或 `max_seq_length`。

**Q: 训练速度很慢?**

确保已正确安装 CUDA 和 PyTorch GPU 版本。CPU 训练会非常慢，不建议使用。

**Q: 如何使用自己的模型?**

修改 `--model` 参数指向本地模型路径或 HuggingFace 模型名称。

**Q: Adapter 文件有多大?**

通常 50-100MB，取决于 LoRA rank 设置。

## 技术栈

- 基础模型: Qwen2-7B
- 微调方法: LoRA / QLoRA (PEFT)
- 推理框架: Transformers
- 服务框架: FastAPI
- 聚合算法: FedAvg

## 开发计划

- [ ] 差分隐私保护
- [ ] Byzantine-robust 聚合算法
- [ ] Web UI 界面
- [ ] 模型量化部署 (GGUF)
- [ ] 持续学习防遗忘机制

## 许可证

MIT License
