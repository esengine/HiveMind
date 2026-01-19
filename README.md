# HiveMind

A self-evolving, personalized AI system with federated learning.

[中文文档](./README_CN.md)

## Overview

HiveMind enables each user to have a unique personalized AI assistant while aggregating collective intelligence through federated learning to continuously evolve a more powerful base model.

### Core Concepts

- **Personalization**: Each user owns a unique AI assistant through local training
- **Privacy-Preserving**: Raw data stays local; only model parameters are uploaded
- **Collective Evolution**: Knowledge from all users improves the model through federated aggregation

### Architecture

```
Client                                 Server
┌─────────────────┐                ┌─────────────────┐
│  Base Model     │                │  Aggregator     │
│  (Qwen2-7B)     │  ──Upload───►  │  (FedAvg)       │
│       +         │                │                 │
│  Personal LoRA  │  ◄──Download── │  Model Storage  │
└─────────────────┘                └─────────────────┘
```

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/esengine/HiveMind.git
cd HiveMind

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -e .

# 4. Create sample dataset
python scripts/train.py --create-sample

# 5. Train personalized model (requires GPU)
python scripts/train.py --data ./data/sample_dataset.json

# 6. Test chat
python scripts/chat.py --adapter ./adapters/my_adapter

# 7. Start server (in another terminal)
python scripts/serve.py

# 8. Upload adapter to server
python scripts/upload.py --adapter ./adapters/my_adapter
```

## Requirements

- **Python 3.10 - 3.12** (3.13+ not supported by PyTorch yet)
- NVIDIA GPU with CUDA 11.8+ (recommended)
- Disk: 20GB+ (for base model storage)

### Model Selection by VRAM

| VRAM | Recommended Model | Command |
|------|-------------------|---------|
| 4-6 GB | Qwen2-1.5B | `--model Qwen/Qwen2-1.5B` |
| 8-10 GB | Qwen2-7B (default) | `--model Qwen/Qwen2-7B` |
| 16+ GB | Qwen2-7B (full) | `--model Qwen/Qwen2-7B --no-4bit` |

## Installation

### Windows (PowerShell)

```powershell
# Clone repository
git clone https://github.com/esengine/HiveMind.git
cd HiveMind

# Create virtual environment (use Python 3.11 specifically)
py -3.11 -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA support (for GPU training)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install HiveMind
pip install -e .

# Verify GPU is detected
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Linux/macOS

```bash
# Clone repository
git clone https://github.com/esengine/HiveMind.git
cd HiveMind

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support (Linux with NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Or CPU only (macOS / Linux without GPU)
pip install torch

# Install HiveMind
pip install -e .
```

## Detailed Guide

### 1. Prepare Training Data

Create a JSON file in Alpaca format:

```json
[
  {
    "instruction": "Your question or instruction",
    "input": "Optional additional input",
    "output": "Expected response"
  }
]
```

Or generate sample data using the built-in command:

```bash
python scripts/train.py --create-sample
```

### 2. Local Training

```bash
# Basic training
python scripts/train.py --data ./data/sample_dataset.json

# Custom parameters
python scripts/train.py \
  --data ./data/my_data.json \
  --output ./adapters/my_adapter \
  --epochs 3 \
  --batch-size 4
```

Training parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | Required | Training data path |
| `--output` | `./adapters/my_adapter` | Adapter output directory |
| `--model` | `Qwen/Qwen2-7B` | Base model |
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 4 | Batch size |
| `--no-4bit` | False | Disable 4-bit quantization |

### 3. Chat Testing

```bash
# Use trained adapter
python scripts/chat.py --adapter ./adapters/my_adapter

# Use base model only
python scripts/chat.py

# Custom system prompt
python scripts/chat.py --adapter ./adapters/my_adapter --system "You are a professional coding assistant"
```

Chat commands:
- Type and press Enter to send
- `clear` - Clear conversation history
- `exit` - Exit chat

### 4. Start Server

```bash
# Default configuration
python scripts/serve.py

# Custom address
python scripts/serve.py --host 0.0.0.0 --port 8000

# Development mode (auto-reload)
python scripts/serve.py --reload
```

Visit `http://localhost:8000/docs` for API documentation.

### 5. Upload Adapter

```bash
# Upload to server
python scripts/upload.py --adapter ./adapters/my_adapter

# Specify server address
python scripts/upload.py \
  --adapter ./adapters/my_adapter \
  --server http://your-server:8000 \
  --user-id your_username

# Check server status
python scripts/upload.py --status

# Download latest aggregated model
python scripts/upload.py --download ./adapters/latest
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/status` | GET | Get server status |
| `/api/v1/adapter/upload` | POST | Upload adapter |
| `/api/v1/adapter/list` | GET | List all adapters |
| `/api/v1/adapter/download/latest` | GET | Download latest aggregated version |
| `/api/v1/aggregate` | POST | Trigger aggregation |

## Project Structure

```
HiveMind/
├── hivemind/
│   ├── config.py          # Global configuration
│   ├── client/
│   │   ├── trainer.py     # LoRA training
│   │   ├── inference.py   # Inference and chat
│   │   └── uploader.py    # Upload module
│   ├── server/
│   │   ├── app.py         # FastAPI application
│   │   ├── aggregator.py  # FedAvg aggregation
│   │   └── storage.py     # Storage management
│   ├── models/
│   │   ├── base.py        # Model loading
│   │   └── lora.py        # LoRA utilities
│   └── utils/
│       └── data.py        # Data processing
├── scripts/
│   ├── train.py           # Training script
│   ├── chat.py            # Chat script
│   ├── serve.py           # Server script
│   └── upload.py          # Upload script
├── data/                  # Training data directory
├── adapters/              # Adapter storage directory
└── tests/                 # Tests
```

## Configuration

Main configuration is in `hivemind/config.py`:

```python
# Model configuration
BASE_MODEL = "Qwen/Qwen2-7B"

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training configuration
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

# Server configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
```

## FAQ

**Q: Out of memory?**

4-bit quantization is enabled by default, reducing VRAM requirements to 8GB. If still insufficient, try reducing `batch_size` or `max_seq_length`.

**Q: Training is slow?**

Ensure CUDA and PyTorch GPU version are properly installed. CPU training is extremely slow and not recommended.

**Q: How to use a custom model?**

Modify the `--model` parameter to point to a local model path or HuggingFace model name.

**Q: How large are adapter files?**

Typically 50-100MB, depending on LoRA rank settings.

## Tech Stack

- Base Model: Qwen2-7B
- Fine-tuning: LoRA / QLoRA (PEFT)
- Inference: Transformers
- Server: FastAPI
- Aggregation: FedAvg

## Roadmap

- [ ] Differential privacy protection
- [ ] Byzantine-robust aggregation algorithms
- [ ] Web UI interface
- [ ] Quantized deployment (GGUF)
- [ ] Continual learning with catastrophic forgetting prevention

## License

MIT License
