#!/usr/bin/env python
"""
HiveMind 本地训练脚本

使用方法:
    python scripts/train.py --data ./data/my_conversations.json
    python scripts/train.py --data ./data/sample.json --epochs 3 --output ./adapters/my_adapter
"""

import argparse
from pathlib import Path

from hivemind.client.trainer import LoRATrainer
from hivemind.config import training_config, ensure_dirs
from hivemind.utils.data import create_sample_dataset


def main():
    parser = argparse.ArgumentParser(description="HiveMind Local Training")

    parser.add_argument(
        "--data",
        type=str,
        help="Path to training data (JSON file in Alpaca or ShareGPT format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(training_config.output_dir),
        help="Output directory for the trained adapter",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-7B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=training_config.num_epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=training_config.batch_size,
        help="Batch size",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (requires more VRAM)",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample dataset and exit",
    )

    args = parser.parse_args()

    # 确保目录存在
    ensure_dirs()

    # 创建示例数据集
    if args.create_sample:
        sample_path = create_sample_dataset()
        print(f"\nSample dataset created at: {sample_path}")
        print("You can now run: python scripts/train.py --data " + str(sample_path))
        return

    # 检查数据文件
    if not args.data:
        print("Error: --data is required. Use --create-sample to create a sample dataset.")
        return

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return

    # 更新训练配置
    training_config.output_dir = Path(args.output)
    training_config.num_epochs = args.epochs
    training_config.batch_size = args.batch_size

    # 创建训练器
    trainer = LoRATrainer(
        model_name=args.model,
        output_dir=Path(args.output),
        use_4bit=not args.no_4bit,
    )

    # 开始训练
    print("\n" + "=" * 50)
    print("HiveMind Training")
    print("=" * 50)
    print(f"Data: {data_path}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"4-bit: {not args.no_4bit}")
    print("=" * 50 + "\n")

    adapter_path = trainer.train(data_path)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Adapter saved to: {adapter_path}")
    print("=" * 50)
    print("\nNext steps:")
    print(f"  1. Test: python scripts/chat.py --adapter {adapter_path}")
    print(f"  2. Upload: python scripts/upload.py --adapter {adapter_path}")


if __name__ == "__main__":
    main()
