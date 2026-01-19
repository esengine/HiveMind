"""
LoRA Adapter 联邦聚合

实现 FedAvg 算法，将多个用户的 LoRA adapter 聚合成一个更强的版本
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file, save_file

from hivemind.config import server_config


class LoRAAggregator:
    """
    LoRA Adapter 聚合器

    使用 FedAvg (联邦平均) 算法聚合多个用户的 adapter
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (server_config.adapters_storage / "aggregated")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_adapter_weights(self, adapter_path: Path) -> dict[str, torch.Tensor]:
        """
        加载单个 adapter 的权重

        Args:
            adapter_path: adapter 目录路径

        Returns:
            权重字典
        """
        # 查找权重文件
        safetensor_file = adapter_path / "adapter_model.safetensors"
        bin_file = adapter_path / "adapter_model.bin"

        if safetensor_file.exists():
            return load_file(safetensor_file)
        elif bin_file.exists():
            return torch.load(bin_file, map_location="cpu")
        else:
            raise FileNotFoundError(f"No adapter weights found in {adapter_path}")

    def fedavg(
        self,
        adapter_weights_list: list[dict[str, torch.Tensor]],
        weights: Optional[list[float]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        FedAvg 联邦平均聚合

        Args:
            adapter_weights_list: 多个 adapter 的权重列表
            weights: 每个 adapter 的权重 (默认等权重)

        Returns:
            聚合后的权重
        """
        if not adapter_weights_list:
            raise ValueError("No adapters to aggregate")

        n_adapters = len(adapter_weights_list)

        # 默认等权重
        if weights is None:
            weights = [1.0 / n_adapters] * n_adapters

        # 确保权重和为 1
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # 获取第一个 adapter 的键作为参考
        reference_keys = set(adapter_weights_list[0].keys())

        # 聚合
        aggregated = {}
        for key in reference_keys:
            # 加权平均
            weighted_sum = None
            for adapter_weights, weight in zip(adapter_weights_list, weights):
                if key not in adapter_weights:
                    continue

                tensor = adapter_weights[key].float()
                if weighted_sum is None:
                    weighted_sum = tensor * weight
                else:
                    weighted_sum += tensor * weight

            if weighted_sum is not None:
                aggregated[key] = weighted_sum

        return aggregated

    def aggregate(
        self,
        adapter_paths: list[Path],
        version: Optional[str] = None,
        weights: Optional[list[float]] = None,
    ) -> Path:
        """
        聚合多个 adapter

        Args:
            adapter_paths: adapter 目录路径列表
            version: 版本号 (默认使用时间戳)
            weights: 每个 adapter 的权重

        Returns:
            聚合后的 adapter 路径
        """
        if len(adapter_paths) < 2:
            raise ValueError("Need at least 2 adapters to aggregate")

        # 加载所有 adapter 权重
        all_weights = []
        valid_paths = []
        for path in adapter_paths:
            try:
                w = self.load_adapter_weights(Path(path))
                all_weights.append(w)
                valid_paths.append(path)
            except FileNotFoundError as e:
                print(f"Warning: Skipping {path}: {e}")

        if len(all_weights) < 2:
            raise ValueError(f"Only {len(all_weights)} valid adapters found, need at least 2")

        # 执行 FedAvg
        aggregated_weights = self.fedavg(all_weights, weights)

        # 生成版本号
        if version is None:
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")

        # 保存聚合结果
        output_path = self.output_dir / version
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存权重 (safetensors 格式)
        save_file(aggregated_weights, output_path / "adapter_model.safetensors")

        # 复制第一个 adapter 的配置文件
        first_adapter = Path(valid_paths[0])
        config_file = first_adapter / "adapter_config.json"
        if config_file.exists():
            import shutil

            shutil.copy(config_file, output_path / "adapter_config.json")

        # 保存聚合元数据
        metadata = {
            "version": version,
            "aggregated_at": datetime.now().isoformat(),
            "source_adapters": [str(p) for p in valid_paths],
            "num_adapters": len(valid_paths),
            "algorithm": "FedAvg",
        }
        with open(output_path / "aggregation_info.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return output_path
