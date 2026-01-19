"""
LoRA Adapter 联邦聚合

实现 FedAvg 算法，将多个用户的 LoRA adapter 聚合成一个更强的版本

内存优化版：支持在低内存服务器上运行 (2GB RAM)
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from hivemind.config import server_config


# 延迟导入 torch，只在聚合时才需要
_torch = None
_safetensors = None


def _lazy_import_torch():
    """延迟导入 torch 和 safetensors"""
    global _torch, _safetensors
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError(
                "torch is required for aggregation. "
                "Install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )
    if _safetensors is None:
        try:
            from safetensors import safe_open
            from safetensors.torch import save_file
            _safetensors = {"safe_open": safe_open, "save_file": save_file}
        except ImportError:
            raise ImportError(
                "safetensors is required for aggregation. "
                "Install with: pip install safetensors"
            )
    return _torch, _safetensors


class LoRAAggregator:
    """
    LoRA Adapter 聚合器

    使用 FedAvg (联邦平均) 算法聚合多个用户的 adapter

    内存优化：逐层加载和聚合，避免一次性加载所有权重
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (server_config.adapters_storage / "aggregated")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._torch_available = None

    def is_torch_available(self) -> bool:
        """检查 torch 是否可用"""
        if self._torch_available is None:
            try:
                import torch
                self._torch_available = True
            except ImportError:
                self._torch_available = False
        return self._torch_available

    def get_adapter_keys(self, adapter_path: Path) -> list[str]:
        """获取 adapter 的所有 key（不加载权重）"""
        torch, safetensors = _lazy_import_torch()

        safetensor_file = adapter_path / "adapter_model.safetensors"
        if safetensor_file.exists():
            with safetensors["safe_open"](safetensor_file, framework="pt") as f:
                return list(f.keys())

        bin_file = adapter_path / "adapter_model.bin"
        if bin_file.exists():
            # 对于 .bin 文件，需要加载才能获取 keys
            data = torch.load(bin_file, map_location="cpu")
            keys = list(data.keys())
            del data
            return keys

        raise FileNotFoundError(f"No adapter weights found in {adapter_path}")

    def load_single_key(self, adapter_path: Path, key: str) -> Any:
        """只加载单个 key 的权重（内存优化）"""
        torch, safetensors = _lazy_import_torch()

        safetensor_file = adapter_path / "adapter_model.safetensors"
        if safetensor_file.exists():
            with safetensors["safe_open"](safetensor_file, framework="pt") as f:
                return f.get_tensor(key)

        bin_file = adapter_path / "adapter_model.bin"
        if bin_file.exists():
            data = torch.load(bin_file, map_location="cpu")
            tensor = data[key]
            del data
            return tensor

        raise FileNotFoundError(f"No adapter weights found in {adapter_path}")

    def aggregate_memory_optimized(
        self,
        adapter_paths: list[Path],
        version: Optional[str] = None,
        weights: Optional[list[float]] = None,
    ) -> Path:
        """
        内存优化版聚合：逐层处理，适合低内存服务器

        Args:
            adapter_paths: adapter 目录路径列表
            version: 版本号 (默认使用时间戳)
            weights: 每个 adapter 的权重

        Returns:
            聚合后的 adapter 路径
        """
        torch, safetensors = _lazy_import_torch()

        if len(adapter_paths) < 2:
            raise ValueError("Need at least 2 adapters to aggregate")

        # 验证所有路径
        valid_paths = []
        for path in adapter_paths:
            path = Path(path)
            safetensor_file = path / "adapter_model.safetensors"
            bin_file = path / "adapter_model.bin"
            if safetensor_file.exists() or bin_file.exists():
                valid_paths.append(path)
            else:
                print(f"Warning: Skipping {path}: No adapter weights found")

        if len(valid_paths) < 2:
            raise ValueError(f"Only {len(valid_paths)} valid adapters found, need at least 2")

        n_adapters = len(valid_paths)

        # 默认等权重
        if weights is None:
            weights = [1.0 / n_adapters] * n_adapters
        else:
            # 确保权重和为 1
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

        # 获取所有 keys
        all_keys = self.get_adapter_keys(valid_paths[0])

        # 生成版本号
        if version is None:
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")

        # 输出路径
        output_path = self.output_dir / version
        output_path.mkdir(parents=True, exist_ok=True)

        # 逐层聚合（内存优化核心）
        aggregated = {}
        for key in all_keys:
            weighted_sum = None

            for adapter_path, weight in zip(valid_paths, weights):
                try:
                    tensor = self.load_single_key(adapter_path, key).float()

                    if weighted_sum is None:
                        weighted_sum = tensor * weight
                    else:
                        weighted_sum += tensor * weight

                    # 及时释放内存
                    del tensor

                except Exception as e:
                    print(f"Warning: Failed to load key {key} from {adapter_path}: {e}")
                    continue

            if weighted_sum is not None:
                aggregated[key] = weighted_sum

            # 每处理 10 个 key 打印进度
            if len(aggregated) % 10 == 0:
                print(f"Aggregated {len(aggregated)}/{len(all_keys)} keys...")

        # 保存聚合结果
        safetensors["save_file"](aggregated, output_path / "adapter_model.safetensors")

        # 释放内存
        del aggregated

        # 复制第一个 adapter 的配置文件
        config_file = valid_paths[0] / "adapter_config.json"
        if config_file.exists():
            shutil.copy(config_file, output_path / "adapter_config.json")

        # 保存聚合元数据
        metadata = {
            "version": version,
            "aggregated_at": datetime.now().isoformat(),
            "source_adapters": [str(p) for p in valid_paths],
            "num_adapters": len(valid_paths),
            "algorithm": "FedAvg",
            "memory_optimized": True,
        }
        with open(output_path / "aggregation_info.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Aggregation complete: {output_path}")
        return output_path

    # 保留原有接口，但使用内存优化版本
    def aggregate(
        self,
        adapter_paths: list[Path],
        version: Optional[str] = None,
        weights: Optional[list[float]] = None,
    ) -> Path:
        """聚合多个 adapter（自动使用内存优化版本）"""
        return self.aggregate_memory_optimized(adapter_paths, version, weights)
