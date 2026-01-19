"""
Adapter 存储管理
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from hivemind.config import server_config


class AdapterMetadata(BaseModel):
    """Adapter 元数据"""

    user_id: str
    base_model: str
    version: str = "1.0.0"
    uploaded_at: str = ""
    training_samples: int = 0
    epochs: int = 0

    def model_post_init(self, __context):
        if not self.uploaded_at:
            self.uploaded_at = datetime.now().isoformat()


class AdapterStorage:
    """管理上传的 LoRA Adapter"""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or server_config.adapters_storage
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_dir / "registry.json"
        self._load_registry()

    def _load_registry(self):
        """加载 adapter 注册表"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.registry = json.load(f)
        else:
            self.registry = {"adapters": {}, "aggregated_versions": []}

    def _save_registry(self):
        """保存 adapter 注册表"""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def save_adapter(
        self,
        user_id: str,
        adapter_path: Path,
        metadata: AdapterMetadata,
    ) -> str:
        """
        保存用户上传的 adapter

        Args:
            user_id: 用户ID
            adapter_path: adapter 文件路径 (临时上传路径)
            metadata: adapter 元数据

        Returns:
            保存后的 adapter ID
        """
        # 创建用户目录
        user_dir = self.storage_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        # 生成版本号
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        adapter_id = f"{user_id}_{timestamp}"
        dest_dir = user_dir / timestamp

        # 复制 adapter 文件
        if adapter_path.is_dir():
            shutil.copytree(adapter_path, dest_dir)
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(adapter_path, dest_dir / adapter_path.name)

        # 保存元数据
        metadata_path = dest_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(metadata.model_dump_json(indent=2))

        # 更新注册表
        self.registry["adapters"][adapter_id] = {
            "user_id": user_id,
            "path": str(dest_dir),
            "metadata": metadata.model_dump(),
        }
        self._save_registry()

        return adapter_id

    def get_adapter(self, adapter_id: str) -> Optional[dict]:
        """获取 adapter 信息"""
        return self.registry["adapters"].get(adapter_id)

    def list_adapters(self, user_id: Optional[str] = None) -> list[dict]:
        """列出所有 adapter"""
        adapters = []
        for aid, info in self.registry["adapters"].items():
            if user_id is None or info["user_id"] == user_id:
                adapters.append({"id": aid, **info})
        return adapters

    def get_pending_adapters(self) -> list[dict]:
        """获取待聚合的 adapter (每个用户最新的一个)"""
        latest_by_user = {}
        for aid, info in self.registry["adapters"].items():
            user_id = info["user_id"]
            if user_id not in latest_by_user:
                latest_by_user[user_id] = {"id": aid, **info}
            else:
                # 比较时间戳，保留最新的
                current_time = info["metadata"]["uploaded_at"]
                existing_time = latest_by_user[user_id]["metadata"]["uploaded_at"]
                if current_time > existing_time:
                    latest_by_user[user_id] = {"id": aid, **info}

        return list(latest_by_user.values())

    def save_aggregated(self, aggregated_path: Path, version: str) -> str:
        """保存聚合后的 adapter"""
        dest_dir = self.storage_dir / "aggregated" / version
        dest_dir.mkdir(parents=True, exist_ok=True)

        if aggregated_path.is_dir():
            shutil.copytree(aggregated_path, dest_dir, dirs_exist_ok=True)
        else:
            shutil.copy(aggregated_path, dest_dir)

        self.registry["aggregated_versions"].append(
            {"version": version, "path": str(dest_dir), "created_at": datetime.now().isoformat()}
        )
        self._save_registry()

        return version

    def get_latest_aggregated(self) -> Optional[dict]:
        """获取最新的聚合版本"""
        if not self.registry["aggregated_versions"]:
            return None
        return self.registry["aggregated_versions"][-1]
