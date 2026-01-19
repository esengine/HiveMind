"""
Adapter 上传模块

将本地训练的 LoRA adapter 上传到 HiveMind 服务器
"""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import httpx

from hivemind.config import client_config


class AdapterUploader:
    """上传 LoRA Adapter 到服务器"""

    def __init__(
        self,
        server_url: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.server_url = server_url or client_config.server_url
        self.user_id = user_id or client_config.user_id

    def _create_zip(self, adapter_path: Path) -> Path:
        """将 adapter 目录打包为 zip"""
        tmp_dir = Path(tempfile.mkdtemp())
        zip_path = tmp_dir / "adapter.zip"

        shutil.make_archive(
            str(zip_path.with_suffix("")),
            "zip",
            adapter_path,
        )

        return zip_path

    def upload(
        self,
        adapter_path: Path,
        base_model: str = "Qwen/Qwen2-7B",
        training_samples: int = 0,
        epochs: int = 0,
    ) -> dict:
        """
        上传 adapter 到服务器

        Args:
            adapter_path: adapter 目录路径
            base_model: 基础模型名称
            training_samples: 训练样本数
            epochs: 训练轮数

        Returns:
            服务器响应
        """
        adapter_path = Path(adapter_path)

        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter not found: {adapter_path}")

        # 检查是目录还是文件
        if adapter_path.is_dir():
            # 打包为 zip
            print(f"Packaging adapter from: {adapter_path}")
            zip_path = self._create_zip(adapter_path)
            file_to_upload = zip_path
        else:
            file_to_upload = adapter_path

        # 上传
        print(f"Uploading to: {self.server_url}")

        try:
            with open(file_to_upload, "rb") as f:
                files = {"file": (file_to_upload.name, f, "application/octet-stream")}
                data = {
                    "user_id": self.user_id,
                    "base_model": base_model,
                    "training_samples": str(training_samples),
                    "epochs": str(epochs),
                }

                response = httpx.post(
                    f"{self.server_url}/api/v1/adapter/upload",
                    files=files,
                    data=data,
                    timeout=300.0,  # 5分钟超时
                )

            response.raise_for_status()
            result = response.json()

            print(f"Upload successful! Adapter ID: {result.get('adapter_id')}")
            return result

        except httpx.HTTPError as e:
            print(f"Upload failed: {e}")
            raise

        finally:
            # 清理临时文件
            if adapter_path.is_dir():
                zip_path.parent.rmdir()

    def download_latest(self, output_path: Path) -> Path:
        """
        下载最新的聚合 adapter

        Args:
            output_path: 保存路径

        Returns:
            下载的文件路径
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Downloading latest aggregated adapter...")

        try:
            with httpx.stream(
                "GET",
                f"{self.server_url}/api/v1/adapter/download/latest",
                timeout=300.0,
            ) as response:
                response.raise_for_status()

                # 从 header 获取文件名
                content_disposition = response.headers.get("content-disposition", "")
                if "filename=" in content_disposition:
                    filename = content_disposition.split("filename=")[1].strip('"')
                else:
                    filename = "adapter_model.safetensors"

                file_path = output_path / filename

                with open(file_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)

            print(f"Downloaded to: {file_path}")
            return file_path

        except httpx.HTTPError as e:
            print(f"Download failed: {e}")
            raise

    def get_server_status(self) -> dict:
        """获取服务器状态"""
        try:
            response = httpx.get(
                f"{self.server_url}/api/v1/status",
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            print(f"Failed to get server status: {e}")
            raise
