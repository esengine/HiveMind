"""
HiveMind 服务端 FastAPI 应用
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from hivemind.config import server_config
from hivemind.server.aggregator import LoRAAggregator
from hivemind.server.storage import AdapterMetadata, AdapterStorage


class UploadResponse(BaseModel):
    """上传响应"""

    success: bool
    adapter_id: str
    message: str


class AggregateResponse(BaseModel):
    """聚合响应"""

    success: bool
    version: str
    num_adapters: int
    message: str


class StatusResponse(BaseModel):
    """状态响应"""

    total_adapters: int
    total_users: int
    latest_aggregated_version: Optional[str]
    pending_for_aggregation: int


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""

    app = FastAPI(
        title="HiveMind Server",
        description="蜂巢智慧 - 联邦学习 LoRA 聚合服务",
        version="0.1.0",
    )

    # 初始化存储和聚合器
    storage = AdapterStorage()
    aggregator = LoRAAggregator()

    @app.get("/")
    async def root():
        return {"message": "HiveMind Server is running", "version": "0.1.0"}

    @app.get("/api/v1/status", response_model=StatusResponse)
    async def get_status():
        """获取服务状态"""
        adapters = storage.list_adapters()
        users = set(a["user_id"] for a in adapters)
        latest = storage.get_latest_aggregated()
        pending = storage.get_pending_adapters()

        return StatusResponse(
            total_adapters=len(adapters),
            total_users=len(users),
            latest_aggregated_version=latest["version"] if latest else None,
            pending_for_aggregation=len(pending),
        )

    @app.post("/api/v1/adapter/upload", response_model=UploadResponse)
    async def upload_adapter(
        file: UploadFile = File(...),
        user_id: str = Form(...),
        base_model: str = Form("Qwen/Qwen2-7B"),
        training_samples: int = Form(0),
        epochs: int = Form(0),
    ):
        """
        上传 LoRA Adapter

        接受 .safetensors 文件或包含 adapter 的 .zip 文件
        """
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / file.filename

            # 保存上传的文件
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # 如果是 zip 文件，解压
            if file.filename.endswith(".zip"):
                shutil.unpack_archive(file_path, tmp_path / "extracted")
                adapter_path = tmp_path / "extracted"
            else:
                adapter_path = file_path

            # 创建元数据
            metadata = AdapterMetadata(
                user_id=user_id,
                base_model=base_model,
                training_samples=training_samples,
                epochs=epochs,
            )

            # 保存到存储
            adapter_id = storage.save_adapter(user_id, adapter_path, metadata)

        return UploadResponse(
            success=True,
            adapter_id=adapter_id,
            message=f"Adapter uploaded successfully",
        )

    @app.get("/api/v1/adapter/list")
    async def list_adapters(user_id: Optional[str] = None):
        """列出所有 adapter"""
        adapters = storage.list_adapters(user_id)
        return {"adapters": adapters}

    @app.post("/api/v1/aggregate", response_model=AggregateResponse)
    async def trigger_aggregation(version: Optional[str] = None):
        """
        触发聚合

        将所有待聚合的 adapter 使用 FedAvg 合并
        """
        pending = storage.get_pending_adapters()

        if len(pending) < server_config.min_adapters_for_aggregation:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {server_config.min_adapters_for_aggregation} adapters, "
                f"but only {len(pending)} available",
            )

        # 获取 adapter 路径
        adapter_paths = [Path(a["path"]) for a in pending]

        # 执行聚合
        try:
            output_path = aggregator.aggregate(adapter_paths, version)
            final_version = output_path.name

            # 保存到存储
            storage.save_aggregated(output_path, final_version)

            return AggregateResponse(
                success=True,
                version=final_version,
                num_adapters=len(pending),
                message="Aggregation completed successfully",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/adapter/download/latest")
    async def download_latest():
        """下载最新的聚合 adapter"""
        latest = storage.get_latest_aggregated()
        if not latest:
            raise HTTPException(status_code=404, detail="No aggregated adapter available")

        adapter_path = Path(latest["path"])
        safetensor_file = adapter_path / "adapter_model.safetensors"

        if not safetensor_file.exists():
            raise HTTPException(status_code=404, detail="Adapter file not found")

        return FileResponse(
            safetensor_file,
            media_type="application/octet-stream",
            filename=f"hivemind_{latest['version']}.safetensors",
        )

    @app.get("/api/v1/adapter/download/{version}")
    async def download_version(version: str):
        """下载指定版本的聚合 adapter"""
        adapter_path = storage.output_dir / "aggregated" / version
        safetensor_file = adapter_path / "adapter_model.safetensors"

        if not safetensor_file.exists():
            raise HTTPException(status_code=404, detail=f"Version {version} not found")

        return FileResponse(
            safetensor_file,
            media_type="application/octet-stream",
            filename=f"hivemind_{version}.safetensors",
        )

    return app


# 创建应用实例
app = create_app()
