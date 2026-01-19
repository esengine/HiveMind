#!/usr/bin/env python
"""
HiveMind Adapter 上传脚本

使用方法:
    python scripts/upload.py --adapter ./adapters/my_adapter
    python scripts/upload.py --adapter ./adapters/my_adapter --server http://localhost:8000
"""

import argparse
from pathlib import Path

from hivemind.client.uploader import AdapterUploader
from hivemind.config import client_config


def main():
    parser = argparse.ArgumentParser(description="Upload LoRA Adapter to HiveMind Server")

    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=client_config.server_url,
        help="HiveMind server URL",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=client_config.user_id,
        help="User ID",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of training samples (for metadata)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of training epochs (for metadata)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show server status and exit",
    )
    parser.add_argument(
        "--download",
        type=str,
        default=None,
        help="Download latest aggregated adapter to this path",
    )

    args = parser.parse_args()

    uploader = AdapterUploader(
        server_url=args.server,
        user_id=args.user_id,
    )

    # 显示状态
    if args.status:
        try:
            status = uploader.get_server_status()
            print("\n" + "=" * 50)
            print("HiveMind Server Status")
            print("=" * 50)
            print(f"Total Adapters: {status['total_adapters']}")
            print(f"Total Users: {status['total_users']}")
            print(f"Latest Version: {status['latest_aggregated_version'] or 'None'}")
            print(f"Pending for Aggregation: {status['pending_for_aggregation']}")
            print("=" * 50)
        except Exception as e:
            print(f"Error: {e}")
        return

    # 下载
    if args.download:
        try:
            downloaded = uploader.download_latest(Path(args.download))
            print(f"\nDownloaded to: {downloaded}")
        except Exception as e:
            print(f"Error: {e}")
        return

    # 上传
    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        print(f"Error: Adapter not found: {adapter_path}")
        return

    print("\n" + "=" * 50)
    print("Uploading Adapter")
    print("=" * 50)
    print(f"Adapter: {adapter_path}")
    print(f"Server: {args.server}")
    print(f"User ID: {args.user_id}")
    print("=" * 50 + "\n")

    try:
        result = uploader.upload(
            adapter_path=adapter_path,
            training_samples=args.samples,
            epochs=args.epochs,
        )
        print("\n" + "=" * 50)
        print("Upload Complete!")
        print(f"Adapter ID: {result['adapter_id']}")
        print("=" * 50)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
