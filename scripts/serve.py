#!/usr/bin/env python
"""
HiveMind 服务端启动脚本

使用方法:
    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8000
"""

import argparse

import uvicorn

from hivemind.config import server_config, ensure_dirs


def main():
    parser = argparse.ArgumentParser(description="HiveMind Server")

    parser.add_argument(
        "--host",
        type=str,
        default=server_config.host,
        help="Host to bind",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=server_config.port,
        help="Port to bind",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)",
    )

    args = parser.parse_args()

    # 确保目录存在
    ensure_dirs()

    print("\n" + "=" * 50)
    print("HiveMind Server")
    print("=" * 50)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print("=" * 50 + "\n")

    uvicorn.run(
        "hivemind.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
