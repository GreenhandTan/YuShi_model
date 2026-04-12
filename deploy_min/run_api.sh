#!/bin/bash

# HTTP 服务启动器
# 用法: ./run_api.sh [--host 0.0.0.0] [--port 8000] [--workers 4]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="0.0.0.0"
PORT="8000"
WORKERS="4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

cd "${SCRIPT_DIR}"

echo "启动内容审核服务..."
echo "  监听: ${HOST}:${PORT}"
echo "  文档: http://${HOST}:${PORT}/docs"
echo ""

python -m uvicorn api_server:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}"
