#!/usr/bin/env bash

# 启动内容审核 API，并在启动前自动清理端口占用。
# 用法：bash run_api.sh [--host 0.0.0.0] [--port 8000] [--workers 2] [--onnx_gpu]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="0.0.0.0"
PORT="8000"
WORKERS="2"
ONNX_USE_GPU="0"
VIOLATION_CONF_THRESHOLD="${VIOLATION_CONF_THRESHOLD:-0.30}"
PYTHON_BIN=""

for candidate in \
  "${SCRIPT_DIR}/.venv/bin/python" \
  "${SCRIPT_DIR}/venv/bin/python" \
  "${PWD}/.venv/bin/python" \
  "${PWD}/venv/bin/python" \
  "${SCRIPT_DIR}/.venv/Scripts/python.exe" \
  "${SCRIPT_DIR}/venv/Scripts/python.exe" \
  "${PWD}/.venv/Scripts/python.exe" \
  "${PWD}/venv/Scripts/python.exe"
do
  if [[ -x "${candidate}" ]]; then
    PYTHON_BIN="${candidate}"
    break
  fi
done

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

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
    --onnx_gpu)
      ONNX_USE_GPU="1"
      shift 1
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: bash run_api.sh [--host 0.0.0.0] [--port 8000] [--workers 2] [--onnx_gpu]"
      exit 1
      ;;
  esac
done

echo "[run_api] 检查端口占用: ${PORT}"

# 优先用 lsof 找占用端口的 LISTEN 进程
if command -v lsof >/dev/null 2>&1; then
  PIDS="$(lsof -t -iTCP:${PORT} -sTCP:LISTEN || true)"
  if [[ -n "${PIDS}" ]]; then
    echo "[run_api] 发现占用进程: ${PIDS}，准备终止"
    kill -TERM ${PIDS} || true
    sleep 1

    # 如果还在占用，强制结束
    PIDS_AFTER="$(lsof -t -iTCP:${PORT} -sTCP:LISTEN || true)"
    if [[ -n "${PIDS_AFTER}" ]]; then
      echo "[run_api] 端口仍被占用，执行强制终止: ${PIDS_AFTER}"
      kill -KILL ${PIDS_AFTER} || true
    fi
  fi
elif command -v fuser >/dev/null 2>&1; then
  # 备选：fuser 直接杀端口
  fuser -k "${PORT}/tcp" || true
else
  echo "[run_api] 未找到 lsof/fuser，无法自动清理端口，请手动释放 ${PORT}"
fi

echo "[run_api] 启动服务: http://${HOST}:${PORT}"
echo "[run_api] 使用 Python: ${PYTHON_BIN}"
echo "[run_api] ONNX GPU: ${ONNX_USE_GPU}"
echo "[run_api] 违规阈值: ${VIOLATION_CONF_THRESHOLD}"

# 需要在当前目录存在 api_server.py
cd "${SCRIPT_DIR}"
ONNX_USE_GPU="${ONNX_USE_GPU}" \
VIOLATION_CONF_THRESHOLD="${VIOLATION_CONF_THRESHOLD}" \
"${PYTHON_BIN}" -m uvicorn api_server:app --host "${HOST}" --port "${PORT}" --workers "${WORKERS}"
