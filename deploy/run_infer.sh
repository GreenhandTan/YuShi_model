#!/usr/bin/env bash

# 启动内容审核推理，并优先使用虚拟环境中的 Python。
# 用法：bash run_infer.sh "待审核文本" [--threshold 0.52] [--max_length 256] [--batch_size 4] [--onnx_gpu]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

TEXT="${1:-}"
THRESHOLD="0.52"
MAX_LENGTH="256"
BATCH_SIZE="4"
ONNX_GPU="0"
OUTPUT=""

if [[ -z "${TEXT}" ]]; then
  echo "错误: 必须提供待审核文本"
  echo "用法: bash run_infer.sh \"text\" [--threshold 0.52] [--max_length 256] [--batch_size 4] [--onnx_gpu]"
  exit 1
fi

shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --max_length)
      MAX_LENGTH="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --onnx_gpu)
      ONNX_GPU="1"
      shift 1
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

cd "${SCRIPT_DIR}"

CMD=(
  "${PYTHON_BIN}" infer_onnx.py
  --model ./checkpoints/model.onnx
  --vocab ./checkpoints/vocab.json
  --prompt "${TEXT}"
  --max_length "${MAX_LENGTH}"
  --batch_size "${BATCH_SIZE}"
  --violation_conf_threshold "${THRESHOLD}"
)

if [[ "${ONNX_GPU}" == "1" ]]; then
  CMD+=(--use_gpu)
fi

if [[ -n "${OUTPUT}" ]]; then
  CMD+=(--output "${OUTPUT}")
fi

echo "[run_infer] 使用 Python: ${PYTHON_BIN}"
"${CMD[@]}"
