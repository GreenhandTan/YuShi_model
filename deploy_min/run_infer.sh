#!/bin/bash

# 推理脚本启动器
# 用法: ./run_infer.sh "待审核文本" [--threshold 0.65] [--max_length 256] [--batch_size 64] [--threads 4]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${SCRIPT_DIR}/checkpoints/best.pt"
VOCAB="${SCRIPT_DIR}/checkpoints/vocab.json"
TEXT="${1:-}"
THRESHOLD="0.65"
MAX_LENGTH="256"
BATCH_SIZE="64"
THREADS="4"

if [ -z "$TEXT" ]; then
    echo "错误: 必须提供待审核文本"
    echo "用法: $0 \"text\" [--threshold 0.65] [--max_length 256] [--batch_size 64] [--threads 4]"
    exit 1
fi

shift || true

# 解析可选参数
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
        --threads)
            THREADS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

python "${SCRIPT_DIR}/infer.py" \
    --checkpoint "${CHECKPOINT}" \
    --vocab "${VOCAB}" \
    --prompt "${TEXT}" \
    --max_length "${MAX_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --violation_conf_threshold "${THRESHOLD}" \
    --num_threads "${THREADS}"
