#!/bin/bash
# YuShi 多层级内容审核 API 启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
PORT=${1:-8000}
CHECKPOINT="${CHECKPOINT:-./checkpoints/best.pt}"
RULES_DIR="${RULES_DIR:-./rules}"
THRESHOLD="${THRESHOLD:-0.30}"

echo "=========================================="
echo "YuShi Content Audit"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Rules dir:  $RULES_DIR"
echo "Threshold:  $THRESHOLD"
echo "Port:       $PORT"
echo "=========================================="

python api_server.py \
    --port "$PORT" \
    --checkpoint "$CHECKPOINT" \
    --rules_dir "$RULES_DIR" \
    --threshold "$THRESHOLD"
