"""
YuShi Content Audit API Server
多层级内容审核服务 — 规则引擎 + Encoder 模型

启动方式:
  python api_server.py --port 8000
  python api_server.py --port 8000 --checkpoint ./checkpoints/best.pt
  python api_server.py --port 8000 --rules_dir ./rules
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
import argparse
import uvicorn

from infer import AuditInferencer

BASE_DIR = Path(__file__).resolve().parent

# 默认路径
_DEFAULT_CHECKPOINT = os.getenv(
    "CHECKPOINT_PATH",
    str(BASE_DIR / "checkpoints" / "best.pt"),
)
_DEFAULT_RULES_DIR = os.getenv(
    "RULES_DIR",
    str(BASE_DIR / "rules") if (BASE_DIR / "rules").exists() else None,
)
_DEFAULT_THRESHOLD = float(os.getenv("VIOLATION_CONF_THRESHOLD", "0.30"))
_DEFAULT_MAX_LENGTH = int(os.getenv("INFER_MAX_LENGTH", "256"))
_DEFAULT_BATCH_SIZE = int(os.getenv("INFER_BATCH_SIZE", "8"))

_engine: Optional[AuditInferencer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = AuditInferencer(
        checkpoint_path=_DEFAULT_CHECKPOINT,
        device="auto",
        max_length=_DEFAULT_MAX_LENGTH,
        batch_size=_DEFAULT_BATCH_SIZE,
        enforce_safe_consistency=True,
        violation_conf_threshold=_DEFAULT_THRESHOLD,
        rules_dir=_DEFAULT_RULES_DIR,
    )
    yield
    _engine = None


app = FastAPI(
    title="YuShi Content Audit Service",
    version="2.0.0",
    description="多层级内容审核服务 — 规则引擎 + Encoder 模型",
    lifespan=lifespan,
)


class AuditRequest(BaseModel):
    text: str = Field(..., min_length=1, description="单条待审核文本")


class AuditBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="批量待审核文本")


@app.get("/health")
def health() -> dict:
    rule_stats = _engine.rule_engine.stats() if _engine and _engine.rule_engine else {}
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "version": "2.0.0",
        "architecture": "two-tier (rule_engine + encoder_model)",
        "checkpoint": _DEFAULT_CHECKPOINT,
        "rules_dir": _DEFAULT_RULES_DIR,
        "threshold": _DEFAULT_THRESHOLD,
        "rule_stats": rule_stats,
    }


@app.post("/audit")
def audit(payload: AuditRequest) -> dict:
    if _engine is None:
        raise RuntimeError("Model is not initialized")
    return _engine.audit(payload.text)


@app.post("/audit/batch")
def audit_batch(payload: AuditBatchRequest) -> dict:
    if _engine is None:
        raise RuntimeError("Model is not initialized")
    return _engine.audit_batch(payload.texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YuShi API Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--checkpoint", type=str, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--rules_dir", type=str, default=_DEFAULT_RULES_DIR)
    parser.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD)
    args = parser.parse_args()

    _DEFAULT_CHECKPOINT = args.checkpoint
    _DEFAULT_RULES_DIR = args.rules_dir
    _DEFAULT_THRESHOLD = args.threshold

    uvicorn.run(app, host=args.host, port=args.port)
