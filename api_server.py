from pathlib import Path
import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from infer import AuditInferencer
from infer_onnx import AuditONNXInferencer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BACKEND = os.getenv("INFER_BACKEND", "onnx").strip().lower()

_LOCAL_CKPT = BASE_DIR / "checkpoints_final_9to1" / "best.pt"
_LOCAL_VOCAB = BASE_DIR / "checkpoints_final_9to1" / "vocab.json"
_DEPLOY_ONNX = BASE_DIR / "checkpoints" / "model.onnx"
_DEPLOY_VOCAB = BASE_DIR / "checkpoints" / "vocab.json"

DEFAULT_CHECKPOINT = os.getenv(
    "CHECKPOINT_PATH",
    str(_LOCAL_CKPT if _LOCAL_CKPT.exists() else BASE_DIR / "checkpoints" / "best.pt"),
)
DEFAULT_ONNX_MODEL = os.getenv(
    "ONNX_MODEL_PATH",
    str(_DEPLOY_ONNX if _DEPLOY_ONNX.exists() else BASE_DIR / "checkpoints_final_9to1" / "model.onnx"),
)
DEFAULT_VOCAB = os.getenv(
    "VOCAB_PATH",
    str(_DEPLOY_VOCAB if _DEPLOY_VOCAB.exists() else _LOCAL_VOCAB),
)
DEFAULT_THRESHOLD = float(os.getenv("VIOLATION_CONF_THRESHOLD", "0.52"))
DEFAULT_DEVICE = os.getenv("INFER_DEVICE", "auto")
DEFAULT_ONNX_USE_GPU = os.getenv("ONNX_USE_GPU", "0") == "1"
DEFAULT_MAX_LENGTH = int(os.getenv("INFER_MAX_LENGTH", "256"))
DEFAULT_BATCH_SIZE = int(os.getenv("INFER_BATCH_SIZE", "16"))

app = FastAPI(title="YuShi Content Audit Service", version="1.1.0")
_engine: Optional[object] = None
_runtime_backend: str = "unknown"


class AuditRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Single input text")


class AuditBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="Batch input texts")


@app.on_event("startup")
def startup_event() -> None:
    global _engine
    global _runtime_backend
    if _engine is not None:
        return

    if DEFAULT_BACKEND == "onnx":
        _engine = AuditONNXInferencer(
            model_path=DEFAULT_ONNX_MODEL,
            vocab_path=DEFAULT_VOCAB,
            use_gpu=DEFAULT_ONNX_USE_GPU,
            max_length=DEFAULT_MAX_LENGTH,
            batch_size=DEFAULT_BATCH_SIZE,
            enforce_safe_consistency=True,
            violation_conf_threshold=DEFAULT_THRESHOLD,
        )
        _runtime_backend = "onnx"
        return

    _engine = AuditInferencer(
        checkpoint_path=DEFAULT_CHECKPOINT,
        vocab_path=DEFAULT_VOCAB,
        device=DEFAULT_DEVICE,
        use_bf16=False,
        max_length=DEFAULT_MAX_LENGTH,
        batch_size=DEFAULT_BATCH_SIZE,
        enforce_safe_consistency=True,
        violation_conf_threshold=DEFAULT_THRESHOLD,
    )
    _runtime_backend = "pytorch"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "backend": _runtime_backend,
        "checkpoint": DEFAULT_CHECKPOINT,
        "onnx_model": DEFAULT_ONNX_MODEL,
        "vocab": DEFAULT_VOCAB,
        "threshold": DEFAULT_THRESHOLD,
        "device": DEFAULT_DEVICE,
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
