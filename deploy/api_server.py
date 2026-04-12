from pathlib import Path
import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from infer_onnx import AuditONNXInferencer

BASE_DIR = Path(__file__).resolve().parent
_DEPLOY_ONNX = BASE_DIR / "checkpoints" / "model.onnx"
_DEPLOY_VOCAB = BASE_DIR / "checkpoints" / "vocab.json"

DEFAULT_ONNX_MODEL = os.getenv(
    "ONNX_MODEL_PATH",
    str(_DEPLOY_ONNX),
)
DEFAULT_VOCAB = os.getenv(
    "VOCAB_PATH",
    str(_DEPLOY_VOCAB),
)
DEFAULT_THRESHOLD = float(os.getenv("VIOLATION_CONF_THRESHOLD", "0.30"))
DEFAULT_ONNX_USE_GPU = os.getenv("ONNX_USE_GPU", "0") == "1"
DEFAULT_MAX_LENGTH = int(os.getenv("INFER_MAX_LENGTH", "256"))
DEFAULT_BATCH_SIZE = int(os.getenv("INFER_BATCH_SIZE", "16"))

app = FastAPI(title="YuShi Content Audit Service", version="1.1.0")
_engine: Optional[AuditONNXInferencer] = None


class AuditRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Single input text")


class AuditBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="Batch input texts")


@app.on_event("startup")
def startup_event() -> None:
    global _engine
    if _engine is not None:
        return

    _engine = AuditONNXInferencer(
        model_path=DEFAULT_ONNX_MODEL,
        vocab_path=DEFAULT_VOCAB,
        use_gpu=DEFAULT_ONNX_USE_GPU,
        max_length=DEFAULT_MAX_LENGTH,
        batch_size=DEFAULT_BATCH_SIZE,
        enforce_safe_consistency=True,
        violation_conf_threshold=DEFAULT_THRESHOLD,
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "backend": "onnx",
        "onnx_model": DEFAULT_ONNX_MODEL,
        "vocab": DEFAULT_VOCAB,
        "threshold": DEFAULT_THRESHOLD,
        "device": "cuda" if DEFAULT_ONNX_USE_GPU else "cpu",
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
