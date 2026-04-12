from pathlib import Path
import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from infer import AuditInferencer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = os.getenv("CHECKPOINT_PATH", str(BASE_DIR / "checkpoints_final_9to1" / "best.pt"))
DEFAULT_VOCAB = os.getenv("VOCAB_PATH", str(BASE_DIR / "checkpoints_final_9to1" / "vocab.json"))
DEFAULT_THRESHOLD = float(os.getenv("VIOLATION_CONF_THRESHOLD", "0.52"))
DEFAULT_DEVICE = os.getenv("INFER_DEVICE", "auto")
DEFAULT_MAX_LENGTH = int(os.getenv("INFER_MAX_LENGTH", "256"))
DEFAULT_BATCH_SIZE = int(os.getenv("INFER_BATCH_SIZE", "16"))

app = FastAPI(title="YuShi Content Audit Service", version="1.0.0")
_engine: Optional[AuditInferencer] = None


class AuditRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Single input text")


class AuditBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="Batch input texts")


@app.on_event("startup")
def startup_event() -> None:
    global _engine
    if _engine is None:
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


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "checkpoint": DEFAULT_CHECKPOINT,
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
