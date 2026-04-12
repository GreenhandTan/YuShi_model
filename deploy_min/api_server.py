from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from infer import AuditInferencer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "best.pt"
DEFAULT_VOCAB = BASE_DIR / "checkpoints" / "vocab.json"

app = FastAPI(title="Content Audit Service", version="1.0.0")
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
            checkpoint_path=str(DEFAULT_CHECKPOINT),
            vocab_path=str(DEFAULT_VOCAB),
            device="auto",
            use_bf16=False,
            max_length=256,
            batch_size=16,
            enforce_safe_consistency=True,
            violation_conf_threshold=0.65,
        )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "vocab": str(DEFAULT_VOCAB),
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
