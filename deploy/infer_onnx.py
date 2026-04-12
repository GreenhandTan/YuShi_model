"""ONNX inference for YuShi content audit model."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort

from tokenizer_min import SimpleTokenizer

VIOLATION_TYPES = [
    "safe",
    "politics",
    "pornography",
    "violence",
    "abuse",
    "spam",
    "fraud",
    "other",
]
RISK_LEVELS = ["safe", "low", "medium", "high", "critical"]


class AuditONNXInferencer:
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        use_gpu: bool = False,
        max_length: int = 256,
        batch_size: int = 16,
        enforce_safe_consistency: bool = True,
        violation_conf_threshold: float = 0.30,
    ):
        self.max_length = max_length
        self.batch_size = batch_size
        self.enforce_safe_consistency = enforce_safe_consistency
        self.violation_conf_threshold = max(0.0, min(1.0, violation_conf_threshold))

        self.tokenizer = SimpleTokenizer.load(vocab_path)

        providers = ["CPUExecutionProvider"]
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]

    def _encode_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        pad_id = self.tokenizer.pad_token_id
        encoded = []
        masks = []
        for text in texts:
            token_ids = self.tokenizer.encode(text)[: self.max_length]
            if len(token_ids) == 0:
                token_ids = [pad_id]
            attn = [1] * len(token_ids)
            if len(token_ids) < self.max_length:
                pad_len = self.max_length - len(token_ids)
                token_ids = token_ids + [pad_id] * pad_len
                attn = attn + [0] * pad_len
            encoded.append(token_ids)
            masks.append(attn)

        input_ids = np.asarray(encoded, dtype=np.int64)
        attention_mask = np.asarray(masks, dtype=np.int64)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @staticmethod
    def _build_reason(is_violation: bool, risk_level: str, vtype: str, confidence: float) -> str:
        level_cn_map = {
            "safe": "安全",
            "low": "低风险",
            "medium": "中等风险",
            "high": "高风险",
            "critical": "严重风险",
        }
        type_cn_map = {
            "safe": "未检测到明显违规",
            "politics": "涉政敏感内容",
            "pornography": "色情或低俗内容",
            "violence": "暴力或血腥内容",
            "abuse": "辱骂或人身攻击内容",
            "fraud": "欺诈或诈骗相关内容",
            "spam": "垃圾信息或广告",
            "other": "其他违规内容",
        }

        if not is_violation:
            return f"内容合规，{level_cn_map[risk_level]}，置信度 {confidence:.1%}"
        return f"检测到{type_cn_map.get(vtype, '违规')}，{level_cn_map.get(risk_level, risk_level)}，置信度 {confidence:.1%}。"

    def _postprocess(self, item: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(item)
        conf = float(out.get("confidence", 0.0))
        pred_violation = bool(out.get("is_violation", False))

        if pred_violation and conf < self.violation_conf_threshold:
            out["is_violation"] = False
            out["reason"] = (
                f"原始违规置信度 {conf:.1%} 低于阈值 {self.violation_conf_threshold:.1%}，"
                "按阈值策略回退为合规。"
            )

        if self.enforce_safe_consistency and not bool(out.get("is_violation", False)):
            out["risk_level"] = "safe"
            out["violation_type"] = "safe"

        return out

    def audit_batch(self, texts: List[str]) -> Dict[str, Any]:
        start = time.time()
        results: List[Dict[str, Any]] = []

        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            feed = self._encode_batch(chunk)
            outputs = self.session.run(None, feed)

            # Export order: violation_logits, risk_logits, type_logits, confidence
            violation_logits, risk_logits, type_logits, confidence = outputs

            for j, text in enumerate(chunk):
                is_violation = int(np.argmax(violation_logits[j])) == 1
                risk_level = RISK_LEVELS[int(np.argmax(risk_logits[j]))]
                violation_type = VIOLATION_TYPES[int(np.argmax(type_logits[j]))]
                conf = float(confidence[j])

                result = {
                    "is_violation": bool(is_violation),
                    "risk_level": risk_level,
                    "violation_type": violation_type if is_violation else "safe",
                    "confidence": round(conf, 4),
                    "reason": self._build_reason(
                        is_violation=bool(is_violation),
                        risk_level=risk_level,
                        vtype=violation_type if is_violation else "safe",
                        confidence=conf,
                    ),
                }
                results.append(self._postprocess(result))

        elapsed = time.time() - start
        violation_count = sum(1 for r in results if r["is_violation"])

        return {
            "results": results,
            "summary": {
                "total": len(results),
                "violation_count": violation_count,
                "safe_count": len(results) - violation_count,
                "latency_seconds": round(elapsed, 4),
            },
        }

    def audit(self, text: str) -> Dict[str, Any]:
        return self.audit_batch([text])["results"][0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YuShi ONNX inference")
    p.add_argument("--model", type=str, required=True, help="ONNX model path")
    p.add_argument("--vocab", type=str, required=True, help="Tokenizer vocab JSON path")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt", type=str, default=None)
    mode.add_argument("--prompts", nargs="+", default=None)
    mode.add_argument("--input_file", type=str, default=None)

    p.add_argument("--output", type=str, default=None)
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--violation_conf_threshold", type=float, default=0.30)
    p.add_argument("--disable_safe_consistency", action="store_true")
    return p.parse_args()


def _load_texts_from_jsonl(input_file: str) -> List[str]:
    texts: List[str] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") or obj.get("prompt") or ""
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
    return texts


def main() -> None:
    args = parse_args()

    engine = AuditONNXInferencer(
        model_path=args.model,
        vocab_path=args.vocab,
        use_gpu=args.use_gpu,
        max_length=args.max_length,
        batch_size=args.batch_size,
        enforce_safe_consistency=not args.disable_safe_consistency,
        violation_conf_threshold=args.violation_conf_threshold,
    )

    if args.prompt is not None:
        result: Any = engine.audit(args.prompt)
    elif args.prompts is not None:
        result = engine.audit_batch(args.prompts)
    else:
        texts = _load_texts_from_jsonl(args.input_file)
        result = engine.audit_batch(texts)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[OUTPUT] 结果已保存: {out_path}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
