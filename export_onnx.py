"""Export YuShi PyTorch checkpoint to ONNX for deployment inference."""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model import ContentAuditExpert


class OnnxWrapper(torch.nn.Module):
    def __init__(self, model: ContentAuditExpert):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(input_ids, attention_mask)
        return out["violation_logits"], out["risk_logits"], out["type_logits"], out["confidence"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export PT checkpoint to ONNX")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, default="./checkpoints/model.onnx")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone_name = ckpt.get("backbone_name", "hfl/chinese-roberta-wwm-ext")
    model_args = ckpt.get("args", {})

    # 加载 tokenizer
    tokenizer_dir = Path(args.checkpoint).parent / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    else:
        tokenizer = AutoTokenizer.from_pretrained(backbone_name)

    model = ContentAuditExpert(
        backbone_name=backbone_name,
        dropout=model_args.get("dropout", 0.1),
        pool_last_weight=model_args.get("pool_last_weight", 0.6),
        freeze_backbone_layers=0,
    )

    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = OnnxWrapper(model).eval()

    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (2, args.max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((2, args.max_length), dtype=torch.long)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        opset_version=args.opset,
        dynamo=False,
        input_names=["input_ids", "attention_mask"],
        output_names=["violation_logits", "risk_logits", "type_logits", "confidence"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "violation_logits": {0: "batch"},
            "risk_logits": {0: "batch"},
            "type_logits": {0: "batch"},
            "confidence": {0: "batch"},
        },
    )

    print(f"[OK] ONNX 导出完成: {output_path}")


if __name__ == "__main__":
    main()
