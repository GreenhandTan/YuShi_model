"""
推理脚本
使用训练好的内容审核专家模型进行文本审核，输出标准 JSON 格式

支持:
- 单条 / 批量文本审核
- 文件输入 (JSONL)，文件输出 (JSON)
- 交互模式（逐条输入）
- 所有输出均为标准 JSON 格式
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

from model import ContentAuditExpert
from dataset import SimpleTokenizer


# ============================================================
# 推理器
# ============================================================


class AuditInferencer:
    """
    内容审核推理器
    
    核心职责:
    - 加载模型 & 词汇表
    - 执行审核推理
    - 输出标准 JSON 格式的审核结果
    """

    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        device: str = "auto",
        use_bf16: bool = False,
        max_length: int = 256,
        batch_size: int = 4,
        enforce_safe_consistency: bool = True,
        violation_conf_threshold: float = 0.0,
    ):
        """
        Args:
            checkpoint_path: 模型 .pt 检查点路径
            vocab_path: 词汇表 JSON 路径
            device: "auto" / "cpu" / "cuda" / "cuda:N"
            use_bf16: 是否 BF16 推理
        """
        self.device = self._resolve_device(device)
        self.use_bf16 = use_bf16 and (self.device.type == "cuda")
        self.max_length = max_length
        self.batch_size = batch_size
        self.enforce_safe_consistency = enforce_safe_consistency
        self.violation_conf_threshold = max(0.0, min(1.0, violation_conf_threshold))

        # 加载 tokenizer
        print(f"加载词汇表: {vocab_path}")
        self.tokenizer = SimpleTokenizer.load(vocab_path)

        # 加载模型
        print(f"加载模型: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        model_args = ckpt.get("args", {})
        self.model = ContentAuditExpert(
            vocab_size=self.tokenizer.vocab_size,
            dim=model_args.get("dim", 256),
            n_layers=model_args.get("n_layers", 6),
            n_heads=model_args.get("n_heads", 4),
            ffn_multiplier=model_args.get("ffn_multiplier", 4),
            max_seq_len=model_args.get("max_seq_len", 1024),
            pad_token_id=self.tokenizer.pad_token_id,
        )

        sd = ckpt["model_state_dict"]
        new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(new_sd)
        self.model.to(self.device)

        if self.use_bf16:
            self.model = self.model.bfloat16()
        self.model.eval()

        step = ckpt.get("global_step", "?")
        score = ckpt.get("best_val_score", None)
        print(f"[OK] 模型就绪! step={step}" + 
              (f", best_score={score:.4f}" if score else ""))

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """对推理结果做阈值与一致性后处理。"""
        out = dict(result)

        conf = float(out.get("confidence", 0.0))
        pred_violation = bool(out.get("is_violation", False))

        # 当预测为违规但置信度低于阈值时，回退为合规
        if pred_violation and conf < self.violation_conf_threshold:
            out["is_violation"] = False
            out["reason"] = (
                f"原始违规置信度 {conf:.1%} 低于阈值 {self.violation_conf_threshold:.1%}，"
                "按阈值策略回退为合规。"
            )

        # 一致性规则: 合规时强制 risk/type 为 safe
        if self.enforce_safe_consistency and not bool(out.get("is_violation", False)):
            out["risk_level"] = "safe"
            out["violation_type"] = "safe"

        return out

    # ---- 单条审核 ----

    @torch.inference_mode()
    def audit(self, text: str) -> Dict[str, Any]:
        """
        审核单条文本，返回 JSON 兼容的审核结果字典

        Returns:
            {
                "is_violation": bool,
                "risk_level": str,
                "violation_type": str,
                "confidence": float,
                "reason": str,
            }
        """
        result = self.model.audit(
            text,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return self._postprocess_result(result)

    def audit_json(self, text: str) -> str:
        """审核单条文本，返回 JSON 字符串"""
        result = self.audit(text)
        return json.dumps(result, ensure_ascii=False, indent=2)

    # ---- 批量审核 ----

    @torch.inference_mode()
    def audit_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        批量审核多条文本
        
        Returns:
            {
                "results": [result_dict, ...],
                "summary": {
                    "total": int,
                    "violation_count": int,
                    "safe_count": int,
                    "latency_seconds": float,
                }
            }
        """
        start = time.time()
        results = self.model.audit_batch(
            texts,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        results = [self._postprocess_result(r) for r in results]
        elapsed = time.time() - start

        violation_count = sum(1 for r in results if r["is_violation"])
        safe_count = len(results) - violation_count

        output = {
            "results": results,
            "summary": {
                "total": len(results),
                "violation_count": violation_count,
                "safe_count": safe_count,
                "latency_seconds": round(elapsed, 4),
            },
        }
        return output

    def audit_batch_json(self, texts: List[str]) -> str:
        """批量审核，返回 JSON 字符串"""
        result = self.audit_batch(texts)
        return json.dumps(result, ensure_ascii=False, indent=2)

    # ---- 文件输入/输出 ----

    def audit_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        从 JSONL 文件读取待审核文本，执行批量审核，写入 JSON 输出文件
        
        输入格式: JSONL，每行 {"text": "..."} 或 {"prompt": "..."}
        输出格式: JSON (含 results + summary)
        
        Returns:
            输出文件路径
        """
        # 读输入
        texts = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text") or data.get("prompt") or ""
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
                except json.JSONDecodeError:
                    print(f"[WARN] 第 {line_num} 行解析失败，跳过")

        if not texts:
            raise ValueError(f"{input_path} 中未找到有效文本数据")

        print(f"\n从 {input_path} 读取了 {len(texts)} 条文本")
        
        # 审核
        result = self.audit_batch(texts, **kwargs)

        # 确定输出路径
        if output_path is None:
            out_name = Path(input_path).stem + "_audit_result.json"
            output_path = str(Path(input_path).parent / out_name)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        s = result["summary"]
        print(f"\n[OUTPUT] 结果已保存: {output_path}")
        print(f"   总计: {s['total']} 条 | "
              f"违规: {s['violation_count']} 条 | "
              f"合规: {s['safe_count']} 条 | "
              f"耗时: {s['latency_seconds']:.2f}s")

        return output_path

    # ---- 交互模式 ----

    def interactive(self):
        """交互式逐条审核"""
        print("\n" + "=" * 55)
        print("[AUDIT] 内容审核交互模式")
        print("   输入待审核文本后回车查看结果")
        print("   输入 quit / exit / q 退出\n")

        while True:
            try:
                text = input("待审文本 > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not text:
                continue
            if text.lower() in ("quit", "exit", "q"):
                break

            t0 = time.time()
            result = self.audit(text)
            elapsed = time.time() - t0

            # 打印 JSON 结果
            print("\n--- 审核结果 ---")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("-" * 38)
            print(f"[TIME] 耗时: {elapsed:.3f}s\n")


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="内容审核专家模型推理 — 标准JSON输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 单条审核
  python infer.py --checkpoint ./checkpoints/best.pt -- ./vocab.json --prompt "测试文本"

  # 批量审核
  python infer.py --ckpt best.pt --vocab vocab.json --prompts "文本1" "文本2"

  # 文件审核 → JSON 输出
  python infer.py --ckpt best.pt --vocab vocab.json --input_file input.jsonl

  # 交互模式
  python infer.py --ckpt best.pt --vocab vocab.json --interactive
        """
    )

    # 必需参数
    p.add_argument("--checkpoint", "--ckpt", type=str, required=True,
                   help="模型检查点路径 (.pt)")
    p.add_argument("--vocab", type=str, required=True,
                   help="词汇表 JSON 路径")

    # 输入模式 (互斥)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, default=None, help="单条待审文本")
    group.add_argument("--prompts", nargs="+", default=None, help="多条待审文本")
    group.add_argument("--input_file", type=str, default=None, help="输入 JSONL 文件路径")
    group.add_argument("--interactive", action="store_true", help="交互模式")

    # 输出
    p.add_argument("--output", "-o", type=str, default=None, help="输出 JSON 文件路径")

    # 设备
    p.add_argument("--device", type=str, default="auto",
                   help="设备 (auto/cpu/cuda/cuda:0)")
    p.add_argument("--bf16", action="store_true", help="使用 BF16 推理")
    p.add_argument("--max_length", type=int, default=256, help="推理最大文本长度")
    p.add_argument("--batch_size", type=int, default=4, help="批量推理 batch 大小")
    p.add_argument("--num_threads", type=int, default=2, help="CPU 推理线程数")
    p.add_argument("--violation_conf_threshold", type=float, default=0.0,
                   help="违规判定最小置信度阈值 (0~1)。低于阈值的违规将回退为合规")
    p.add_argument("--disable_safe_consistency", action="store_true",
                   help="关闭一致性后处理（默认开启：合规时强制 risk/type=safe）")

    return p.parse_args()


def main():
    args = parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(max(1, min(args.num_threads, 2)))

    # 初始化推理器
    engine = AuditInferencer(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        device=args.device,
        use_bf16=args.bf16,
        max_length=args.max_length,
        batch_size=args.batch_size,
        enforce_safe_consistency=not args.disable_safe_consistency,
        violation_conf_threshold=args.violation_conf_threshold,
    )

    # ---- 分发到各模式 ----
    if args.interactive:
        engine.interactive()

    elif args.input_file:
        output_file = args.output or None
        engine.audit_file(args.input_file, output_file)

    else:
        prompts = args.prompts or [args.prompt]

        if len(prompts) == 1:
            # 单条
            result_json = engine.audit_json(prompts[0])
            print(result_json)
            
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(result_json)
                print(f"\n[SAVE] 已保存: {args.output}")

        else:
            # 批量
            result_json = engine.audit_batch_json(prompts)
            print(result_json)

            output_file = args.output or "audit_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result_json)
            print(f"\n[SAVE] 已保存: {output_file}")


if __name__ == "__main__":
    main()
