"""
推理脚本 — 多层级内容审核

架构:
  第一层: 规则引擎 (CPU, <1ms) — 高置信度关键词/正则匹配
  第二层: Encoder 模型 (CPU, <30ms) — 预训练模型分类
  第三层: 兜底策略 — 置信度阈值 + 一致性后处理

使用方式:
  python infer.py --checkpoint ./checkpoints/best.pt --prompt "测试文本"
  python infer.py --checkpoint ./checkpoints/best.pt --prompts "文本1" "文本2"
  python infer.py --checkpoint ./checkpoints/best.pt --input_file input.jsonl
  python infer.py --checkpoint ./checkpoints/best.pt --interactive
  python infer.py --checkpoint ./checkpoints/best.pt --rules_dir ./rules --prompt "测试"
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer

from model import ContentAuditExpert
from rule_engine import RuleEngine


# ============================================================
# 推理器
# ============================================================


class AuditInferencer:
    """
    多层级内容审核推理器
    
    两级架构:
    1. 规则引擎: <1ms, 处理高置信度场景
    2. Encoder 模型: <30ms, 处理灰色地带
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        max_length: int = 256,
        batch_size: int = 8,
        enforce_safe_consistency: bool = True,
        violation_conf_threshold: float = 0.30,
        multi_thresholds: Optional[Dict[str, float]] = None,
        rules_dir: Optional[str] = None,
        skip_rules: bool = False,
    ):
        self.device = self._resolve_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.enforce_safe_consistency = enforce_safe_consistency
        self.violation_conf_threshold = max(0.0, min(1.0, violation_conf_threshold))
        self.skip_rules = skip_rules

        self.multi_thresholds = multi_thresholds or {
            "safe": 0.20, "low": 0.30, "medium": 0.40,
            "high": 0.50, "critical": 0.70,
        }

        # ---- 加载规则引擎 ----
        if not skip_rules:
            self.rule_engine = RuleEngine(rules_dir=rules_dir)
            stats = self.rule_engine.stats()
            print(f"[规则引擎] 已加载: {stats}")
        else:
            self.rule_engine = None

        # ---- 加载模型 ----
        print(f"加载模型: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        backbone_name = ckpt.get("backbone_name", "hfl/chinese-roberta-wwm-ext")
        model_args = ckpt.get("args", {})

        # 加载 tokenizer
        tokenizer_dir = Path(checkpoint_path).parent / "tokenizer"
        if tokenizer_dir.exists():
            print(f"加载 tokenizer: {tokenizer_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        else:
            print(f"加载 tokenizer: {backbone_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)

        # 加载模型
        self.model = ContentAuditExpert(
            backbone_name=backbone_name,
            dropout=model_args.get("dropout", 0.1),
            pool_last_weight=model_args.get("pool_last_weight", 0.6),
            freeze_backbone_layers=0,  # 推理时不冻结任何层
        )

        sd = ckpt["model_state_dict"]
        new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(new_sd)
        self.model.to(self.device)
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
        """阈值与一致性后处理"""
        out = dict(result)

        conf = float(out.get("confidence", 0.0))
        pred_violation = bool(out.get("is_violation", False))
        risk_level = str(out.get("risk_level", "safe")).lower()

        threshold = self.multi_thresholds.get(risk_level, self.violation_conf_threshold)

        if pred_violation and conf < threshold:
            out["is_violation"] = False
            out["reason"] = (
                f"风险等级 {risk_level} 的违规置信度 {conf:.1%} 低于阈值 {threshold:.1%}，"
                "按多阈值策略回退为合规。"
            )

        if self.enforce_safe_consistency and not bool(out.get("is_violation", False)):
            out["risk_level"] = "safe"
            out["violation_type"] = "safe"

        return out

    def audit(self, text: str) -> Dict[str, Any]:
        """
        审核单条文本 — 两级架构

        Returns:
            审核结果 dict (与 V1 格式一致，额外包含 layer 字段)
        """
        # ---- 第一层: 规则引擎 ----
        if self.rule_engine and not self.skip_rules:
            rule_result = self.rule_engine.check(text)
            if rule_result is not None:
                # 规则引擎已判定
                result = self._postprocess_result(rule_result)
                result["layer"] = "rule"
                return result

        # ---- 第二层: 模型推理 ----
        with torch.inference_mode():
            model_result = self.model.audit(
                text, tokenizer=self.tokenizer, max_length=self.max_length,
            )

        result = self._postprocess_result(model_result)
        result["layer"] = "model"
        return result

    def audit_json(self, text: str) -> str:
        result = self.audit(text)
        return json.dumps(result, ensure_ascii=False, indent=2)

    def audit_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        批量审核 — 两级架构

        规则引擎命中的文本直接出结果，未命中的批量送模型推理
        """
        start = time.time()

        # 第一层: 规则引擎筛选
        results = [None] * len(texts)
        model_indices = []

        if self.rule_engine and not self.skip_rules:
            for i, text in enumerate(texts):
                rule_result = self.rule_engine.check(text)
                if rule_result is not None:
                    results[i] = self._postprocess_result(rule_result)
                    results[i]["layer"] = "rule"
                else:
                    model_indices.append(i)
        else:
            model_indices = list(range(len(texts)))

        # 第二层: 模型推理 (仅对规则引擎未命中的文本)
        if model_indices:
            model_texts = [texts[i] for i in model_indices]

            with torch.inference_mode():
                model_results = self.model.audit_batch(
                    model_texts, tokenizer=self.tokenizer,
                    max_length=self.max_length, batch_size=self.batch_size,
                )

            for idx, model_result in zip(model_indices, model_results):
                result = self._postprocess_result(model_result)
                result["layer"] = "model"
                results[idx] = result

        elapsed = time.time() - start

        violation_count = sum(1 for r in results if r["is_violation"])
        rule_hit_count = sum(1 for r in results if r.get("layer") == "rule")
        model_hit_count = sum(1 for r in results if r.get("layer") == "model")

        return {
            "results": results,
            "summary": {
                "total": len(results),
                "violation_count": violation_count,
                "safe_count": len(results) - violation_count,
                "rule_hit_count": rule_hit_count,
                "model_hit_count": model_hit_count,
                "latency_seconds": round(elapsed, 4),
            },
        }

    def audit_batch_json(self, texts: List[str]) -> str:
        result = self.audit_batch(texts)
        return json.dumps(result, ensure_ascii=False, indent=2)

    def audit_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """从 JSONL 文件读取文本，执行批量审核"""
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

        result = self.audit_batch(texts)

        if output_path is None:
            out_name = Path(input_path).stem + "_audit_result.json"
            output_path = str(Path(input_path).parent / out_name)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        s = result["summary"]
        print(f"\n[OUTPUT] 结果已保存: {output_path}")
        print(f"   总计: {s['total']} 条 | 违规: {s['violation_count']} 条 | "
              f"合规: {s['safe_count']} 条 | "
              f"规则命中: {s['rule_hit_count']} 条 | 模型推理: {s['model_hit_count']} 条 | "
              f"耗时: {s['latency_seconds']:.2f}s")

        return output_path

    def interactive(self):
        """交互式逐条审核"""
        print("\n" + "=" * 55)
        print("[AUDIT] 多层级内容审核交互模式")
        print("   第一层: 规则引擎 (<1ms)")
        print("   第二层: Encoder 模型 (<30ms)")
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

            print("\n--- 审核结果 ---")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("-" * 38)
            print(f"[TIME] 耗时: {elapsed:.3f}s | 层级: {result.get('layer', '?')}\n")


# ============================================================
# 命令行入口
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="多层级内容审核推理 — 规则引擎 + Encoder 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python infer.py --checkpoint ./checkpoints/best.pt --prompt "测试文本"
  python infer.py --checkpoint ./checkpoints/best.pt --prompts "文本1" "文本2"
  python infer.py --checkpoint ./checkpoints/best.pt --input_file input.jsonl
  python infer.py --checkpoint ./checkpoints/best.pt --interactive
  python infer.py --checkpoint ./checkpoints/best.pt --rules_dir ./rules --prompt "测试"
        """
    )

    p.add_argument("--checkpoint", "--ckpt", type=str, required=True,
                   help="模型检查点路径 (.pt)")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, default=None, help="单条待审文本")
    group.add_argument("--prompts", nargs="+", default=None, help="多条待审文本")
    group.add_argument("--input_file", type=str, default=None, help="输入 JSONL 文件路径")
    group.add_argument("--interactive", action="store_true", help="交互模式")

    p.add_argument("--output", "-o", type=str, default=None, help="输出 JSON 文件路径")
    p.add_argument("--device", type=str, default="auto", help="设备 (auto/cpu/cuda)")
    p.add_argument("--max_length", type=int, default=256, help="最大文本长度")
    p.add_argument("--batch_size", type=int, default=8, help="批量推理 batch 大小")
    p.add_argument("--num_threads", type=int, default=2, help="CPU 推理线程数")
    p.add_argument("--violation_conf_threshold", type=float, default=0.30,
                   help="违规判定最小置信度阈值")
    p.add_argument("--multi_thresholds", type=str, default=None,
                   help="多阈值策略 (JSON)")
    p.add_argument("--disable_safe_consistency", action="store_true")
    p.add_argument("--rules_dir", type=str, default=None, help="自定义规则目录")
    p.add_argument("--skip_rules", action="store_true", help="跳过规则引擎，仅用模型")

    return p.parse_args()


def main():
    args = parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        torch.set_num_interop_threads(max(1, min(args.num_threads, 2)))

    multi_thresholds = None
    if args.multi_thresholds:
        try:
            multi_thresholds = json.loads(args.multi_thresholds)
        except json.JSONDecodeError:
            print(f"[ERROR] 无法解析 --multi_thresholds: {args.multi_thresholds}")
            sys.exit(1)

    engine = AuditInferencer(
        checkpoint_path=args.checkpoint,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        enforce_safe_consistency=not args.disable_safe_consistency,
        violation_conf_threshold=args.violation_conf_threshold,
        multi_thresholds=multi_thresholds,
        rules_dir=args.rules_dir,
        skip_rules=args.skip_rules,
    )

    if args.interactive:
        engine.interactive()
    elif args.input_file:
        engine.audit_file(args.input_file, args.output)
    else:
        prompts = args.prompts or [args.prompt]
        if len(prompts) == 1:
            result_json = engine.audit_json(prompts[0])
            print(result_json)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(result_json)
                print(f"\n[SAVE] 已保存: {args.output}")
        else:
            result_json = engine.audit_batch_json(prompts)
            print(result_json)
            output_file = args.output or "audit_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result_json)
            print(f"\n[SAVE] 已保存: {output_file}")


if __name__ == "__main__":
    main()
