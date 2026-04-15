"""
训练脚本
训练内容审核专家模型 (ContentAuditExpert)

支持:
- 多任务联合训练 (违规检测 + 风险等级 + 违规类型 + 置信度)
- 混合精度训练 (AMP / BF16)
- 学习率调度 (Cosine with Warmup)
- Checkpoint 保存/加载
- WandB 日志
"""

import os
import sys
import time
import json
import math
import random
import re
import argparse
import importlib.util
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from model import ContentAuditExpert, VIOLATION_TYPES, RISK_LEVELS
from dataset import (
    AuditDataset, SimpleTokenizer, create_dataloader,
    VIOLATION_TYPE_TO_ID, RISK_LEVEL_TO_ID,
)


# ============================================================
# 日志
# ============================================================

def setup_logger(name: str = "train", log_file: Optional[str] = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


logger = setup_logger()


# ============================================================
# 学习率调度器
# ============================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0.0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# 多任务损失计算
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: 处理类别严重不平衡和困难样本
    能让模型自动去死磕那些“看似安全实际违规”的伪装文本，或“看似违规实际正常”的闲聊
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction='none')
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class AuditLoss(nn.Module):
    """
    多任务审核损失
    
    组合四个子任务的加权损失:
    - violation_loss: 是否违规 (Focal Loss, 较高权重 — 核心指标)
    - risk_loss: 风险等级分类 (加权交叉熵 + 标签平滑)
    - type_loss: 违规类型分类 (加权交叉熵 + 标签平滑)
    """

    def __init__(
        self,
        violation_weight: float = 2.0,
        risk_weight: float = 1.0,
        type_weight: float = 1.5,
        violation_class_weights: Optional[torch.Tensor] = None,
        risk_class_weights: Optional[torch.Tensor] = None,
        type_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.w_violation = violation_weight
        self.w_risk = risk_weight
        self.w_type = type_weight

        # 违规二分类引入 Focal Loss (gamma=2.0)，强迫模型专注于最难分对的擦边球和日常闲聊误杀
        self.violation_ce = FocalLoss(weight=violation_class_weights, gamma=2.0, label_smoothing=0.0)
        # 多分类引入 Label Smoothing (0.05)，防止模型过度自信产生误伤
        self.risk_ce = nn.CrossEntropyLoss(weight=risk_class_weights, label_smoothing=0.05)
        self.type_ce = nn.CrossEntropyLoss(weight=type_class_weights, label_smoothing=0.05)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: 模型前向输出 (包含 violation_logits, risk_logits, type_logits)
            labels: 标签字典 {violation, risk_level, violation_type}
        
        Returns:
            losses 字典，包含各子损失和 total_loss
        """
        # 违规二分类 loss
        # violation_logits 形状 (B, 2)，labels["violation"] 形状 (B,)
        violation_loss = self.violation_ce(outputs["violation_logits"], labels["violation"])

        # 风险等级多分类 loss (使用加权版本)
        risk_loss = self.risk_ce(outputs["risk_logits"], labels["risk_level"])

        # 违规类型多分类 loss (使用加权版本)
        type_loss = self.type_ce(outputs["type_logits"], labels["violation_type"])

        # 加权总损失
        total = (
            self.w_violation * violation_loss
            + self.w_risk * risk_loss
            + self.w_type * type_loss
        )

        return {
            "total": total,
            "violation": violation_loss,
            "risk": risk_loss,
            "type": type_loss,
        }


# ============================================================
# 评估指标
# ============================================================

@torch.no_grad()
def evaluate(
    model: ContentAuditExpert,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """在验证集上评估各项指标"""
    model.eval()
    
    total_violation_correct = 0
    violation_tp = 0
    violation_fp = 0
    violation_tn = 0
    violation_fn = 0
    total_risk_correct = 0
    total_type_correct = 0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask)

        # 统计准确率
        pred_violation = outputs["violation_logits"].argmax(dim=-1).cpu()
        true_violation = batch["labels"]["violation"]
        total_violation_correct += (pred_violation == true_violation).sum().item()
        violation_tp += ((pred_violation == 1) & (true_violation == 1)).sum().item()
        violation_fp += ((pred_violation == 1) & (true_violation == 0)).sum().item()
        violation_tn += ((pred_violation == 0) & (true_violation == 0)).sum().item()
        violation_fn += ((pred_violation == 0) & (true_violation == 1)).sum().item()

        pred_risk = outputs["risk_logits"].argmax(dim=-1).cpu()
        true_risk = batch["labels"]["risk_level"]
        total_risk_correct += (pred_risk == true_risk).sum().item()

        pred_type = outputs["type_logits"].argmax(dim=-1).cpu()
        true_type = batch["labels"]["violation_type"]
        total_type_correct += (pred_type == true_type).sum().item()

        total_samples += input_ids.size(0)

    n = max(total_samples, 1)
    violation_precision = violation_tp / (violation_tp + violation_fp) if (violation_tp + violation_fp) else 0.0
    violation_recall = violation_tp / (violation_tp + violation_fn) if (violation_tp + violation_fn) else 0.0
    violation_f1 = (
        2 * violation_precision * violation_recall / (violation_precision + violation_recall)
        if (violation_precision + violation_recall)
        else 0.0
    )
    metrics = {
        "val_acc_violation": round(total_violation_correct / n, 4),
        "val_violation_precision": round(violation_precision, 4),
        "val_violation_recall": round(violation_recall, 4),
        "val_violation_f1": round(violation_f1, 4),
        "val_acc_risk": round(total_risk_correct / n, 4),
        "val_acc_type": round(total_type_correct / n, 4),
    }
    return metrics


# ============================================================
# 训练器
# ============================================================



import numpy as np

def _compute_violation_class_weights(dataset):
    counts = [0, 0]
    for sample in dataset.samples:
        is_vio = sample.get("is_violation", 0)
        counts[is_vio] += 1
    total = sum(counts)
    if min(counts) == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float)
    # Weights proportional to inverse class frequency
    weights = [total / (2.0 * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float)

def _compute_risk_level_class_weights(dataset):
    counts = [0]*5
    for sample in dataset.samples:
        r_level = sample.get("risk_level_id", 0)
        counts[r_level] += 1
    total = sum(counts)
    non_zero_classes = sum(1 for c in counts if c > 0)
    weights = [total / (non_zero_classes * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float)

def _compute_violation_type_class_weights(dataset):
    counts = [0]*8
    for sample in dataset.samples:
        v_type = sample.get("violation_type_id", 0)
        counts[v_type] += 1
    total = sum(counts)
    non_zero_classes = sum(1 for c in counts if c > 0)
    weights = [total / (non_zero_classes * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float)


class Trainer:
    """内容审核模型训练器"""

    def __init__(
        self,
        model: ContentAuditExpert,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        args: argparse.Namespace,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
        self.model.to(self.device)
        self.non_blocking = self.device.type == "cuda"

        # 超参数
        self.lr = args.learning_rate
        self.weight_decay = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        self.grad_accum_steps = args.grad_accum_steps
        self.epochs = args.epochs
        self.save_every = args.save_every
        self.eval_every = args.eval_every
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = args.use_amp and not args.use_bf16
        self.use_bf16 = args.use_bf16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self._compile_fallback_used = False
        self._original_model = getattr(model, "_orig_mod", None)

        violation_class_weights = None
        risk_class_weights = None
        type_class_weights = None

        if hasattr(train_loader.dataset, "samples"):
            violation_class_weights = _compute_violation_class_weights(train_loader.dataset)
            risk_class_weights = _compute_risk_level_class_weights(train_loader.dataset)
            type_class_weights = _compute_violation_type_class_weights(train_loader.dataset)

            safe_count = len(train_loader.dataset) - sum(int(sample["is_violation"]) for sample in train_loader.dataset.samples)
            violation_count = sum(int(sample["is_violation"]) for sample in train_loader.dataset.samples)
            self._log(
                "  违规样本分布: safe={safe} | violation={violation} | class_weights={weights}".format(
                    safe=safe_count,
                    violation=violation_count,
                    weights=[round(float(v), 4) for v in violation_class_weights.tolist()],
                )
            )

            # 日志输出风险等级和违规类型的分布
            from dataset import ID_TO_RISK_LEVEL, ID_TO_VIOLATION_TYPE
            risk_dist = {ID_TO_RISK_LEVEL[i]: count for i, count in enumerate(
                [sum(1 for s in train_loader.dataset.samples if s["risk_level_id"] == i) for i in range(5)]
            )}
            self._log(f"  风险等级分布: {risk_dist}")

            type_dist = {ID_TO_VIOLATION_TYPE[i]: count for i, count in enumerate(
                [sum(1 for s in train_loader.dataset.samples if s["violation_type_id"] == i) for i in range(8)]
            )}
            self._log(f"  违规类型分布: {type_dist}")

        # 损失函数
        self.criterion = AuditLoss(
            violation_weight=2.0,
            risk_weight=1.0,
            type_weight=1.5,
            violation_class_weights=violation_class_weights,
            risk_class_weights=risk_class_weights,
            type_class_weights=type_class_weights,
        ).to(self.device)

        # 优化器 & 调度器
        no_decay = ["bias", "LayerNorm.weight"]
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        
        import inspect
        use_fused = 'fused' in inspect.signature(AdamW).parameters and self.device.type == 'cuda'
        if use_fused:
            self.optimizer = AdamW(param_groups, lr=self.lr, eps=1e-8, fused=True)
            self._log("  [Opt] 已开启 Fused AdamW 优化，加速大模型参数更新底座")
        else:
            self.optimizer = AdamW(param_groups, lr=self.lr, eps=1e-8)
            
        self.optimizer.zero_grad(set_to_none=True)

        total_steps = len(train_loader) // self.grad_accum_steps * self.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)

        self.global_step = 0
        self.best_val_score = 0.0
        self.start_time = time.time()

        # WandB
        if HAS_WANDB and getattr(args, "use_wandb", False):
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
            )
            self._wandb = True
        else:
            self._wandb = False

    def _log(self, msg):
        logger.info(msg)

    def _save_checkpoint(self, name: str):
        state_dict = self.model.state_dict()
        checkpoint = {
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_score": self.best_val_score,
            "args": vars(self.args),
        }
        path = self.output_dir / name
        torch.save(checkpoint, path)
        self._log(f"[SAVE] 已保存: {path}")

    @classmethod
    def load_model(cls, model: ContentAuditExpert, path: str, device="cpu") -> ContentAuditExpert:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(new_sd)
        logger.info(f"[OK] 加载 checkpoint: {path} (step={ckpt.get('global_step','?')})")
        return model

    def train(self):
        self._log("=" * 60)
        self._log("开始训练内容审核专家模型")
        params = self.model.count_parameters()
        self._log(f"  参数量:   {params['total']:,}")
        self._log(f"  设备:     {self.device}")
        self._log(f"  AMP:      {'BF16' if self.use_bf16 else ('FP16' if self.use_amp else '关闭')}")
        self._log(f"  训练集:   {len(self.train_loader.dataset)} 条")
        if self.val_loader:
            self._log(f"  验证集:   {len(self.val_loader.dataset)} 条")
        self._log("=" * 60)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_total_loss = 0.0
            num_batches = len(self.train_loader)

            for step, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                label_batch = {k: v.to(self.device) for k, v in batch["labels"].items()}

                # 前向传播：若 torch.compile 在当前 GPU 上不稳定，则自动回退到原始模型
                forward_success = False
                last_error = None
                for attempt in range(2):
                    try:
                        if self.use_bf16:
                            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                outputs = self.model(input_ids, attention_mask)
                                losses = self.criterion(outputs, label_batch)
                                loss = losses["total"] / self.grad_accum_steps
                        elif self.use_amp:
                            with torch.amp.autocast("cuda", dtype=torch.float16):
                                outputs = self.model(input_ids, attention_mask)
                                losses = self.criterion(outputs, label_batch)
                                loss = losses["total"] / self.grad_accum_steps
                        else:
                            outputs = self.model(input_ids, attention_mask)
                            losses = self.criterion(outputs, label_batch)
                            loss = losses["total"] / self.grad_accum_steps
                        forward_success = True
                        break
                    except Exception as exc:
                        last_error = exc
                        if self._original_model is not None and not self._compile_fallback_used:
                            self._compile_fallback_used = True
                            self.model = self._original_model.to(self.device)
                            self.model.train()
                            self._log(f"[WARN] torch.compile 在当前设备上不稳定，已自动回退到原始模型: {exc}")
                            continue
                        raise

                if not forward_success:
                    raise last_error

                # 反向传播
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # 梯度更新
                if (step + 1) % self.grad_accum_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    # 日志
                    if self.global_step % 50 == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed = time.time() - self.start_time
                        self._log(
                            f"[Epoch {epoch+1}/{self.epochs}] "
                            f"Step {step+1}/{num_batches} | "
                            f"Loss: {losses['total'].item()*self.grad_accum_steps:.4f} | "
                            f"L_vio: {losses['violation'].item():.4f} | "
                            f"L_risk: {losses['risk'].item():.4f} | "
                            f"L_type: {losses['type'].item():.4f} | "
                            f"LR: {lr:.2e}"
                        )
                        
                        if self._wandb:
                            wandb.log({
                                "train_loss": losses["total"].item() * self.grad_accum_steps,
                                "lr": lr,
                            }, step=self.global_step)

                    # 验证
                    if self.val_loader and self.eval_every > 0 and self.global_step % self.eval_every == 0:
                        metrics = evaluate(self.model, self.val_loader, self.device)
                        combined_score = (
                            metrics["val_violation_recall"] * 2 +
                            metrics["val_violation_f1"] * 2 +
                            metrics["val_acc_risk"] +
                            metrics["val_acc_type"]
                        )
                        self._log(
                            f"  Val -> ViolAcc:{metrics['val_acc_violation']:.4f} "
                            f"ViolP:{metrics['val_violation_precision']:.4f} "
                            f"ViolR:{metrics['val_violation_recall']:.4f} "
                            f"ViolF1:{metrics['val_violation_f1']:.4f} "
                            f"RiskAcc:{metrics['val_acc_risk']:.4f} "
                            f"TypeAcc:{metrics['val_acc_type']:.4f}"
                        )
                        if self._wandb:
                            wandb.log(metrics, step=self.global_step)

                        if combined_score > self.best_val_score:
                            self.best_val_score = combined_score
                            self._save_checkpoint("best.pt")
                            self._log(f"  [BEST] 新的最佳模型! Score={combined_score:.4f}")

                    # 定期保存
                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        self._save_checkpoint(f"step_{self.global_step}.pt")

                epoch_total_loss += losses["total"].item()

            avg_loss = epoch_total_loss / num_batches
            self._log(f"\nEpoch {epoch+1} 完成 — Avg Loss: {avg_loss:.4f}\n")

        # 最终保存
        self._save_checkpoint("final.pt")

        elapsed_h = (time.time() - self.start_time) / 3600
        self._log(f"\n训练完成! 总耗时: {elapsed_h:.2f}h | 最佳验证分: {self.best_val_score:.4f}")
        
        if self._wandb:
            wandb.finish()


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="训练内容审核专家模型")

    # 数据
    p.add_argument("--train_data", type=str, default=None, help="训练数据 JSONL 路径")
    p.add_argument("--val_data", type=str, default=None, help="验证数据 JSONL 路径")
    p.add_argument("--vocab_file", type=str, default=None, help="词汇表 JSON 路径")
    p.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")
    p.add_argument("--batch_size", type=int, default=16, help="批次大小")
    p.add_argument("--val_batch_size", type=int, default=32, help="验证批次大小")
    p.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    p.add_argument("--persistent_workers", action="store_true", help="保持 DataLoader worker 常驻，减少 epoch 间重建开销")
    p.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader 预取批次数；num_workers>0 时生效")

    # 模型
    p.add_argument("--dim", type=int, default=256, help="隐藏维度 (默认256，轻量)")
    p.add_argument("--n_layers", type=int, default=6, help="Transformer 层数")
    p.add_argument("--n_heads", type=int, default=4, help="注意力头数")
    p.add_argument("--ffn_multiplier", type=int, default=4, help="FFN 倍数")
    p.add_argument(
        "--pool_last_weight",
        type=float,
        default=0.6,
        help="融合池化中 last-token 权重，范围[0,1]；其余权重自动分配给 mean-pooling",
    )

    # 训练超参
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=4)

    # 精度 & 设备
    p.add_argument("--use_cuda", action="store_true")
    p.add_argument("--use_amp", action="store_true", help="FP16 混合精度")
    p.add_argument("--use_bf16", action="store_true", help="BF16 推理")
    p.add_argument("--compile_model", action="store_true", help="使用 torch.compile 尝试进一步提速")
    p.add_argument("--fast_gpu", action="store_true", help="一键启用更激进的 GPU 训练默认值")

    # 保存 & 日志
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=300)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="ContentAuditExpert")
    p.add_argument("--run_name", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()

    if args.fast_gpu:
        if args.use_cuda:
            args.use_amp = True if not args.use_bf16 else args.use_amp
        args.batch_size = max(args.batch_size, 64)
        args.val_batch_size = max(args.val_batch_size, 128)
        args.grad_accum_steps = 1
        args.num_workers = max(args.num_workers, 8)
        args.persistent_workers = True
        args.prefetch_factor = max(args.prefetch_factor, 4)
        args.save_every = max(args.save_every, 2000)
        args.eval_every = max(args.eval_every, 1000)

    if args.compile_model and importlib.util.find_spec("triton") is None:
        logger.warning("当前环境未安装 triton，已自动关闭 --compile_model；将继续使用 AMP / fast_gpu 训练。")
        args.compile_model = False

    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    if not args.train_data:
        raise ValueError("请提供 --train_data 选项指定本地训练数据文件")

    # 构建 tokenizer
    if args.vocab_file and Path(args.vocab_file).exists():
        tokenizer = SimpleTokenizer.load(args.vocab_file)
        print(f"加载词汇表: {args.vocab_file}, vocab={tokenizer.vocab_size}")
    else:
        texts = []
        with open(args.train_data, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                t = d.get("text", "")
                if t:
                    texts.append(t)
        tokenizer = SimpleTokenizer.from_texts(texts, min_freq=2)
        save_path = Path(args.output_dir) / "vocab.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path))

    # 构建数据集
    print("\n加载训练数据...")
    train_dataset = AuditDataset(args.train_data, tokenizer, args.max_seq_len)
    train_loader = create_dataloader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.use_cuda,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    val_loader = None
    if args.val_data:
        print("加载验证数据...")
        val_dataset = AuditDataset(args.val_data, tokenizer, args.max_seq_len)
        val_loader = create_dataloader(
            val_dataset,
            args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.use_cuda,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )

    # 构建模型
    print("\n初始化模型...")
    model = ContentAuditExpert(
        vocab_size=tokenizer.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_multiplier=args.ffn_multiplier,
        max_seq_len=args.max_seq_len,
        pool_last_weight=args.pool_last_weight,
    )
    if args.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            logger.warning(f"torch.compile 启用失败，继续使用原始模型: {exc}")
    print(f"参数量: {model.count_parameters()['total']:,}")

    if args.resume_from:
        Trainer.load_model(model, args.resume_from)

    # 训练
    trainer = Trainer(model, train_loader, val_loader, args)
    trainer.train()


if __name__ == "__main__":
    main()
