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
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast

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

class AuditLoss(nn.Module):
    """
    多任务审核损失
    
    组合四个子任务的加权损失:
    - violation_loss: 是否违规 (二元交叉熵, 较高权重 — 核心指标)
    - risk_loss: 风险等级分类 (交叉熵)
    - type_loss: 违规类型分类 (交叉熵)
    """

    def __init__(
        self,
        violation_weight: float = 2.0,
        risk_weight: float = 1.0,
        type_weight: float = 1.5,
    ):
        super().__init__()
        self.w_violation = violation_weight
        self.w_risk = risk_weight
        self.w_type = type_weight
        
        self.ce = nn.CrossEntropyLoss()
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
        violation_loss = self.ce(outputs["violation_logits"], labels["violation"])

        # 风险等级多分类 loss
        risk_loss = self.ce(outputs["risk_logits"], labels["risk_level"])

        # 违规类型多分类 loss
        type_loss = self.ce(outputs["type_logits"], labels["violation_type"])

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
    total_risk_correct = 0
    total_type_correct = 0
    total_samples = 0
    
    all_losses = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask)

        # 统计准确率
        pred_violation = outputs["violation_logits"].argmax(dim=-1).cpu()
        true_violation = batch["labels"]["violation"]
        total_violation_correct += (pred_violation == true_violation).sum().item()

        pred_risk = outputs["risk_logits"].argmax(dim=-1).cpu()
        true_risk = batch["labels"]["risk_level"]
        total_risk_correct += (pred_risk == true_risk).sum().item()

        pred_type = outputs["type_logits"].argmax(dim=-1).cpu()
        true_type = batch["labels"]["violation_type"]
        total_type_correct += (pred_type == true_type).sum().item()

        total_samples += input_ids.size(0)

    n = max(total_samples, 1)
    metrics = {
        "val_acc_violation": round(total_violation_correct / n, 4),
        "val_acc_risk": round(total_risk_correct / n, 4),
        "val_acc_type": round(total_type_correct / n, 4),
    }
    return metrics


# ============================================================
# 训练器
# ============================================================


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
        self.scaler = GradScaler(enabled=self.use_amp)

        # 损失函数
        self.criterion = AuditLoss(
            violation_weight=2.0,
            risk_weight=1.0,
            type_weight=1.5,
        )

        # 优化器 & 调度器
        no_decay = ["bias", "LayerNorm.weight"]
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(param_groups, lr=self.lr, eps=1e-8)

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

                # 前向传播
                if self.use_bf16:
                    with autocast(dtype=torch.bfloat16):
                        outputs = self.model(input_ids, attention_mask)
                        losses = self.criterion(outputs, label_batch)
                        loss = losses["total"] / self.grad_accum_steps
                elif self.use_amp:
                    with autocast():
                        outputs = self.model(input_ids, attention_mask)
                        losses = self.criterion(outputs, label_batch)
                        loss = losses["total"] / self.grad_accum_steps
                else:
                    outputs = self.model(input_ids, attention_mask)
                    losses = self.criterion(outputs, label_batch)
                    loss = losses["total"] / self.grad_accum_steps

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
                    self.optimizer.zero_grad()
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
                            metrics["val_acc_violation"] * 3 +
                            metrics["val_acc_risk"] +
                            metrics["val_acc_type"]
                        )
                        self._log(
                            f"  📊 Val → ViolAcc:{metrics['val_acc_violation']:.4f} "
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
        self._log(f"\n🎉 训练完成! 总耗时: {elapsed_h:.2f}h | 最佳验证分: {self.best_val_score:.4f}")
        
        if self._wandb:
            wandb.finish()


# ============================================================
# 命令行参数
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="训练内容审核专家模型")

    # 数据
    p.add_argument("--train_data", type=str, required=True, help="训练数据 JSONL 路径")
    p.add_argument("--val_data", type=str, default=None, help="验证数据 JSONL 路径")
    p.add_argument("--vocab_file", type=str, default=None, help="词汇表 JSON 路径")
    p.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")
    p.add_argument("--batch_size", type=int, default=16, help="批次大小")
    p.add_argument("--val_batch_size", type=int, default=32, help="验证批次大小")
    p.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")

    # 模型
    p.add_argument("--dim", type=int, default=256, help="隐藏维度 (默认256，轻量)")
    p.add_argument("--n_layers", type=int, default=6, help="Transformer 层数")
    p.add_argument("--n_heads", type=int, default=4, help="注意力头数")
    p.add_argument("--ffn_multiplier", type=int, default=4, help="FFN 倍数")

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
    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = None
    if args.val_data:
        print("加载验证数据...")
        val_dataset = AuditDataset(args.val_data, tokenizer, args.max_seq_len)
        val_loader = create_dataloader(val_dataset, args.val_batch_size, shuffle=False, num_workers=args.num_workers)

    # 构建模型
    print("\n初始化模型...")
    model = ContentAuditExpert(
        vocab_size=tokenizer.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_multiplier=args.ffn_multiplier,
        max_seq_len=args.max_seq_len,
    )
    print(f"参数量: {model.count_parameters()['total']:,}")

    if args.resume_from:
        Trainer.load_model(model, args.resume_from)

    # 训练
    trainer = Trainer(model, train_loader, val_loader, args)
    trainer.train()


if __name__ == "__main__":
    main()
