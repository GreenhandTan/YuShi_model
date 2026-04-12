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
# Hugging Face 数据集支持
# ============================================================

SUBJECT_TO_LABELS = {
    "不违规": ("safe", "safe"),
    "淫秽色情": ("pornography", "high"),
    "脏话侮辱": ("abuse", "medium"),
    "偏见歧视": ("abuse", "medium"),
    "政治错误": ("politics", "high"),
    "财产隐私": ("fraud", "high"),
    "违法犯罪": ("other", "critical"),
    "身体伤害": ("violence", "high"),
    "心理健康": ("other", "medium"),
    "道德伦理": ("other", "low"),
}

# subject 归一化后同义词映射到标准类目
SUBJECT_ALIASES = {
    "不违規": "不违规",
    "安全": "不违规",
    "色情": "淫秽色情",
    "低俗色情": "淫秽色情",
    "辱骂": "脏话侮辱",
    "人身攻击": "脏话侮辱",
    "歧视": "偏见歧视",
    "偏见": "偏见歧视",
    "涉政": "政治错误",
    "政治": "政治错误",
    "隐私": "财产隐私",
    "个人隐私": "财产隐私",
    "犯罪": "违法犯罪",
    "违法": "违法犯罪",
    "暴力": "身体伤害",
    "自残": "身体伤害",
    "自杀": "身体伤害",
    "心理": "心理健康",
    "伦理": "道德伦理",
}

# 未命中标准类目时的关键词兜底
KEYWORD_RULES = [
    (["色情", "淫秽", "露点", "成人视频"], ("pornography", "high")),
    (["辱骂", "脏话", "侮辱", "歧视", "仇恨"], ("abuse", "medium")),
    (["颠覆", "政变", "分裂", "敏感政治", "六四"], ("politics", "high")),
    (["诈骗", "骗", "隐私", "银行卡", "密码", "钓鱼"], ("fraud", "high")),
    (["爆炸", "杀", "砍", "伤害", "暴力", "恐怖"], ("violence", "critical")),
    (["赌博", "毒品", "违禁", "违法", "犯罪"], ("other", "high")),
    (["伦理", "道德"], ("other", "low")),
    (["焦虑", "抑郁", "自卑", "心理"], ("other", "medium")),
]

CHINESE_HARM_LABEL_TO_TARGET = {
    "不违规": (0, "safe", "safe"),
    "低俗色情": (1, "pornography", "high"),
    "博彩": (1, "fraud", "high"),
    "欺诈": (1, "fraud", "high"),
    "谩骂引战": (1, "abuse", "medium"),
    "黑产广告": (1, "spam", "medium"),
}

LEXICON_FILENAME_RULES = [
    (["政治", "gfw"], ("politics", "high")),
    (["色情", "xx", "成人"], ("pornography", "high")),
    (["博彩", "赌博"], ("fraud", "high")),
    (["诈骗", "欺诈"], ("fraud", "high")),
    (["辱骂", "仇恨", "引战"], ("abuse", "medium")),
    (["非法网址", "网址", "广告"], ("spam", "medium")),
    (["暴力", "枪", "炸"], ("violence", "high")),
]


def _normalize_subject(subject: str) -> str:
    s = str(subject).strip().lower()
    # 清理空白和常见分隔符，提升匹配鲁棒性
    s = re.sub(r"[\s\-_/|]+", "", s)
    return s


def _parse_violation_label(label_value) -> int:
    text = str(label_value).strip().lower()
    if text in {"违规", "1", "true", "yes", "unsafe", "violation"}:
        return 1
    if text in {"不违规", "0", "false", "no", "safe", "non-violation"}:
        return 0
    return 0


def _map_subject_to_labels(subject: str, is_violation: int):
    if not is_violation:
        return "safe", "safe"

    raw_subject = str(subject).strip()
    normalized = _normalize_subject(raw_subject)

    # 1) 精确命中标准类目
    if raw_subject in SUBJECT_TO_LABELS:
        return SUBJECT_TO_LABELS[raw_subject]

    # 2) 同义词映射到标准类目
    canonical_subject = SUBJECT_ALIASES.get(normalized)
    if canonical_subject and canonical_subject in SUBJECT_TO_LABELS:
        return SUBJECT_TO_LABELS[canonical_subject]

    # 3) 关键词兜底
    for keywords, mapped in KEYWORD_RULES:
        if any(k in raw_subject for k in keywords):
            return mapped

    # 未知违规子类兜底到 other
    return "other", "medium"


def _write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_dataset_specs(dataset_specs: str) -> List[Dict[str, str]]:
    """
    解析多数据集配置字符串。

    格式:
      dataset|split|text_field|label_field|subject_field
    多个数据集使用逗号分隔。
    """
    results = []
    if not dataset_specs:
        return results

    for raw in dataset_specs.split(","):
        part = raw.strip()
        if not part:
            continue

        segs = [s.strip() for s in part.split("|")]
        if len(segs) < 2:
            raise ValueError(
                "--hf_datasets 格式错误，应为 name|split|text_field|label_field|subject_field"
            )

        name = segs[0]
        split = segs[1]
        text_field = segs[2] if len(segs) > 2 and segs[2] else "text"
        label_field = segs[3] if len(segs) > 3 and segs[3] else "label"
        subject_field = segs[4] if len(segs) > 4 and segs[4] else "subject"

        results.append({
            "name": name,
            "split": split,
            "text_field": text_field,
            "label_field": label_field,
            "subject_field": subject_field,
        })

    return results


def _ensure_sensitive_lexicon_repo(args: argparse.Namespace) -> Optional[Path]:
    if not args.add_sensitive_lexicon:
        return None

    repo_dir = Path(args.lexicon_repo_dir)
    if (repo_dir / "Vocabulary").exists() or (repo_dir / "Organized").exists():
        return repo_dir

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"[LEXICON] 克隆词库仓库: {args.lexicon_repo_url} -> {repo_dir}")
    subprocess.run(
        ["git", "clone", "--depth", "1", args.lexicon_repo_url, str(repo_dir)],
        check=True,
    )
    return repo_dir


def _map_lexicon_file_to_labels(file_name: str):
    low_name = file_name.lower()
    for keys, mapped in LEXICON_FILENAME_RULES:
        if any(k in low_name for k in keys):
            return mapped
    return "other", "medium"


def _load_sensitive_lexicon_rows(args: argparse.Namespace) -> List[Dict]:
    repo_dir = _ensure_sensitive_lexicon_repo(args)
    if repo_dir is None:
        return []

    roots = [repo_dir / "Vocabulary", repo_dir / "Organized"]
    txt_files = []
    for root in roots:
        if root.exists():
            txt_files.extend(sorted(root.rglob("*.txt")))

    if not txt_files:
        logger.warning("[LEXICON] 未发现词库 txt 文件，跳过")
        return []

    max_samples = args.lexicon_max_samples
    rows = []
    seen = set()

    for fp in txt_files:
        violation_type, risk_level = _map_lexicon_file_to_labels(fp.name)

        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    text = line.strip()
                    if not text or text.startswith("#"):
                        continue
                    if len(text) < 2:
                        continue
                    if text in seen:
                        continue

                    seen.add(text)
                    rows.append({
                        "text": text,
                        "is_violation": 1,
                        "violation_type": violation_type,
                        "risk_level": risk_level,
                        "source": f"Sensitive-lexicon/{fp.relative_to(repo_dir).as_posix()}",
                        "subject": fp.stem,
                        "raw_label": fp.name,
                    })

                    if max_samples > 0 and len(rows) >= max_samples:
                        logger.info(f"[LEXICON] 达到样本上限: {max_samples}")
                        return rows
        except OSError as e:
            logger.warning(f"[LEXICON] 读取失败: {fp} ({e})")

    return rows


def _map_harm_label_to_targets(label_text: str):
    key = str(label_text).strip()
    if key in CHINESE_HARM_LABEL_TO_TARGET:
        return CHINESE_HARM_LABEL_TO_TARGET[key]

    # 兜底
    if key == "不违规":
        return 0, "safe", "safe"
    return 1, "other", "medium"


def _convert_hf_dataset(
    dataset_name: str,
    split: str,
    text_field: str,
    label_field: str,
    subject_field: str,
    max_samples: int,
):
    from datasets import load_dataset

    logger.info(f"[HF] 加载数据集: {dataset_name} split={split}")
    ds = load_dataset(dataset_name, split=split)

    if max_samples > 0:
        limit = min(max_samples, len(ds))
        ds = ds.select(range(limit))
        logger.info(f"[HF] {dataset_name} 仅使用前 {limit} 条样本")

    rows = []
    for item in ds:
        text = str(item.get(text_field, "")).strip()
        if not text:
            continue

        if dataset_name.lower() == "zjunlp/chineseharm-bench":
            raw_label = str(item.get(label_field, "不违规")).strip()
            is_violation, violation_type, risk_level = _map_harm_label_to_targets(raw_label)
            subject = raw_label
        else:
            is_violation = _parse_violation_label(item.get(label_field, "不违规"))
            subject = str(item.get(subject_field, "不违规"))
            violation_type, risk_level = _map_subject_to_labels(subject, is_violation)
            raw_label = str(item.get(label_field, ""))

        rows.append({
            "text": text,
            "is_violation": int(is_violation),
            "violation_type": violation_type,
            "risk_level": risk_level,
            "source": f"{dataset_name}/{split}",
            "subject": subject,
            "raw_label": raw_label,
        })

    return rows


def build_jsonl_from_hf(args: argparse.Namespace):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "未安装 datasets，请先执行: pip install datasets"
        ) from e

    rows = []
    specs = _parse_dataset_specs(args.hf_datasets)

    if specs:
        for spec in specs:
            part_rows = _convert_hf_dataset(
                dataset_name=spec["name"],
                split=spec["split"],
                text_field=spec["text_field"],
                label_field=spec["label_field"],
                subject_field=spec["subject_field"],
                max_samples=args.hf_max_samples,
            )
            rows.extend(part_rows)
            logger.info(f"[HF] 已合并 {spec['name']} 样本: {len(part_rows)} 条")
    else:
        part_rows = _convert_hf_dataset(
            dataset_name=args.hf_dataset_name,
            split=args.hf_split,
            text_field=args.hf_text_field,
            label_field=args.hf_label_field,
            subject_field=args.hf_subject_field,
            max_samples=args.hf_max_samples,
        )
        rows.extend(part_rows)
        logger.info(f"[HF] 已读取样本: {len(part_rows)} 条")

    if not rows:
        raise RuntimeError("从 Hugging Face 数据集中未解析到有效样本")

    if args.add_sensitive_lexicon:
        base_rows_count = len(rows)
        lexicon_rows = _load_sensitive_lexicon_rows(args)
        if lexicon_rows:
            ratio = min(max(args.lexicon_max_ratio, 0.0), 0.95)
            # 约束: lexicon / (base + lexicon) <= ratio
            # 推导: lexicon <= ratio/(1-ratio) * base
            allowed_lexicon = int(base_rows_count * ratio / max(1e-8, 1.0 - ratio))

            if allowed_lexicon <= 0:
                lexicon_rows = []
                logger.info("[LEXICON] 词库比例上限过低，已跳过词库样本")
            elif len(lexicon_rows) > allowed_lexicon:
                rng_local = random.Random(args.hf_seed + 7)
                rng_local.shuffle(lexicon_rows)
                lexicon_rows = lexicon_rows[:allowed_lexicon]
                logger.info(
                    f"[LEXICON] 按比例上限截断词库样本: {allowed_lexicon} 条 "
                    f"(max_ratio={ratio:.2f})"
                )

            rows.extend(lexicon_rows)
            logger.info(f"[LEXICON] 已合并词库样本: {len(lexicon_rows)} 条")
        else:
            logger.warning("[LEXICON] 未合并任何词库样本")

    rng = random.Random(args.hf_seed)
    rng.shuffle(rows)

    val_ratio = min(max(args.hf_val_ratio, 0.0), 0.5)
    val_size = int(len(rows) * val_ratio)
    if val_ratio > 0 and val_size == 0 and len(rows) > 1:
        val_size = 1

    if val_size > 0:
        val_rows = rows[:val_size]
        train_rows = rows[val_size:]
    else:
        val_rows = []
        train_rows = rows

    if not train_rows:
        raise RuntimeError("训练集为空，请降低 --hf_val_ratio 或增加 --hf_max_samples")

    output_dir = Path(args.output_dir)
    train_path = output_dir / "hf_train.jsonl"
    val_path = output_dir / "hf_val.jsonl"

    _write_jsonl(train_path, train_rows)
    logger.info(f"[HF] 训练集已写入: {train_path} ({len(train_rows)} 条)")

    if val_rows:
        _write_jsonl(val_path, val_rows)
        logger.info(f"[HF] 验证集已写入: {val_path} ({len(val_rows)} 条)")
        return str(train_path), str(val_path)

    return str(train_path), None


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
                            f"  Val -> ViolAcc:{metrics['val_acc_violation']:.4f} "
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

    # Hugging Face 数据源
    p.add_argument("--hf_dataset_name", type=str, default=None, help="Hugging Face 数据集名，例如 SUSTech/ChineseSafe")
    p.add_argument("--hf_datasets", type=str, default=None,
                   help="多个HF数据集配置，格式: name|split|text|label|subject,name2|split2|text2|label2|subject2")
    p.add_argument("--hf_split", type=str, default="test", help="Hugging Face split 名")
    p.add_argument("--hf_text_field", type=str, default="text", help="Hugging Face 文本字段名")
    p.add_argument("--hf_label_field", type=str, default="label", help="Hugging Face 违规标签字段名")
    p.add_argument("--hf_subject_field", type=str, default="subject", help="Hugging Face 子类型字段名")
    p.add_argument("--hf_val_ratio", type=float, default=0.1, help="HF 数据随机划分验证集比例")
    p.add_argument("--hf_seed", type=int, default=42, help="HF 数据划分随机种子")
    p.add_argument("--hf_max_samples", type=int, default=0, help="HF 样本数上限，0 表示使用全部")

    # Sensitive-lexicon 词库增强
    p.add_argument("--add_sensitive_lexicon", action="store_true", help="将 Sensitive-lexicon 词库样本并入训练集")
    p.add_argument("--lexicon_repo_url", type=str, default="https://github.com/konsheng/Sensitive-lexicon.git",
                   help="Sensitive-lexicon 仓库地址")
    p.add_argument("--lexicon_repo_dir", type=str, default="./external/Sensitive-lexicon",
                   help="Sensitive-lexicon 本地目录；不存在时自动 clone")
    p.add_argument("--lexicon_max_samples", type=int, default=20000,
                   help="词库样本上限，0 表示不限制")
    p.add_argument("--lexicon_max_ratio", type=float, default=0.2,
                   help="词库样本占总样本比例上限 (默认0.2)")

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

    # 数据源选择: 本地 JSONL 或 Hugging Face
    if args.hf_datasets or args.hf_dataset_name:
        train_path, val_path = build_jsonl_from_hf(args)
        args.train_data = train_path
        if args.val_data is None:
            args.val_data = val_path
    elif not args.train_data:
        raise ValueError("请提供 --train_data，或提供 --hf_dataset_name 使用 Hugging Face 数据集")

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
