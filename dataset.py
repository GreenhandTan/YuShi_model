"""
数据加载模块
使用 HuggingFace tokenizer 的内容审核数据集

- 使用预训练 BPE tokenizer，具备词级语义理解
- 兼容 HuggingFace AutoTokenizer
"""

import json
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# 标签编码映射 (与 V1 保持一致)
VIOLATION_TYPE_TO_ID = {
    "safe": 0, "politics": 1, "pornography": 2, "violence": 3,
    "abuse": 4, "spam": 5, "fraud": 6, "other": 7,
}
ID_TO_VIOLATION_TYPE = {v: k for k, v in VIOLATION_TYPE_TO_ID.items()}

RISK_LEVEL_TO_ID = {
    "safe": 0, "low": 1, "medium": 2, "high": 3, "critical": 4,
}
ID_TO_RISK_LEVEL = {v: k for k, v in RISK_LEVEL_TO_ID.items()}


def validate_label_consistency(
    is_violation: int,
    violation_type: str,
    risk_level: str,
) -> Tuple[bool, Optional[str]]:
    """验证标签一致性 (与 V1 一致)"""
    violation_type_lower = str(violation_type).lower().strip()
    risk_level_lower = str(risk_level).lower().strip()

    if is_violation == 0 and violation_type_lower != "safe":
        return False, f"矛盾标签: is_violation=0 但 violation_type={violation_type}"
    if violation_type_lower == "safe" and is_violation != 0:
        return False, f"矛盾标签: violation_type=safe 但 is_violation={is_violation}"
    if risk_level_lower == "safe" and is_violation != 0:
        return False, f"矛盾标签: risk_level=safe 但 is_violation={is_violation}"
    if is_violation == 1 and risk_level_lower == "safe":
        return False, f"矛盾标签: is_violation=1 但 risk_level=safe"

    return True, None


class AuditDataset(Dataset):
    """
    内容审核数据集 (使用 HuggingFace tokenizer)
    
    输出格式:
        - input_ids: token 序列
        - attention_mask: 注意力掩码
        - labels: 字典形式的标签
    """

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_seq_len: int = 256,
        text_field: str = "text",
    ):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_field = text_field

        if not self.file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        self.samples = []
        self._load_data()

        print(f"[{self.file_path.name}] 加载 {len(self.samples)} 条样本")

    def _load_data(self):
        """从 JSONL 文件加载数据"""
        inconsistent_count = 0
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] 第 {line_num} 行 JSON 解析失败: {e}")
                    continue

                text = data.get(self.text_field, "")
                if not text:
                    continue

                violation_label = int(data.get("is_violation", 0))
                vtype_str = str(data.get("violation_type", "safe")).lower()
                risk_str = str(data.get("risk_level", "safe")).lower()

                is_valid, error_msg = validate_label_consistency(violation_label, vtype_str, risk_str)
                if not is_valid:
                    print(f"[WARN] 第 {line_num} 行 - {error_msg}: {text[:50]}")
                    inconsistent_count += 1
                    if violation_label == 0:
                        vtype_str = "safe"
                        risk_str = "safe"
                    elif risk_str == "safe":
                        risk_str = "low"

                self.samples.append({
                    "text": text,
                    "is_violation": violation_label,
                    "violation_type_id": VIOLATION_TYPE_TO_ID.get(vtype_str, 0),
                    "risk_level_id": RISK_LEVEL_TO_ID.get(risk_str, 0),
                })

        if inconsistent_count > 0:
            print(f"[INFO] 检测到 {inconsistent_count} 条不一致的标签，已进行自动修复")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 使用 HuggingFace tokenizer 编码
        encoded = self.tokenizer(
            sample["text"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": {
                "violation": sample["is_violation"],
                "risk_level": sample["risk_level_id"],
                "violation_type": sample["violation_type_id"],
            },
            "raw_text": sample["text"],
        }


def audit_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """自定义 collate 函数"""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": {
            "violation": torch.tensor(
                [item["labels"]["violation"] for item in batch], dtype=torch.long
            ),
            "risk_level": torch.tensor(
                [item["labels"]["risk_level"] for item in batch], dtype=torch.long
            ),
            "violation_type": torch.tensor(
                [item["labels"]["violation_type"] for item in batch], dtype=torch.long
            ),
        },
        "raw_texts": [item["raw_text"] for item in batch],
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = 2,
) -> DataLoader:
    """创建 DataLoader"""
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    kwargs = {
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": audit_collate_fn,
        "pin_memory": pin_memory,
        "drop_last": shuffle,
    }

    if num_workers > 0 and persistent_workers:
        kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **kwargs)
