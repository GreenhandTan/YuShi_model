"""
数据加载模块
为内容审核模型提供训练/验证数据加载支持

数据格式: JSONL，每行一个样本，包含:
  - text: 待审核文本内容
  - is_violation: 是否违规 (0/1)
  - violation_type: 违规类型标签
  - risk_level: 风险等级标签 (可选)
"""

import json
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


# ============================================================
# 标签编码映射
# ============================================================

# 中文内容审核标签体系
# violation_type: 违规类型 (7类 + safe)
VIOLATION_TYPE_TO_ID = {
    "safe": 0,
    "politics": 1,        # 政治/敏感
    "pornography": 2,     # 色情/低俗
    "violence": 3,        # 暴力/血腥
    "abuse": 4,           # 辱骂/人身攻击
    "spam": 5,            # 广告/垃圾信息
    "fraud": 6,           # 欺诈/诈骗
    "other": 7,           # 其他违规
}
ID_TO_VIOLATION_TYPE = {v: k for k, v in VIOLATION_TYPE_TO_ID.items()}

# risk_level: 风险等级 (5级)
RISK_LEVEL_TO_ID = {
    "safe": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}
ID_TO_RISK_LEVEL = {v: k for k, v in RISK_LEVEL_TO_ID.items()}

RISK_LEVEL_TO_ID = {
    "safe": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}
ID_TO_RISK_LEVEL = {v: k for k, v in RISK_LEVEL_TO_ID.items()}


def validate_label_consistency(
    is_violation: int,
    violation_type: str,
    risk_level: str,
) -> Tuple[bool, Optional[str]]:
    """
    验证标签一致性。

    规则:
    - 如果 is_violation=0，则 violation_type 必须是 "safe"
    - 如果 violation_type="safe"，则 is_violation 必须是 0
    - risk_level "safe" 应该对应 is_violation=0

    返回: (is_valid, error_message)
    """
    violation_type_lower = str(violation_type).lower().strip()
    risk_level_lower = str(risk_level).lower().strip()

    # 规则1: 不违规必须是 safe 类型
    if is_violation == 0 and violation_type_lower != "safe":
        return False, f"矛盾标签: is_violation=0 但 violation_type={violation_type}"

    # 规则2: safe 类型必须不违规
    if violation_type_lower == "safe" and is_violation != 0:
        return False, f"矛盾标签: violation_type=safe 但 is_violation={is_violation}"

    # 规则3: safe 风险等级应对应 is_violation=0
    if risk_level_lower == "safe" and is_violation != 0:
        return False, f"矛盾标签: risk_level=safe 但 is_violation={is_violation}"

    # 规则4: 如果违规，风险等级不应为 safe
    if is_violation == 1 and risk_level_lower == "safe":
        return False, f"矛盾标签: is_violation=1 但 risk_level=safe"

    return True, None


class AuditDataset(Dataset):
    """
    内容审核数据集
    
    从 JSONL 文件加载标注数据，每个样本包含文本和多维审核标签。
    
    输出格式:
        - input_ids: token 序列 (已 padding)
        - attention_mask: 注意力掩码
        - labels: 字典形式的标签 {
            "violation": 是否违规 (0/1),
            "risk_level": 风险等级 id,
            "violation_type": 违规类型 id,
          }
    """

    def __init__(
        self,
        file_path: str,
        tokenizer=None,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
        text_field: str = "text",
    ):
        """
        Args:
            file_path: JSONL 数据文件路径
            tokenizer: 分词器实例 (需有 encode 方法)
            max_seq_len: 最大序列长度
            pad_token_id: padding token id
            text_field: 文本字段名
        """
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.text_field = text_field

        if not self.file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        # 加载并解析数据
        self.samples = []
        self._load_data()

        print(f"[{self.file_path.name}] "
              f"加载 {len(self.samples)} 条样本")

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

                # 编码标签
                violation_label = int(data.get("is_violation", 0))
                vtype_str = str(data.get("violation_type", "safe")).lower()
                risk_str = str(data.get("risk_level", "safe")).lower()

                # 验证标签一致性
                is_valid, error_msg = validate_label_consistency(violation_label, vtype_str, risk_str)
                if not is_valid:
                    print(f"[WARN] 第 {line_num} 行 - {error_msg}: {text[:50]}")
                    inconsistent_count += 1
                    # 尝试自动修复
                    if violation_label == 0:
                        vtype_str = "safe"
                        risk_str = "safe"
                    elif risk_str == "safe":
                        risk_str = "low"  # 违规但 risk_level 为 safe，改为 low

                sample = {
                    "text": text,
                    "is_violation": violation_label,
                    "violation_type_id": VIOLATION_TYPE_TO_ID.get(vtype_str, 0),
                    "risk_level_id": RISK_LEVEL_TO_ID.get(risk_str, 0),
                }
                self.samples.append(sample)

        if inconsistent_count > 0:
            print(f"[INFO] 检测到 {inconsistent_count} 条不一致的标签，已进行自动修复")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Tokenize
        tokens = self.tokenizer.encode(sample["text"])
        tokens = tokens[:self.max_seq_len]  # 截断

        seq_len = len(tokens)
        input_ids = tokens + [self.pad_token_id] * (self.max_seq_len - seq_len)
        attention_mask = [1] * seq_len + [0] * (self.max_seq_len - seq_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": {
                "violation": sample["is_violation"],
                "risk_level": sample["risk_level_id"],
                "violation_type": sample["violation_type_id"],
            },
            "raw_text": sample["text"],
        }


def audit_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """自定义 collate 函数 —— 打包 batch 数据"""
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
    """创建 DataLoader 的便捷函数"""
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    dataloader_kwargs = {
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": audit_collate_fn,
        "pin_memory": pin_memory,
        "drop_last": shuffle,
    }

    if num_workers > 0 and persistent_workers:
        dataloader_kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        dataset,
        **dataloader_kwargs,
    )


# ============================================================
# 简易 Tokenizer（用于演示和快速测试）
# ============================================================

class SimpleTokenizer:
    """字符级简易分词器"""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.token2id = vocab if vocab else {
            self.PAD_TOKEN: 0, self.UNK_TOKEN: 1,
            self.EOS_TOKEN: 2, self.BOS_TOKEN: 3,
        }
        self.id2token = {v: k for k, v in self.token2id.items()}

    @classmethod
    def from_texts(cls, texts: List[str], min_freq: int = 1) -> "SimpleTokenizer":
        instance = cls()
        freq = {}
        for text in texts:
            for ch in text:
                freq[ch] = freq.get(ch, 0) + 1
        idx = len(instance.token2id)
        for char, count in sorted(freq.items()):
            if count >= min_freq and char not in instance.token2id:
                instance.token2id[char] = idx
                instance.id2token[idx] = char
                idx += 1
        print(f"构建词汇表完成: {len(instance.token2id)} 个 token")
        return instance

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_token_id(self) -> int:
        return 0

    def encode(self, text: str) -> List[int]:
        return [self.token2id.get(c, 1) for c in text]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        special = {self.PAD_TOKEN, self.UNK_TOKEN, self.EOS_TOKEN, self.BOS_TOKEN}
        chars = []
        for tid in token_ids:
            t = self.id2token.get(tid, self.UNK_TOKEN)
            if skip_special_tokens and t in special:
                continue
            chars.append(t)
        return "".join(chars)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)
        print(f"词汇表已保存: {path}")

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            return cls(vocab=json.load(f))


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("数据模块测试")
    print("=" * 60)

    # 创建测试数据文件
    test_data = [
        {"text": "这是一段正常的内容", "is_violation": 0, "violation_type": "safe", "risk_level": "safe"},
        {"text": "包含敏感违规内容的示例", "is_violation": 1, "violation_type": "politics", "risk_level": "high"},
        {"text": "广告推广信息垃圾信息", "is_violation": 1, "violation_type": "spam", "risk_level": "low"},
    ]

    tmp_dir = tempfile.mkdtemp()
    test_file = os.path.join(tmp_dir, "test_audit.jsonl")

    with open(test_file, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 构建 tokenizer 和数据集
    texts = [d["text"] for d in test_data]
    tokenizer = SimpleTokenizer.from_texts(texts)
    print(f"词汇量: {tokenizer.vocab_size}")

    dataset = AuditDataset(
        file_path=test_file,
        tokenizer=tokenizer,
        max_seq_len=64,
    )
    print(f"数据集大小: {len(dataset)}")

    # 测试单条
    sample = dataset[0]
    print(f"\n样本0:")
    print(f"  input_ids 形状:   {sample['input_ids'].shape}")
    print(f"  attention_mask:   {sample['attention_mask'].shape[:20]}...")
    print(f"  labels:           {sample['labels']}")
    print(f"  原文:             {sample['raw_text']}")

    # 测试 DataLoader
    dataloader = create_dataloader(dataset, batch_size=2, num_workers=0)
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  input_ids:      {batch['input_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  violation 标签: {batch['labels']['violation'].tolist()}")
        print(f"  type 标签:      {batch['labels']['violation_type'].tolist()}")
        print(f"  raw_texts:      {batch['raw_texts']}")
        break

    print("\n[OK] 数据模块测试通过!")
