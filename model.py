"""
内容审核专家模型 (ContentAuditExpert)
基于预训练中文 Encoder (chinese-roberta-wwm-ext) 的文本审核分类模型

架构:
- 预训练 Encoder backbone (12层, 768维)
- 多任务分类头 (违规检测 + 风险等级 + 违规类型 + 置信度)
- 融合池化 (last-token + mean-pooling)

设计目标:
- 轻量化：CPU 云服务器可流畅运行 (~400MB 模型, ~20ms/条)
- 输出标准 JSON 格式审核结果
- 支持多维度审核（违规类型、风险等级、置信度等）
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any

from transformers import AutoTokenizer, AutoModel


# ============================================================
# 审核标签体系 (与 V1 保持一致)
# ============================================================

VIOLATION_TYPES = [
    "safe",           # 安全/合规
    "politics",       # 涉政敏感
    "pornography",    # 色情低俗
    "violence",       # 暴力血腥
    "abuse",          # 辱骂人身攻击
    "spam",           # 垃圾广告
    "fraud",          # 欺诈诈骗
    "other",          # 其他违规
]

RISK_LEVELS = ["safe", "low", "medium", "high", "critical"]


# ============================================================
# 主模型
# ============================================================


class ContentAuditExpert(nn.Module):
    """
    内容审核专家模型
    
    基于预训练 Encoder (chinese-roberta-wwm-ext) + 多任务分类头。
    输入文本，输出结构化 JSON 审核结果。
    
    架构特点:
    - 预训练 Encoder backbone (12层, 768维)
    - 多任务分类头 (违规检测 + 风险等级 + 违规类型 + 置信度)
    - 融合池化 (last-token + mean-pooling)
    - 支持 CPU 推理，模型体积 ~400MB
    """

    RESULT_TEMPLATE = {
        "is_violation": False,
        "risk_level": "safe",
        "violation_type": "safe",
        "confidence": 1.0,
        "reason": "",
    }

    def __init__(
        self,
        backbone_name: str = "hfl/chinese-roberta-wwm-ext",
        dropout: float = 0.1,
        pool_last_weight: float = 0.6,
        freeze_backbone_layers: int = 0,
    ):
        """
        Args:
            backbone_name: HuggingFace 预训练模型名称
            dropout: Dropout 概率
            pool_last_weight: 融合池化中 last-token 的权重
            freeze_backbone_layers: 冻结 backbone 底部 N 层 (0=不冻结)
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.pool_last_weight = float(max(0.0, min(1.0, pool_last_weight)))

        # ---- 加载预训练 Encoder ----
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for roberta-base

        # ---- 冻结底层参数 (可选，减少训练开销) ----
        if freeze_backbone_layers > 0:
            # 冻结 embedding 层
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            # 冻结指定数量的 encoder 层
            for i in range(min(freeze_backbone_layers, len(self.encoder.encoder.layer))):
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = False

        # ---- 多任务分类头 (与 V1 保持一致的设计) ----
        self.head_dropout = nn.Dropout(min(dropout + 0.1, 0.4))

        # 二分类：是否违规
        self.violation_head = nn.Linear(hidden_size, 2)

        # 风险等级分类 (5 类)
        self.risk_head = nn.Linear(hidden_size, len(RISK_LEVELS))

        # 违规类型分类 (8 类: safe + 7种违规)
        self.type_head = nn.Linear(hidden_size, len(VIOLATION_TYPES))

        # 置信度回归 (单值)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # 初始化分类头权重
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """初始化分类头权重"""
        for module in [self.violation_head, self.risk_head, self.type_head]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        for module in self.confidence_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        编码输入文本，返回序列表示

        Returns:
            last_hidden_state: (B, T, D) 序列隐状态
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        池化：获取用于分类的句子级表示

        策略: 融合 last-token 和 mean-pooling (与 V1 一致)
        """
        if attention_mask is not None:
            # 找每个样本最后一个有效位置
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            seq_lengths = lengths - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_pooled = hidden_states[batch_idx, seq_lengths]

            valid_mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            mean_pooled = (hidden_states * valid_mask).sum(dim=1) / lengths.unsqueeze(-1).to(hidden_states.dtype)
        else:
            last_pooled = hidden_states[:, -1, :]
            mean_pooled = hidden_states.mean(dim=1)

        pooled = self.pool_last_weight * last_pooled + (1.0 - self.pool_last_weight) * mean_pooled
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 — 训练时使用

        Returns:
            各分类头的 logits / 输出字典
        """
        hidden_states = self._encode(input_ids, attention_mask)
        pooled = self._pool(hidden_states, attention_mask)
        pooled_dropped = self.head_dropout(pooled)

        return {
            "violation_logits": self.violation_head(pooled_dropped),      # (B, 2)
            "risk_logits": self.risk_head(pooled_dropped),                # (B, 5)
            "type_logits": self.type_head(pooled_dropped),                # (B, 8)
            "confidence": self.confidence_head(pooled_dropped).squeeze(-1),  # (B,)
        }

    @torch.inference_mode()
    def audit(
        self,
        text: str,
        tokenizer=None,
        max_length: int = 256,
    ) -> Dict[str, Any]:
        """
        对单条文本执行审核推理，返回标准 JSON 兼容的结果

        Args:
            text: 待审核的文本字符串
            tokenizer: HuggingFace tokenizer 实例
            max_length: 最大截断长度

        Returns:
            审核结果 dict
        """
        self.eval()

        if tokenizer is None:
            raise ValueError("推理时必须提供 tokenizer")

        # 使用 HuggingFace tokenizer 编码
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device_for_infer())
        attention_mask = encoded["attention_mask"].to(self.device_for_infer())

        outputs = self.forward(input_ids, attention_mask)

        # 解析各类预测结果
        violation_pred = outputs["violation_logits"].argmax(dim=-1).item()
        is_violation = bool(violation_pred == 1)

        risk_idx = outputs["risk_logits"].argmax(dim=-1).item()
        risk_level = RISK_LEVELS[risk_idx]
        if not is_violation:
            risk_level = "safe"

        type_idx = outputs["type_logits"].argmax(dim=-1).item()
        violation_type = VIOLATION_TYPES[type_idx]

        confidence = round(outputs["confidence"].item(), 4)

        reason = self._build_reason(is_violation, risk_level, violation_type, confidence, text[:50])

        return {
            "is_violation": is_violation,
            "risk_level": risk_level,
            "violation_type": violation_type if is_violation else "safe",
            "confidence": confidence,
            "reason": reason,
        }

    @torch.inference_mode()
    def audit_batch(
        self,
        texts: List[str],
        tokenizer=None,
        max_length: int = 256,
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        批量审核多条文本
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            encoded = tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device_for_infer())
            attention_mask = encoded["attention_mask"].to(self.device_for_infer())

            outputs = self.forward(input_ids, attention_mask)

            for j in range(len(batch_texts)):
                is_violation = outputs["violation_logits"][j].argmax().item() == 1
                risk_level = RISK_LEVELS[outputs["risk_logits"][j].argmax().item()]
                if not is_violation:
                    risk_level = "safe"
                violation_type = VIOLATION_TYPES[outputs["type_logits"][j].argmax().item()]
                confidence = round(outputs["confidence"][j].item(), 4)
                reason = self._build_reason(
                    is_violation, risk_level,
                    violation_type if is_violation else "safe",
                    confidence, batch_texts[j][:50],
                )

                results.append({
                    "is_violation": is_violation,
                    "risk_level": risk_level,
                    "violation_type": violation_type if is_violation else "safe",
                    "confidence": confidence,
                    "reason": reason,
                })

        return results

    def device_for_infer(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _build_reason(is_violation: bool, risk_level: str, vtype: str,
                      confidence: float, text_preview: str) -> str:
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
        level_cn_map = {
            "safe": "安全", "low": "低风险", "medium": "中等风险",
            "high": "高风险", "critical": "严重风险",
        }

        if not is_violation:
            return f"内容合规，{level_cn_map[risk_level]}，置信度 {confidence:.1%}"

        type_desc = type_cn_map.get(vtype, "违规")
        return (
            f"检测到{type_desc}，"
            f"{level_cn_map.get(risk_level, risk_level)}，"
            f"置信度 {confidence:.1%}。"
        )

    def to_json_output(self, result: Dict[str, Any]) -> str:
        return json.dumps(result, ensure_ascii=False, indent=2)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("内容审核专家模型测试 (Encoder-based)")
    print("=" * 60)

    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    # 初始化模型
    model = ContentAuditExpert(
        backbone_name="hfl/chinese-roberta-wwm-ext",
        dropout=0.1,
        pool_last_weight=0.6,
        freeze_backbone_layers=6,  # 冻结底部 6 层，减少训练开销
    )

    params = model.count_parameters()
    print(f"\n模型参数量: {params['total']:,} ({params['trainable']:,} 可训练)")

    # 模拟前向传播
    texts = ["今天天气真好", "这是一段违规测试文本"]
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    model.train()
    outputs = model(encoded["input_ids"], encoded["attention_mask"])
    print(f"\n训练模式输出:")
    print(f"  violation_logits: {outputs['violation_logits'].shape}")
    print(f"  risk_logits:      {outputs['risk_logits'].shape}")
    print(f"  type_logits:      {outputs['type_logits'].shape}")
    print(f"  confidence:       {outputs['confidence'].shape}")

    print("\n[OK] 测试通过!")
