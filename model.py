"""
内容审核专家模型 (ContentAuditExpert)
基于 Transformer Decoder 的轻量级文本审核分类模型

设计目标:
- 轻量化：可在云服务器 (CPU / 低显存 GPU) 上流畅运行
- 输出标准 JSON 格式审核结果
- 支持多维度审核（违规类型、风险等级、置信度等）
- 推理延迟低，适合在线部署
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any


# ============================================================
# 模型组件
# ============================================================


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """RMS 归一化"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class Attention(nn.Module):
    """多头因果注意力"""

    def __init__(self, n_heads: int, dim: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0, "dim 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """SwiGLU 前馈网络"""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Transformer 解码器块"""

    def __init__(self, n_heads: int, dim: int, ffn_dim: int, norm_eps: float = 1e-6,
                 dropout: float = 0.0):
        super().__init__()
        self.attention = Attention(n_heads, dim, dropout)
        self.feed_forward = FeedForward(dim, ffn_dim, dropout)
        self.attention_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), mask=mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


# ============================================================
# 内容审核标签体系
# ============================================================

AUDIT_LABELS = {
    "is_violation": "是否违规",
    "risk_level": "风险等级 (safe/low/medium/high/critical)",
    "violation_type": "违规类型 (politics/pornography/violence/fraud/spam/other/safe)",
    "confidence": "置信度 (0.0~1.0)",
    "reason": "审核理由说明",
}

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
    
    轻量级 Transformer Decoder，输入文本，输出结构化 JSON 审核结果。
    
    架构特点:
    - Transformer Decoder-only + RoPE + RMSNorm + SwiGLU
    - 分类头输出多维度审核标签
    - 支持 CPU/GPU 推理，模型体积小、延迟低
    """

    # 审核结果模板（确保输出格式标准化）
    RESULT_TEMPLATE = {
        "is_violation": False,
        "risk_level": "safe",
        "violation_type": "safe",
        "confidence": 1.0,
        "reason": "",
    }

    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        ffn_multiplier: int = 4,
        max_seq_len: int = 1024,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        pad_token_id: int = 0,
        pool_last_weight: float = 0.6,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        # 融合池化权重：保留长文本末尾语义，同时引入全局均值增强短文本稳定性
        self.pool_last_weight = float(max(0.0, min(1.0, pool_last_weight)))
        ffn_dim = dim * ffn_multiplier

        # ---- 主干网络 ----
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(n_heads, dim, ffn_dim, norm_eps, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim, norm_eps)

        # ---- 多任务分类头 ----
        # 使用融合池化：last-token 与 mean-pooling 融合表示
        hidden_size = dim

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

        # 初始化权重
        self.apply(self._init_weights)

        # 因果掩码
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1),
            persistent=False,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码输入文本，返回序列表示
        
        Returns:
            hidden_states: (B, T, D) 序列隐状态
        """
        _, T = input_ids.shape
        h = self.tok_embeddings(input_ids)

        mask = self.causal_mask[:T, :T].unsqueeze(0)  # (1, T, T)
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            mask = mask + (1.0 - padding_mask.float()) * (-10000.0)

        for layer in self.layers:
            h = layer(h, mask=mask)

        h = self.norm(h)
        return h

    def _pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        池化：获取用于分类的句子级表示

        策略: 融合 last-token 和 mean-pooling
        - last-token: 对长文本更敏感，保留尾部判别信号
        - mean-pooling: 对短文本更稳定，降低单 token 波动
        """
        if attention_mask is not None:
            # 找每个样本最后一个有效位置
            lengths = attention_mask.sum(dim=1).clamp(min=1)  # (B,)
            seq_lengths = lengths - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_pooled = hidden_states[batch_idx, seq_lengths]  # (B, D)

            valid_mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, T, 1)
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
        
        Args:
            input_ids: (B, T) token ids
            attention_mask: (B, T) 注意力掩码
        
        Returns:
            各分类头的 logits / 输出字典
        """
        hidden_states = self._encode(input_ids, attention_mask)
        pooled = self._pool(hidden_states, attention_mask)

        return {
            "violation_logits": self.violation_head(pooled),      # (B, 2)
            "risk_logits": self.risk_head(pooled),                # (B, 5)
            "type_logits": self.type_head(pooled),                # (B, 7)
            "confidence": self.confidence_head(pooled).squeeze(-1),  # (B,)
        }

    @torch.inference_mode()
    def audit(
        self,
        text: str,
        tokenizer=None,
        max_length: int = 512,
    ) -> Dict[str, Any]:
        """
        对单条文本执行审核推理，返回标准 JSON 兼容的结果
        
        这是模型的核心接口 —— 输入文本，直接输出结构化的审核结果。
        
        Args:
            text: 待审核的文本字符串
            tokenizer: 分词器实例
            max_length: 最大截断长度
        
        Returns:
            审核结果 dict (可直接 json.dumps 序列化):
            {
                "is_violation": bool,
                "risk_level": str,
                "violation_type": str,
                "confidence": float,
                "reason": str,
            }
        """
        self.eval()

        # Tokenize
        if tokenizer is not None:
            tokens = tokenizer.encode(text)[:max_length]
            input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device_for_infer())
        else:
            # 无 tokenizer 时假设输入已经是 tensor
            raise ValueError("推理时必须提供 tokenizer")

        attention_mask = torch.ones_like(input_ids)

        # 前向推理
        outputs = self.forward(input_ids, attention_mask)

        # ---- 解析各类预测结果 ----

        # 是否违规: violation_head 输出 [非违规, 违规]
        violation_pred = outputs["violation_logits"].argmax(dim=-1).item()
        is_violation = bool(violation_pred == 1)

        # 风险等级
        risk_idx = outputs["risk_logits"].argmax(dim=-1).item()
        risk_level = RISK_LEVELS[risk_idx]
        
        # 强逻辑约束：如果不违规，则风险等级强制为 safe
        if not is_violation:
            risk_level = "safe"

        # 违规类型
        type_idx = outputs["type_logits"].argmax(dim=-1).item()
        violation_type = VIOLATION_TYPES[type_idx]

        # 置信度
        confidence = round(outputs["confidence"].item(), 4)

        # 构建审核理由
        reason = self._build_reason(is_violation, risk_level, violation_type, confidence, text[:50])

        result = {
            "is_violation": is_violation,
            "risk_level": risk_level,
            "violation_type": violation_type if is_violation else "safe",
            "confidence": confidence,
            "reason": reason,
        }

        return result

    @torch.inference_mode()
    def audit_batch(
        self,
        texts: List[str],
        tokenizer=None,
        max_length: int = 512,
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        批量审核多条文本
        
        Args:
            texts: 文本列表
            tokenizer: 分词器
            max_length: 最大长度
            batch_size: 批量大小 (自动分批处理)
        
        Returns:
            审核结果列表
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize 并 padding
            encoded_batch = []
            masks = []
            for text in batch_texts:
                tokens = tokenizer.encode(text)[:max_length]
                encoded_batch.append(tokens)
                masks.append([1] * len(tokens))

            # Padding 到同一长度
            max_len = max(len(t) for t in encoded_batch)
            pad_id = getattr(tokenizer, 'pad_token_id', 0)
            
            padded = []
            for t in encoded_batch:
                padded.append(t + [pad_id] * (max_len - len(t)))
            for m in masks:
                m.extend([0] * (max_len - len(m)))

            input_ids = torch.tensor(padded, dtype=torch.long, device=self.device_for_infer())
            attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device_for_infer())

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
        """获取推理设备"""
        return next(self.parameters()).device

    @staticmethod
    def _build_reason(is_violation: bool, risk_level: str, vtype: str,
                      confidence: float, text_preview: str) -> str:
        """生成审核理由描述"""
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
        """将审核结果序列化为标准 JSON 字符串"""
        return json.dumps(result, ensure_ascii=False, indent=2)

    def count_parameters(self) -> Dict[str, int]:
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("内容审核专家模型测试")
    print("=" * 60)

    model = ContentAuditExpert(
        vocab_size=10000,
        dim=256,
        n_layers=4,
        n_heads=4,
        ffn_multiplier=4,
        max_seq_len=512,
    )

    params = model.count_parameters()
    print(f"\n模型参数量: {params['total']:,} ({params['trainable']:,} 可训练)")

    # 模拟前向传播
    B, T = 2, 64
    input_ids = torch.randint(0, 10000, (B, T))
    mask = torch.ones(B, T, dtype=torch.long)
    
    model.train()
    outputs = model(input_ids, mask)
    print(f"\n训练模式输出:")
    print(f"  violation_logits: {outputs['violation_logits'].shape}")   # (2, 2)
    print(f"  risk_logits:      {outputs['risk_logits'].shape}")         # (2, 5)
    print(f"  type_logits:      {outputs['type_logits'].shape}")         # (B, 8)
    print(f"  confidence:       {outputs['confidence'].shape}")          # (2,)

    # 模拟推理输出格式
    print(f"\n推理输出示例 (JSON):")
    demo_result = {
        "is_violation": True,
        "risk_level": "high",
        "violation_type": "fraud",
        "confidence": 0.9234,
        "reason": "检测到欺诈或诈骗相关内容，高风险，置信度 92.3%。",
    }
    print(model.to_json_output(demo_result))

    print("\n[OK] 测试通过!")
