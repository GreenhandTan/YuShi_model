<div align="center">

# 御史 (YuShi Model) 内容审核模型

轻量级中文内容审核模型 — 规则引擎 + Encoder 模型两级架构，CPU 服务器即可部署。

> 御史：中国古代一种官职，负责稽查百官

[快速开始](#快速开始) | [架构设计](#架构设计) | [训练](#训练) | [部署](#部署) | [许可证](#许可证)

</div>

---

**免责声明：**
当前模型受限于数据集质量及规模，仍处于**初级测试阶段**，可能存在误判、漏判或特定领域知识偏差，因此**不建议直接用于商业化生产环境**。模型输出结果仅供参考与技术交流，具体业务落地前请务必进行充分的验证与针对新场景微调训练。

## 项目概述

本项目提供了一个**轻量级、可本地部署**的中文内容审核解决方案。核心设计目标：

- **不依赖外部 LLM API**：无隐私风险，无调用成本
- **不需要 GPU**：普通 CPU 云服务器即可运行
- **两级审核架构**：规则引擎极速处理明确场景，Encoder 模型处理灰色地带
- **多维度输出**：违规检测 + 风险等级 + 违规类型 + 置信度

输入文本后，输出结构化审核结果：

```json
{
  "is_violation": true,
  "risk_level": "high",
  "violation_type": "spam",
  "confidence": 0.95,
  "reason": "命中违规关键词: 刷单兼职",
  "layer": "rule"
}
```

## 架构设计

```
用户输入文本
       ↓
┌─────────────────────────────────────┐
│  第一层：规则引擎 (<1ms)             │
│  - 黑名单关键词匹配                  │
│  - 正则模式匹配 (变体/谐音)          │
│  - 白名单快速放行                    │
│  命中 → 直接返回结果                  │
└─────────────┬───────────────────────┘
              ↓ 未命中 (~20% 流量)
┌─────────────────────────────────────┐
│  第二层：Encoder 模型 (~20ms)        │
│  - chinese-roberta-wwm-ext backbone │
│  - 多任务分类头                      │
│  - 置信度阈值后处理                  │
└─────────────────────────────────────┘
```

**性能预估 (4 核 CPU)**：
- 规则引擎：<1ms/条
- 模型推理：~20ms/条
- 批量吞吐：100~200 条/秒
- 模型大小：~400MB
- 内存占用：<1GB

## 技术栈

- PyTorch 2.1+
- Transformers (HuggingFace)
- FastAPI
- chinese-roberta-wwm-ext (预训练 Encoder)

## 快速开始

### 环境要求

- Python 3.10+
- 4 核 CPU / 8GB 内存即可（无需 GPU）

### 安装依赖

```bash
# CPU 训练和推理
pip install -r requirements-train-cpu.txt

# 或仅部署推理
pip install -r requirements-deploy-cpu.txt
```

### 测试规则引擎

```bash
python rule_engine.py
```

### 测试模型加载

```bash
python model.py
```

## 数据格式

训练数据使用 JSONL 格式（每行一个 JSON 对象）：

```json
{"text": "...", "is_violation": 0, "violation_type": "safe", "risk_level": "safe"}
```

标签定义：

- 违规类型: `safe|politics|pornography|violence|abuse|spam|fraud|other`
- 风险等级: `safe|low|medium|high|critical`

## 训练

```bash
python train.py \
  --train_data DataSet/cleaned/dataset_final_v4.0_train.jsonl \
  --val_data DataSet/cleaned/dataset_final_v4.0_val.jsonl \
  --backbone hfl/chinese-roberta-wwm-ext \
  --freeze_layers 6 \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 2e-4 \
  --use_cuda \
  --fast_gpu \
  --output_dir ./checkpoints/
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backbone` | `hfl/chinese-roberta-wwm-ext` | 预训练 Encoder 模型 |
| `--freeze_layers` | `6` | 冻结 backbone 底部 N 层 |
| `--max_seq_len` | `256` | 最大序列长度 |
| `--learning_rate` | `2e-4` | 学习率（backbone 使用 0.1 倍） |
| `--pool_last_weight` | `0.6` | 融合池化 last-token 权重 |

## 推理

### 单条审核

```bash
python infer.py \
  --checkpoint ./checkpoints/best.pt \
  --prompt "今天天气很好，准备去跑步。"
```

### 批量审核

```bash
python infer.py \
  --checkpoint ./checkpoints/best.pt \
  --prompts "今天天气很好" "代发视频兼职日结，私聊我。"
```

### 文件批处理

```bash
python infer.py \
  --checkpoint ./checkpoints/best.pt \
  --input_file ./input.jsonl \
  --output ./predictions.json
```

### 交互模式

```bash
python infer.py \
  --checkpoint ./checkpoints/best.pt \
  --interactive
```

### 自定义规则目录

```bash
python infer.py \
  --checkpoint ./checkpoints/best.pt \
  --rules_dir ./rules \
  --prompt "测试文本"
```

### 跳过规则引擎（仅用模型）

```bash
python infer.py \
  --checkpoint ./checkpoints/best.pt \
  --skip_rules \
  --prompt "测试文本"
```

## 部署

### 启动 API 服务

```bash
# 方式一：使用启动脚本
bash run_api.sh 8000

# 方式二：直接启动
python api_server.py --port 8000 --checkpoint ./checkpoints/best.pt
```

### API 接口

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 单条审核
curl -X POST http://127.0.0.1:8000/audit \
  -H "Content-Type: application/json" \
  -d '{"text":"代发兼职日结，私聊我"}'

# 批量审核
curl -X POST http://127.0.0.1:8000/audit/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["今天天气真好","约炮加我微信"]}'
```

### Web 测试界面

```bash
cd web_test
python server.py
# 浏览器打开 http://127.0.0.1:8090
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHECKPOINT_PATH` | `./checkpoints/best.pt` | 模型路径 |
| `RULES_DIR` | `./rules` | 规则目录 |
| `VIOLATION_CONF_THRESHOLD` | `0.30` | 违规置信度阈值 |
| `INFER_MAX_LENGTH` | `256` | 最大文本长度 |
| `INFER_BATCH_SIZE` | `8` | 批量大小 |

## 规则引擎扩展

### 自定义规则目录结构

```
rules/
├── blacklist.json   # 黑名单关键词
├── whitelist.json   # 白名单关键词
└── patterns.json    # 正则模式
```

### 动态添加规则（代码方式）

```python
from rule_engine import RuleEngine

engine = RuleEngine()
engine.add_blacklist("违规词", violation_type="other", risk_level="high")
engine.add_whitelist("安全短语")
engine.save_rules("./rules")
```

## 主要特性

- **两级审核架构**：规则引擎 (<1ms) + Encoder 模型 (~20ms)
- **预训练知识**：基于 chinese-roberta-wwm-ext，天然理解中文语义
- **多任务学习**：违规检测 + 风险等级 + 违规类型 + 置信度
- **规则可扩展**：支持黑白名单 + 正则模式，可热更新
- **CPU 友好**：~400MB 模型，<1GB 内存，无需 GPU
- **CLI + HTTP API**：支持单条/批量/文件/交互模式

## 项目结构

```
YuShi_model/
├── model.py                 # Encoder 模型定义
├── dataset.py               # 数据加载 (HuggingFace tokenizer)
├── rule_engine.py           # 规则引擎
├── train.py                 # 训练脚本
├── infer.py                 # 推理脚本 (两级架构)
├── api_server.py            # FastAPI 服务
├── export_onnx.py           # ONNX 导出 (可选)
├── run_api.sh               # API 启动脚本
├── checkpoints/             # 模型检查点 (训练后生成)
├── web_test/                # Web 测试界面
└── DataSet/                 # 训练数据集
```

## 开源发布说明

本仓库已配置为不提交大型文件和私有数据。

部署包发布：推送到 main/master 分支且核心文件变更时，GitHub Actions 自动打包发布。

## 许可证

本项目采用 MIT 许可证开源。详见 [LICENSE](LICENSE)。
