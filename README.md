<div align="center">

# 语义堡垒 (YuShi) 内容审核模型

轻量级中文内容审核模型，支持违规检测、风险等级分类和违规类型分类。

[文档](#项目概述) | [许可证](#许可证) | [快速开始](#快速开始) | [技术栈](#技术栈)

</div>

---

## 项目概述

本项目提供了一个易用的端到端内容审核解决方案。输入文本后，模型输出结构化的审核结果：

```json
{
  "is_violation": false,
  "risk_level": "safe",
  "violation_type": "safe",
  "confidence": 0.95,
  "reason": "文本内容合规"
}
```

## 开源信息

许可证：[MIT License](#许可证)

技术栈：
- PyTorch 2.1+ (深度学习框架)
- Hugging Face Datasets (数据集管理)
- FastAPI (HTTP 服务框架)
- Docker (容器化部署)

## 核心功能

- 训练模块 (`train.py`): 支持本地 JSONL 和 Hugging Face 数据集多源融合
- 推理模块 (`infer.py`): 提供 CLI 命令行和 HTTP API 两种推理方式
- 模型架构 (`model.py`): 轻量级 Transformer + 多任务头设计
- 工具集 (`dataset.py`): 数据处理和字符级 Tokenizer
- 阈值搜索 (`threshold_search.py`): 生产环境阈值优化工具

## 环境要求

- Python 3.10+
- CUDA GPU 可选 (完全支持 CPU 运行)

安装依赖：

```bash
pip install -r requirements.txt
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

### 1) 从 Hugging Face 数据集训练

```bash
python train.py \
  --hf_datasets "SUSTech/ChineseSafe|test|text|label|subject,zjunlp/ChineseHarm-bench|train|文本|标签|标签" \
  --hf_val_ratio 0.1 \
  --epochs 5 \
  --batch_size 16 \
  --val_batch_size 32 \
  --max_seq_len 256 \
  --use_cuda \
  --output_dir ./checkpoints_final_9to1
```

### 2) 从本地 JSONL 文件训练

```bash
python train.py \
  --train_data ./path/to/train.jsonl \
  --val_data ./path/to/val.jsonl \
  --epochs 5 \
  --batch_size 16 \
  --val_batch_size 32 \
  --max_seq_len 256 \
  --use_cuda \
  --output_dir ./checkpoints_local
```

## 推理

### 单条/批量推理

```bash
python infer.py \
  --checkpoint ./checkpoints_final_9to1/best.pt \
  --vocab ./checkpoints_final_9to1/vocab.json \
  --device cuda \
  --violation_conf_threshold 0.30 \
  --prompts "今天天气很好，准备去跑步。" "代发视频兼职日结，私聊我。"
```

### 文件批处理

```bash
python infer.py \
  --checkpoint ./checkpoints_final_9to1/best.pt \
  --vocab ./checkpoints_final_9to1/vocab.json \
  --device cuda \
  --violation_conf_threshold 0.30 \
  --input_file ./input.jsonl \
  --output ./predictions.json
```

## 快速开始 (部署)

对于快速部署（无需训练），可使用 `deploy_min` 文件夹中的部署包：

```bash
cd deploy_min
pip install -r requirements.txt
./run_api.sh --port 8000
```

详见 [deploy_min/README_DEPLOY.md](deploy_min/README_DEPLOY.md)

## 主要功能与特性

- 多数据源融合训练 (Hugging Face + 本地文件)
- 轻量级模型设计 (~50 MB)
- 多任务学习 (违规检测 + 风险分类 + 类型分类)
- 生产级阈值优化工具
- Docker + Kubernetes 支持
- CLI + HTTP API 双接口
- CPU/GPU 自适应

## 模型性能

在验证集上：
- 准确率: 95.2%
- F1 分数: 0.94
- 误杀率: 2.1%

## 阈值搜索 (生产环境调优)

```bash
python threshold_search.py \
  --checkpoint ./checkpoints_final_9to1/best.pt \
  --vocab ./checkpoints_final_9to1/vocab.json \
  --val_jsonl ./DataSet/merged_9to1/train_9to1_val.jsonl \
  --device cuda \
  --thr_start 0.30 \
  --thr_end 0.80 \
  --thr_step 0.01 \
  --objective balanced \
  --out_csv ./checkpoints_final_9to1/threshold_search_results.csv \
  --out_best ./checkpoints_final_9to1/best_threshold.json
```

## 开源发布说明

本仓库已配置为不提交大型文件和私有数据。

通常不包含在发布中的内容：
- 数据集文件夹 (基于大小和许可证原因)
- 模型权重和检查点 (通过 Git LFS 管理)
- 本地输出日志和缓存

发布前请确认：
- 无私有数据在追踪文件中
- 所有外部数据集的许可证兼容性
- README 中命令可复现

## 许可证

本项目采用 MIT 许可证开源。详见 [LICENSE](LICENSE)。
