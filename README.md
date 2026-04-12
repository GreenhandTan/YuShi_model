<div align="center">

# 御史 (YuShi Model) 内容审核模型

轻量级中文内容审核模型，支持违规检测、风险等级分类和违规类型分类。

> 御史：中国古代一种官职，负责稽查百官

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
- 推理模块 (`infer.py` / `infer_onnx.py`): 本地可用 PyTorch，部署默认 ONNX 加速
- 模型架构 (`model.py`): 轻量级 Transformer + 多任务头设计
- 工具集 (`dataset.py`): 数据处理和字符级 Tokenizer
- 阈值搜索 (`threshold_search.py`): 生产环境阈值优化工具

模型池化策略：
- 默认采用融合池化 (last-token + mean-pooling，默认权重 0.6/0.4)
- 该策略兼顾长文本尾部语义和短文本稳定性，提升短句/词组审核鲁棒性

## 环境要求

- Python 3.10+
- CUDA GPU 可选 (完全支持 CPU 运行)

安装依赖：

**仅 CPU 推理和训练：**

```bash
pip install -r requirements-cpu.txt
```

**GPU 训练和推理（CUDA 11.8）：**

```bash
pip install -r requirements-gpu.txt
```

**其他 CUDA 版本（GPU）：**

```bash
# CUDA 12.1
pip install datasets>=2.14.0 wandb>=0.17.0 numpy>=1.24.0
pip install 'torch>=2.1.0' --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install datasets>=2.14.0 wandb>=0.17.0 numpy>=1.24.0
pip install 'torch>=2.1.0' --index-url https://download.pytorch.org/whl/cu124
```

详见 [PyTorch 官方安装指南](https://pytorch.org/)

## 数据格式

训练数据使用 JSONL 格式（每行一个 JSON 对象）：

```json
{"text": "...", "is_violation": 0, "violation_type": "safe", "risk_level": "safe"}
```

标签定义：

- 违规类型: `safe|politics|pornography|violence|abuse|spam|fraud|other`
- 风险等级: `safe|low|medium|high|critical`

## 训练

### 0) 从 ChineseHarm-bench 转换为训练 JSONL

如果你想直接从 GitHub 仓库获取 ChineseHarm-bench 数据并转成可训练数据集，执行：

```bash
python prepare_chineseharm_bench.py \
  --source github \
  --clone_if_missing \
  --github_branch main \
  --github_bench_path benchmark/bench.json \
  --output_all ./DataSet/chineseharm_bench/chineseharm_all.jsonl \
  --output_train ./DataSet/chineseharm_bench/chineseharm_train.jsonl \
  --output_val ./DataSet/chineseharm_bench/chineseharm_val.jsonl \
  --report_json ./DataSet/chineseharm_bench/chineseharm_report.json \
  --val_ratio 0.1
```

若远程 raw 文件访问超时，脚本会回退到本地仓库路径；加上 `--clone_if_missing` 可在本地目录不存在时自动 clone。

若你更希望从 Hugging Face 拉取，也可以执行：

```bash
python prepare_chineseharm_bench.py \
  --source hf \
  --dataset_name zjunlp/ChineseHarm-bench \
  --split train \
  --output_all ./DataSet/chineseharm_bench/chineseharm_all.jsonl \
  --output_train ./DataSet/chineseharm_bench/chineseharm_train.jsonl \
  --output_val ./DataSet/chineseharm_bench/chineseharm_val.jsonl \
  --report_json ./DataSet/chineseharm_bench/chineseharm_report.json \
  --val_ratio 0.1
```

生成后的字段会统一为 `text/is_violation/violation_type/risk_level`，可以直接拿来训练。

### 0.5) 从 ChineseSafe 转换并并入本地数据集

```bash
python prepare_chinesesafe.py \
  --dataset_name SUSTech/ChineseSafe \
  --split test \
  --max_samples 200000 \
  --output_all ./DataSet/chinesesafe/chinesesafe_all.jsonl \
  --output_train ./DataSet/chinesesafe/chinesesafe_train.jsonl \
  --output_val ./DataSet/chinesesafe/chinesesafe_val.jsonl \
  --report_json ./DataSet/chinesesafe/chinesesafe_report.json \
  --val_ratio 0.1
```

将 ChineseSafe 添加到本地合并训练集：

```bash
python merge_local_datasets.py \
  --inputs ./DataSet/chineseharm_bench/chineseharm_all.jsonl,./DataSet/chinesesafe/chinesesafe_all.jsonl \
  --val_ratio 0.1 \
  --output_all ./DataSet/local_combined/combined_all.jsonl \
  --output_train ./DataSet/local_combined/combined_train.jsonl \
  --output_val ./DataSet/local_combined/combined_val.jsonl \
  --report_json ./DataSet/local_combined/combined_report.json
```

注意：ChineseSafe 当前公开样本量约 2 万（不是 20 万），当 `used_rows < requested_max_samples` 时属于数据源公开数量限制。

### 1) 从 funNLP 筛选词表并转换为训练 JSONL

先将 funNLP 中可用词表转换为本项目训练格式：

```bash
python prepare_funnlp_lexicon.py \
  --clone_if_missing \
  --repo_dir ./external/funNLP \
  --output_jsonl ./DataSet/funnlp/funnlp_lexicon_filtered.jsonl \
  --report_json ./DataSet/funnlp/funnlp_lexicon_report.json \
  --max_total 50000
```

输出 JSONL 字段与训练集一致（`text/is_violation/violation_type/risk_level`），可直接合并进你的本地训练数据。

可选：若需要加入停用词作为 `safe` 对照样本，可增加参数 `--add_safe_from_stopwords`。

### 2) 从 funNLP 词库 + Sensitive-lexicon 训练

```bash
python train.py \
  --train_data ./DataSet/funnlp/funnlp_lexicon_filtered.jsonl \
  --add_sensitive_lexicon \
  --epochs 5 \
  --batch_size 16 \
  --val_batch_size 32 \
  --max_seq_len 256 \
  --use_cuda \
  --output_dir ./checkpoints_funnlp_lexicon
```

这条路线只保留 funNLP 词库和 Sensitive-lexicon 两类来源，不再混入旧的合并安全样本或其他外部训练集。

### 3) 从本地 JSONL 文件训练

```bash
python train.py \
  --train_data ./path/to/train.jsonl \
  --val_data ./path/to/val.jsonl \
  --add_sensitive_lexicon \
  --epochs 5 \
  --batch_size 16 \
  --val_batch_size 32 \
  --max_seq_len 256 \
  --use_cuda \
  --output_dir ./checkpoints_local
```

如果你希望尽量把 Sensitive-lexicon 中的违禁词都并入训练集，保持默认的 `--lexicon_max_ratio 0.95` 即可；它会把词库样本扩到接近全量，并写入增强后的训练集再开始训练。

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

部署与训练分离策略：
- 本地训练：使用 PyTorch（`train.py` + `.pt`）
- 发布部署：使用 ONNX 推理（`model.onnx`），不在部署包中包含训练脚本

当前采用 deploy 目录驱动的自动发布流程：
- 本地维护 `deploy/` 下的部署原始文件
- 当 `deploy/` 内容变更并推送后，GitHub Actions 会自动打包 ZIP/TAR.GZ 并发布到 GitHub Releases

本地部署后的测试网页服务文件夹为 `web_test/`，用于快速验证模型在本地 WSL 服务上的审核效果。

部署时请从 Releases 下载部署包并解压，然后在解压目录中执行：

```bash
# CPU ONNX 推理
pip install onnxruntime fastapi uvicorn pydantic numpy
bash run_api.sh --port 8000
```

GPU 主机可改为：

```bash
pip install onnxruntime-gpu fastapi uvicorn pydantic numpy
bash run_api.sh --port 8000 --onnx_gpu
```

若要从本地 PyTorch 检查点导出 ONNX：

```bash
python export_onnx.py \
  --checkpoint ./checkpoints_final_9to1/best.pt \
  --vocab ./checkpoints_final_9to1/vocab.json \
  --output ./checkpoints_final_9to1/model.onnx \
  --max_length 256
```

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
  --checkpoint ./checkpoints_funnlp_lexicon/best.pt \
  --vocab ./checkpoints_funnlp_lexicon/vocab.json \
  --val_jsonl ./path/to/val.jsonl \
  --device cuda \
  --thr_start 0.30 \
  --thr_end 0.80 \
  --thr_step 0.01 \
  --objective balanced \
  --out_csv ./checkpoints_funnlp_lexicon/threshold_search_results.csv \
  --out_best ./checkpoints_funnlp_lexicon/best_threshold.json
```

## 开源发布说明

本仓库已配置为不提交大型文件和私有数据。

通常不包含在发布中的内容：
- 数据集文件夹 (基于大小和许可证原因)
- 训练检查点目录 (如 `checkpoints_final_9to1/`)
- 本地输出日志和缓存

部署包发布方式：
- 采用 `deploy/` 目录原始文件触发发布
- 当 `deploy/**` 发生变更时，GitHub Actions 自动执行打包与发布
- 发布时会自动生成 ZIP 与 TAR.GZ，并为压缩包文件名追加当天日期后缀
- 同时生成并上传 `checksums.txt` 用于校验完整性

推荐本地发布更新流程（示例）：

```bash
# 1) 更新 deploy 下需要发布的原始文件
# 2) 提交并推送
git add deploy/
git commit -m "chore: update deploy source files"
git push origin main
```

推送后工作流会自动创建新 Release，并上传自动生成的压缩包。

Release 页面部署步骤（ONNX 推理）与此一致：

```bash
# 1) 下载并解压 deploy_*.zip 或 deploy_*.tar.gz
# 2) 进入解压目录

# CPU
pip install -r requirements-cpu.txt
bash run_api.sh --port 8000

# GPU
pip install -r requirements-gpu.txt
bash run_api.sh --port 8000 --onnx_gpu

# 健康检查
curl http://127.0.0.1:8000/health

# 单条审核测试
curl -X POST http://127.0.0.1:8000/audit \
  -H "Content-Type: application/json" \
  -d '{"text":"代发兼职日结，私聊我"}'
```

发布前请确认：
- 无私有数据在追踪文件中
- 所有外部数据集的许可证兼容性
- README 中命令可复现

## 许可证

本项目采用 MIT 许可证开源。详见 [LICENSE](LICENSE)。
