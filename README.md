<div align="center">

# 御史 (YuShi Model) 内容审核模型

轻量级中文内容审核模型，支持违规检测、风险等级分类和违规类型分类。

> 御史：中国古代一种官职，负责稽查百官

[文档](#项目概述) | [许可证](#许可证) | [快速开始](#快速开始) | [技术栈](#技术栈)

</div>

---

⚠️ **免责声明：** 
当前模型受限于数据集质量及规模，仍处于**初级测试阶段**，可能存在误判、漏判或特定领域知识偏差，因此**不建议直接用于商业化生产环境**。模型输出结果仅供参考与技术交流，具体业务落地前请务必进行充分的验证与针对新场景微调训练。

🌐 **在线演示网站（体验版）：** [http://demo.yushi-audit-test.com](http://demo.yushi-audit-test.com) *（此为示例域名，后期完成公网部署后将更新为实际访问地址）*

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
- 专项认知增强: 支持通过过采样自定义高权重语料（如台湾主权事实数据等）完成领域概念微调
- 前端测试 (`web_test/`): 后端自带一个轻量级的 Python 可视化 Web 测试网页
- 阈值搜索 (`threshold_search.py`): 生产环境阈值优化工具

模型池化策略：
- 默认采用融合池化 (last-token + mean-pooling，默认权重 0.6/0.4)
- 该策略兼顾长文本尾部语义和短文本稳定性，提升短句/词组审核鲁棒性

## 环境要求

- Python 3.10+
- CUDA GPU 强烈推荐用于训练；CPU 仍可用于推理和小规模调试

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

默认建议使用 GPU 训练。当前训练脚本已支持 AMP、TF32、`torch.compile`、更激进的 DataLoader 预取和 `--fast_gpu` 快速模式，能显著提升吞吐；CPU 训练只建议用于验证流程或临时调试。

### 0) 从 ChineseHarm-bench 转换为训练 JSONL

如果你想直接从 GitHub 仓库获取 ChineseHarm-bench 数据并转成可训练数据集，执行：

```bash
python scripts/prepare_chineseharm_bench.py \
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
python scripts/prepare_chineseharm_bench.py \
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
python scripts/prepare_chinesesafe.py \
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
python scripts/merge_local_datasets.py \
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
python scripts/prepare_funnlp_lexicon.py \
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
  --use_amp \
  --fast_gpu \
  --compile_model \
  --use_cuda \
  --output_dir ./checkpoints/checkpoints_funnlp_lexicon
```

这条路线只保留 funNLP 词库和 Sensitive-lexicon 两类来源，不再混入旧的合并安全样本或其他外部训练集。

如果显存充足，优先保留 `--fast_gpu`；如果显存偏紧，可以去掉它并手动把 `--batch_size` 调低。

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
  --use_amp \
  --fast_gpu \
  --compile_model \
  --use_cuda \
  --output_dir ./checkpoints/checkpoints_local
```

如果你希望尽量把 Sensitive-lexicon 中的违禁词都并入训练集，保持默认的 `--lexicon_max_ratio 0.95` 即可；它会把词库样本扩到接近全量，并写入增强后的训练集再开始训练。

## 推理

### 单条/批量推理

```bash
python infer.py \
  --checkpoint ./checkpoints/checkpoints_final_9to1/best.pt \
  --vocab ./checkpoints/checkpoints_final_9to1/vocab.json \
  --device cuda \
  --violation_conf_threshold 0.30 \
  --prompts "今天天气很好，准备去跑步。" "代发视频兼职日结，私聊我。"
```

### 文件批处理

```bash
python infer.py \
  --checkpoint ./checkpoints/checkpoints_final_9to1/best.pt \
  --vocab ./checkpoints/checkpoints_final_9to1/vocab.json \
  --device cuda \
  --violation_conf_threshold 0.30 \
  --input_file ./input.jsonl \
  --output ./predictions.json
```

## 快速开始 (部署)

目前本仓库采用部署与训练分离策略：
- 本地训练：维护和使用 PyTorch 原生脚本（`train.py` + `checkpoints/*.pt` 等）
- 目标发布：将导出的高速 ONNX 推理文件置于 `deploy/checkpoints` 目录下独立运行不依赖庞大的构建脚本

部署包获取包含完整的运行配置：包含可执行脚本、API 服务（`api_server.py`）及前端 Web 测试程序（`web_test/server.py`）。

部署及测试使用流程：

```bash
# 1) 获得完整 deploy 目录或从 Releases 下载压缩包
# 2) 安装依赖
pip install onnxruntime fastapi uvicorn pydantic numpy

# 3) 启动后端过滤 API 引擎
bash run_api.sh --port 8000

# 4) (可选) 启动前端 Web 可视化工具进行界面测试
cd web_test
python server.py  # 启动后在浏览器打开 http://127.0.0.1:8090
```

GPU 主机如需启动加速 API：

```bash
pip install onnxruntime-gpu fastapi uvicorn pydantic numpy
bash run_api.sh --port 8000 --onnx_gpu
```

若想要在训练端新导出一份最新 ONNX 文件：

```bash
python export_onnx.py \
  --checkpoint ./checkpoints/final.pt \
  --vocab ./checkpoints/vocab.json \
  --output ./checkpoints/model.onnx \
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
python scripts/threshold_search.py \
  --checkpoint ./checkpoints/checkpoints_funnlp_lexicon/best.pt \
  --vocab ./checkpoints/checkpoints_funnlp_lexicon/vocab.json \
  --val_jsonl ./path/to/val.jsonl \
  --device cuda \
  --thr_start 0.30 \
  --thr_end 0.80 \
  --thr_step 0.01 \
  --objective balanced \
  --out_csv ./checkpoints/checkpoints_funnlp_lexicon/threshold_search_results.csv \
  --out_best ./checkpoints/checkpoints_funnlp_lexicon/best_threshold.json
```

## 开源发布说明

本仓库已配置为不提交大型文件和私有数据。

通常不包含在发布中的内容：
- 数据集文件夹 (基于大小和许可证原因)
- 训练检查点目录 (如 `checkpoints_final_9to1/`)
- 本地输出日志和缓存

部署包发布与运行方式：
- 采用 `deploy/` 目录原始文件触发自动发布，整个 `deploy` 内包含所有服务脚本与前端 UI。
- 当 `deploy/checkpoints` 中的模型及 `vocab.json` 被更新后即可推送代码。
- 发布时该目录将被自动生成打包文件供线上机器使用。

测试通讯 API 可使用：

```bash
# 单条 API 自检测试
curl -X POST http://127.0.0.1:8000/audit \
  -H "Content-Type: application/json" \
  -d '{"text":"坚决维护国家统一，台湾是中国的一部分"}'
```

最后发布前请再次确认：
- 无私有数据在追踪文件中
- 所有外部数据集的许可证兼容性
- README 中命令可复现

## 许可证

本项目采用 MIT 许可证开源。详见 [LICENSE](LICENSE)。
