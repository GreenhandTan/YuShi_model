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

- 部署推理模块 (`infer_onnx.py` / `api_server.py`): ONNX Runtime 高性能推理
- 模型架构 (`model.py`): 轻量级 Transformer + 多任务头设计
- 工具集 (`dataset.py`): 数据处理和字符级 Tokenizer
- 阈值搜索 (`threshold_search.py`): 生产环境阈值优化工具

## 环境要求

- Python 3.10+
- CUDA GPU 可选 (完全支持 CPU 运行)

安装依赖：

**仅 CPU 推理：**

```bash
pip install -r requirements-cpu.txt
```

**GPU 推理（CUDA 11.8）：**

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

## 部署说明

deploy 目录只用于推理部署，不包含训练流程。

- 线上部署仅使用 ONNX 文件：`./checkpoints/model.onnx` + `./checkpoints/vocab.json`
- 不需要 `train.py`、`.pt` 检查点或训练数据
- 当前默认违规阈值为 `0.30`

## 推理

### 单条/批量推理

```bash
python infer_onnx.py \
  --model ./checkpoints/model.onnx \
  --vocab ./checkpoints/vocab.json \
  --use_gpu \
  --violation_conf_threshold 0.30 \
  --prompts "今天天气很好，准备去跑步。" "代发视频兼职日结，私聊我。"
```

### 文件批处理

```bash
python infer_onnx.py \
  --model ./checkpoints/model.onnx \
  --vocab ./checkpoints/vocab.json \
  --use_gpu \
  --violation_conf_threshold 0.30 \
  --input_file ./input.jsonl \
  --output ./predictions.json
```

## 快速开始 (部署)

部署与训练分离策略：
- 本地训练：在仓库根目录完成
- 发布部署：deploy 包只保留 ONNX 推理所需文件

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
  --checkpoint ./checkpoints_complete_6sources/best.pt \
  --vocab ./checkpoints_complete_6sources/vocab.json \
  --output ./deploy/checkpoints/model.onnx \
  --max_length 256

cp ./checkpoints_complete_6sources/vocab.json ./deploy/checkpoints/vocab.json
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
- 训练检查点目录 (如 `checkpoints_complete_6sources/`)
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
