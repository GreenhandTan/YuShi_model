# Minimal Deployment Directory

此目录包含内容审核模型推理所需的最小文件集。支持 Linux、Docker 等多种环境。

## 文件清单

- `infer.py` —— 推理核心脚本
- `api_server.py` —— FastAPI HTTP 服务入口
- `model.py` —— 模型定义
- `dataset.py` —— Tokenizer 与数据处理
- `requirements.txt` —— 依赖列表
- `run_infer.sh` —— CLI 推理启动脚本
- `run_api.sh` —— HTTP 服务启动脚本
- `Dockerfile` —— Docker 镜像构建文件
- `docker-compose.yml` —— Docker Compose 编排文件
- `checkpoints/best.pt` —— 模型权重
- `checkpoints/vocab.json` —— 字符词表

## 快速开始 (Linux / macOS)

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) CLI 推理

```bash
chmod +x run_infer.sh
./run_infer.sh "你这行为涉嫌诈骗，我要举报你"
```

自定义参数：

```bash
./run_infer.sh "测试文本" --threshold 0.60 --max_length 160 --batch_size 32 --threads 2
```

### 3) HTTP 服务

```bash
chmod +x run_api.sh
./run_api.sh --host 0.0.0.0 --port 8000
```

启动后访问：

- API 文档: http://127.0.0.1:8000/docs
- 健康检查: http://127.0.0.1:8000/health

### 4) HTTP 调用示例 (curl)

单条审核：

```bash
curl -X POST "http://127.0.0.1:8000/audit" \
  -H "Content-Type: application/json" \
  -d '{"text":"测试文本"}'
```

批量审核：

```bash
curl -X POST "http://127.0.0.1:8000/audit/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts":["文本1", "文本2", "文本3"]}'
```

### 5) Python API 调用

```python
from infer import AuditInferencer

engine = AuditInferencer(
    checkpoint_path="./checkpoints/best.pt",
    vocab_path="./checkpoints/vocab.json"
)

result = engine.audit("测试文本")
print(result)
```

## Docker 部署

### 构建镜像

```bash
docker build -t content-audit:latest .
```

### 运行容器

```bash
docker run -d -p 8000:8000 --name audit-api content-audit:latest
```

### 测试服务

```bash
curl -X POST "http://127.0.0.1:8000/audit" \
  -H "Content-Type: application/json" \
  -d '{"text":"测试文本"}'
```

### Docker Compose 部署

```bash
docker-compose up -d
```

## 配置参数

### 推理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | `./checkpoints/best.pt` | 模型权重路径 |
| `--vocab` | `./checkpoints/vocab.json` | 词表路径 |
| `--device` | `auto` | 计算设备 (auto/cpu/cuda) |
| `--max_length` | 256 | 最大文本长度 |
| `--batch_size` | 4 | 批处理大小 |
| `--violation_conf_threshold` | 0.0 | 违规置信度阈值 |
| `--num_threads` | 2 | CPU 线程数 (仅 CPU 模式) |

### HTTP 服务参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 0.0.0.0 | 绑定主机 |
| `--port` | 8000 | 绑定端口 |
| `--workers` | 4 | 并发工作进程数 |

## 输出格式

### 推理结果 JSON

```json
{
  "is_violation": false,
  "risk_level": "safe",
  "violation_type": "safe",
  "confidence": 0.95,
  "reason": "文本合规"
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `is_violation` | bool | 是否违规 |
| `risk_level` | str | 风险等级 (safe/low/medium/high/critical) |
| `violation_type` | str | 违规类型 (safe/politics/adult/fraud/spam/violence) |
| `confidence` | float | 违规置信度 [0, 1] |
| `reason` | str | 详细说明 |

## 故障排查

### 找不到模型文件

```
确保 checkpoints/best.pt 和 checkpoints/vocab.json 存在
```

### CUDA 不可用

```bash
# 默认自动下降到 CPU
python ./infer.py --checkpoint ./checkpoints/best.pt --vocab ./checkpoints/vocab.json --prompt "text" --device cpu
```

### 性能优化

```bash
# 增加 workers 并发数 (HTTP 服务)
./run_api.sh --workers 8

# 增加线程数 (CPU 推理)
./run_infer.sh "text" --threads 8
```

## 许可证

见项目根目录 LICENSE
