FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用文件
COPY *.py ./
COPY checkpoints/ ./checkpoints/

# 暴露端口
EXPOSE 8000

# 默认启动 HTTP 服务
CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
