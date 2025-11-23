FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libgdal-dev \
    gdal-bin \
    python3-gdal \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 设置 GDAL 环境变量（rasterio 需要）
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口（如果需要）
EXPOSE 8000

# 启动命令（可以根据需要修改）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

