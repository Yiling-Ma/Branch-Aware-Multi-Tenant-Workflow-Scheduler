FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
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

# Copy dependency files
COPY requirements.txt .

# Set GDAL environment variables (required by rasterio))
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (if needed))
EXPOSE 8000

# Start command (can be modified as needed))
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

