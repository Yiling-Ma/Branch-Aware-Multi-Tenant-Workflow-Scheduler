# WSI Cell Segmentation Scheduler

A distributed task scheduling system for Whole Slide Image (WSI) cell segmentation with branch-aware parallel execution and multi-tenant isolation.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.10+
- Conda (recommended)

### Setup with Docker Compose

1. **Start database and Redis**
   ```bash
   docker-compose up -d db redis
   ```

2. **Install dependencies**
   ```bash
   conda create -n penn python=3.10
   conda activate penn
   pip install -r requirements.txt
   ```

3. **Pre-download models** (optional)
   ```bash
   python download_model.py
   ```

4. **Start services**
   ```bash
   ./start_server.sh 4  # Starts scheduler, 4 workers, and API server
   ```

5. **Open UI**
   - Web Interface: http://127.0.0.1:8000/
   - API Docs (Swagger): http://127.0.0.1:8000/docs

### Local Setup (Without Docker)

1. Install PostgreSQL and Redis locally
2. Update `app/core/config.py`:
   ```python
   POSTGRES_HOST: str = "localhost"
   REDIS_HOST: str = "localhost"
   ```
3. Create database: `createdb scheduler_db`
4. Start services: `./start_server.sh 4`

## Usage

### 1. Create User
- Open File Manager in UI
- Enter username (e.g., "Yiling")
- Click "Create User"
- Copy the generated User ID

### 2. Upload Input Files
- In File Manager, select your user
- Upload `.svs` image files
- Files are stored in `user_files/{user_id}/inputs/`

### 3. Create Job
- Open "Test Scheduler" → "Create Custom Job"
- Paste your User ID
- Select uploaded file from dropdown
- Choose Branch ID, Job Type, and Job Name
- Click "Create Job"

### 4. View Results
- Jobs appear in Test Scheduler
- Click "Load Slide" to view results
- Use "Quality Check" to verify segmentation
- Use "Tissue Mask" to view tissue overlay

### 5. Export Results
- For completed cell segmentation jobs, click "Zarr" or "CSV" buttons
- Files are saved to `user_files/{user_id}/outputs/`
- All job outputs are automatically saved to user directory

## API Documentation

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Key Endpoints

**User & File Management**
- `POST /users/` - Create user
- `GET /users/{user_id}/files` - List user files
- `POST /users/{user_id}/files/upload` - Upload file
- `GET /users/{user_id}/files/download` - Download file
- `DELETE /users/{user_id}/files` - Delete file

**Job Management**
- `POST /workflows/` - Create workflow with jobs
- `GET /jobs/{job_id}/export?format=zarr` - Export results (Zarr)
- `GET /jobs/{job_id}/export?format=csv` - Export results (CSV)

## Export Format

### Zarr Format (Default)
```python
import zarr

store = zarr.ZipStore('cell_segmentation_results.zarr.zip', mode='r')
root = zarr.group(store=store)

# Metadata
print(f"Total cells: {root.attrs['total_cells']}")

# Arrays
cell_ids = root['cell_ids'][:]
centroids = root['centroids'][:]  # Shape: (N, 2)
polygon_counts = root['polygon_point_counts'][:]

# Individual polygons
cell_1_polygon = root['polygons']['cell_1'][:]  # Shape: (points, 2)
```

### CSV Format
```csv
cell_id,centroid_x,centroid_y,polygon_coords,polygon_point_count,area_pixels
1,100,200,"[[95,195],[105,195],[105,205],[95,205]]",4,100
```

## Scaling to 10× More Jobs/Users

### Configuration
```python
# app/core/config.py
MAX_WORKERS: int = 40  # Increase from 4
MAX_ACTIVE_USERS: int = 30  # Increase from 3
```

### Infrastructure
- **Workers**: Deploy across multiple machines with shared database
- **Database**: Use read replicas, increase connection pool
- **Redis**: Use Redis Cluster for distributed tracking
- **API**: Deploy multiple instances behind load balancer
- **Storage**: Move to object storage (S3, MinIO) for large files

### Database Pool
```python
# app/db.py
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)
```

## Testing and Monitoring

### Testing
```bash
# Test export functionality
python test_export.py

# Verify setup
python verify_setup.py
```

### Monitoring
- **Health Check**: http://127.0.0.1:8000/docs (check endpoints)
- **Logs**: Check scheduler and worker logs
- **Metrics**: Monitor database connections, Redis memory, worker CPU/memory

### Production Checklist
- [ ] Database backups configured
- [ ] Log rotation set up
- [ ] Monitoring dashboards (Grafana)
- [ ] Alerting configured
- [ ] SSL/TLS certificates
- [ ] Rate limiting implemented

## Project Structure

```
my-scheduler/
├── app/
│   ├── main.py          # FastAPI routes
│   ├── models.py        # Database models
│   ├── scheduler.py     # Task scheduler
│   ├── worker.py        # Job execution
│   └── core/config.py   # Configuration
├── static/
│   └── index.html       # Web UI
├── user_files/          # User data (persistent)
├── docker-compose.yml   # Docker services
└── start_server.sh      # Start script
```

## Features

- **Branch-Aware Scheduling**: Serial within branches, parallel across branches
- **Multi-Tenant**: Each user isolated with own files and jobs
- **File Management**: Persistent storage in `user_files/{user_id}/`
- **Export**: Zarr and CSV formats with polygon coordinates
- **Real-Time Progress**: Live job status and logs

## Troubleshooting

**Database connection failed**
```bash
docker-compose up -d db redis
docker-compose ps
```

**Jobs stuck in PENDING**
- Check scheduler: `ps aux | grep app.scheduler`
- Check workers: `ps aux | grep app.worker`
- Restart: `./start_server.sh 4`

**Port already in use**
```bash
pkill -f "uvicorn.*app.main"
./restart_server.sh
```
