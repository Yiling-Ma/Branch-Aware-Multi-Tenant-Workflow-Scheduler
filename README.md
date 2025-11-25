# WSI Cell Segmentation Scheduler

A distributed task scheduling system for Whole Slide Image (WSI) cell segmentation with branch-aware parallel execution and multi-tenant isolation.

## Demo

📹 **Demo Video**: [Watch the demo video](https://github.com/Yiling-Ma/Branch-Aware-Multi-Tenant-Workflow-Scheduler/releases/latest) (available in Releases)

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

3. **Download test data** 
manually download from: [OpenSlide Test Data](https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/)
   
   Recommended test files:
   - `CMU-1-Small-Region.svs` (1.85 MB) - Small, fast for testing
   - `CMU-1.svs` (169.33 MB) - Medium size
   - `CMU-2.svs` (372.65 MB) - Large file

4. **Start services**
   ```bash
   ./start_server.sh  # Starts scheduler, 4 workers, and API server
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
4. Start services: `./start_server.sh`

## Usage

### 1. Create User
- Open File Manager in UI
- Enter username (e.g., "Yiling")
- Click "Create User"
- Copy the generated User ID

### 2. Upload Input Files
- In File Manager, select your user
- Upload `.svs` image files (Whole Slide Images)
- Files are stored in `user_files/{user_id}/inputs/`
- **Test Data**: Download sample WSI files from [OpenSlide Test Data](https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/)

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

## Monitoring


### Monitoring
- **Health Check**: http://127.0.0.1:8000/docs (check endpoints)
- **Logs**: Check scheduler and worker logs
- **Metrics**: Monitor database connections, Redis memory, worker CPU/memory


## Project Structure

```
my-scheduler/
├── app/
│   ├── main.py             # FastAPI entry point
│   ├── api/
│   │   └── routes.py       # API routes and endpoints
│   ├── models/
│   │   └── sql_models.py   # Database models
│   ├── services/
│   │   ├── scheduler.py    # Task scheduler
│   │   ├── worker.py       # Job execution & Image processing
│   │   ├── rate_limit.py   # Rate limiting logic
│   │   └── metrics.py      # Monitoring metrics
│   ├── db/
│   │   └── session.py      # Database session management
│   └── core/
│       └── config.py       # Configuration settings
├── static/
│   └── index.html          # Web UI
├── user_files/             # User data (persistent)
├── docker-compose.yml      # Docker services
└── start_server.sh         # Start script
```
