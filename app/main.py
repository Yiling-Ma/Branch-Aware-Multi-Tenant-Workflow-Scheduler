from fastapi import FastAPI, Depends, HTTPException, Header, Response, UploadFile, File
from fastapi.staticfiles import StaticFiles  # New import
from fastapi.responses import RedirectResponse  # For redirect
from fastapi.middleware.cors import CORSMiddleware  # CORS middleware to prevent cross-origin issues
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import select, delete
from app.db import init_db, get_session, engine
from app.models import User, Workflow, Job
from uuid import UUID
import uuid
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
from datetime import datetime
import json
import os
import openslide
import cv2
import numpy as np
import random
import io
import shutil
from pathlib import Path
from starlette.responses import StreamingResponse, Response, FileResponse
from instanseg import InstanSeg
import torch
from openslide.deepzoom import DeepZoomGenerator  # Core component

# Default image path (consistent with worker.py)
DEFAULT_IMAGE = "/Users/yiling/Desktop/penn_proj/my-scheduler/data/CMU-1-Small-Region.svs"

app = FastAPI()

# Preload model for real-time verification (load at startup for faster response)
# Note: This will consume GPU memory; can be changed to lazy loading if memory is tight
try:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    debug_instanseg = InstanSeg("brightfield_nuclei", verbosity=0, device=device)
    print(f"✅ Debug InstanSeg model loaded (device: {device})")
except Exception as e:
    print(f"⚠️ Failed to preload debug model: {e}")
    debug_instanseg = None

# Root path redirects to frontend page
@app.get("/")
async def root():
    """Root path redirects to frontend UI"""
    return RedirectResponse(url="/static/index.html")

# 1. Configure CORS (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Mount static file directory (key step)
# This allows accessing /static/xxx.jpg to get files
static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global cache: avoid reopening SVS files on every tile request (use Redis in production)
SLIDE_CACHE = {}

def get_slide_generator(path):
    """Get or create DeepZoomGenerator instance (with caching)"""
    if path not in SLIDE_CACHE:
        if not os.path.exists(path):
            return None
        slide = openslide.OpenSlide(path)
        # tile_size=256 is standard size, overlap=1 for seamless tiling
        SLIDE_CACHE[path] = DeepZoomGenerator(slide, tile_size=256, overlap=1)
    return SLIDE_CACHE[path]

# Initialize database tables on startup
@app.on_event("startup")
async def on_startup():
    await init_db()

# --- API 1: Create test user ---
class UserCreate(BaseModel):
    username: str

@app.post("/users/", response_model=User)
async def create_user(user_data: UserCreate, session: AsyncSession = Depends(get_session)):
    # Check if user with same name already exists
    statement = select(User).where(User.username == user_data.username)
    result = await session.exec(statement)
    existing_user = result.first()
    
    if existing_user:
        # If user exists, return existing user
        return existing_user
    
    # If not exists, create new user
    user = User(username=user_data.username)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user

@app.get("/users/")
async def get_all_users(session: AsyncSession = Depends(get_session)):
    """Get all users"""
    statement = select(User)
    result = await session.exec(statement)
    users = result.all()
    return users

# --- User File Management ---
USER_FILES_DIR = "user_files"

def get_user_files_dir(user_id: UUID) -> str:
    """Get user's file directory path"""
    return os.path.join(USER_FILES_DIR, str(user_id))

def ensure_user_dirs(user_id: UUID):
    """Ensure user directories exist"""
    user_dir = get_user_files_dir(user_id)
    inputs_dir = os.path.join(user_dir, "inputs")
    outputs_dir = os.path.join(user_dir, "outputs")
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    return user_dir, inputs_dir, outputs_dir

async def copy_export_file_to_user_dir(session: AsyncSession, job: Job, source_path: str, filename: str):
    """
    Copy an exported file to user's output directory.
    This is called when user exports results (Zarr/CSV).
    """
    try:
        # Get workflow and user
        from app.models import Workflow
        workflow_stmt = select(Workflow).where(Workflow.id == job.workflow_id)
        workflow_result = await session.exec(workflow_stmt)
        workflow = workflow_result.first()
        
        if not workflow:
            print(f"⚠️  Workflow not found for job {job.id}, skipping export file copy")
            return
        
        user_id = workflow.user_id
        
        # Create user output directory
        user_output_dir = os.path.join(USER_FILES_DIR, str(user_id), "outputs")
        os.makedirs(user_output_dir, exist_ok=True)
        
        # Destination path
        dest_filename = f"job_{job.id}_{filename}"
        dest_path = os.path.join(user_output_dir, dest_filename)
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        print(f"✅ Copied export file to user directory: {dest_path}")
            
    except Exception as e:
        print(f"⚠️  Error copying export file to user directory: {e}")
        import traceback
        traceback.print_exc()

@app.post("/users/{user_id}/files/upload")
async def upload_user_file(
    user_id: UUID,
    file: UploadFile = File(...),
    x_user_id: str = Header(..., alias="X-User-ID"),
    session: AsyncSession = Depends(get_session)
):
    """
    Upload a file for a user (input file).
    Files are stored in user_files/{user_id}/inputs/
    """
    # Multi-tenant isolation
    try:
        request_user_id = UUID(x_user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    if request_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied: Can only upload files for your own account")
    
    # Verify user exists
    user_statement = select(User).where(User.id == user_id)
    user_result = await session.exec(user_statement)
    user = user_result.first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Ensure user directories exist
    _, inputs_dir, _ = ensure_user_dirs(user_id)
    
    # Save file
    file_path = os.path.join(inputs_dir, file.filename)
    
    # Check if file already exists
    if os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"File '{file.filename}' already exists. Please delete it first or use a different name.")
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        file_size = os.path.getsize(file_path)
        
        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "file_path": f"/user_files/{user_id}/inputs/{file.filename}",
            "size_bytes": file_size,
            "uploaded_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/users/{user_id}/files")
async def list_user_files(
    user_id: UUID,
    x_user_id: str = Header(..., alias="X-User-ID"),
    session: AsyncSession = Depends(get_session)
):
    """
    List all files for a user (both inputs and outputs).
    Returns files organized by type.
    """
    # Multi-tenant isolation
    try:
        request_user_id = UUID(x_user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    if request_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied: Can only view your own files")
    
    # Verify user exists
    user_statement = select(User).where(User.id == user_id)
    user_result = await session.exec(user_statement)
    user = user_result.first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Ensure directories exist
    user_dir, inputs_dir, outputs_dir = ensure_user_dirs(user_id)
    
    # List input files
    input_files = []
    if os.path.exists(inputs_dir):
        for filename in os.listdir(inputs_dir):
            file_path = os.path.join(inputs_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                input_files.append({
                    "filename": filename,
                    "type": "input",
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "download_url": f"/users/{user_id}/files/download?filename={filename}&type=input"
                })
    
    # List output files
    output_files = []
    if os.path.exists(outputs_dir):
        for filename in os.listdir(outputs_dir):
            file_path = os.path.join(outputs_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                output_files.append({
                    "filename": filename,
                    "type": "output",
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "download_url": f"/users/{user_id}/files/download?filename={filename}&type=output"
                })
    
    return {
        "user_id": str(user_id),
        "username": user.username,
        "input_files": sorted(input_files, key=lambda x: x["created_at"], reverse=True),
        "output_files": sorted(output_files, key=lambda x: x["created_at"], reverse=True),
        "total_input_files": len(input_files),
        "total_output_files": len(output_files),
        "total_size_bytes": sum(f["size_bytes"] for f in input_files + output_files)
    }

@app.get("/users/{user_id}/files/download")
async def download_user_file(
    user_id: UUID,
    filename: str,
    file_type: str,  # "input" or "output"
    x_user_id: str = Header(..., alias="X-User-ID"),
    session: AsyncSession = Depends(get_session)
):
    """
    Download a user file.
    """
    # Multi-tenant isolation
    try:
        request_user_id = UUID(x_user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    if request_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied: Can only download your own files")
    
    # Verify user exists
    user_statement = select(User).where(User.id == user_id)
    user_result = await session.exec(user_statement)
    user = user_result.first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Determine file path
    if file_type == "input":
        file_path = os.path.join(get_user_files_dir(user_id), "inputs", filename)
    elif file_type == "output":
        file_path = os.path.join(get_user_files_dir(user_id), "outputs", filename)
    else:
        raise HTTPException(status_code=400, detail="file_type must be 'input' or 'output'")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security: prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Return file
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

@app.delete("/users/{user_id}/files")
async def delete_user_file(
    user_id: UUID,
    filename: str,
    file_type: str,  # "input" or "output"
    x_user_id: str = Header(..., alias="X-User-ID"),
    session: AsyncSession = Depends(get_session)
):
    """
    Delete a user file.
    """
    # Multi-tenant isolation
    try:
        request_user_id = UUID(x_user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    if request_user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied: Can only delete your own files")
    
    # Verify user exists
    user_statement = select(User).where(User.id == user_id)
    user_result = await session.exec(user_statement)
    user = user_result.first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Determine file path
    if file_type == "input":
        file_path = os.path.join(get_user_files_dir(user_id), "inputs", filename)
    elif file_type == "output":
        file_path = os.path.join(get_user_files_dir(user_id), "outputs", filename)
    else:
        raise HTTPException(status_code=400, detail="file_type must be 'input' or 'output'")
    
    # Security: prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        os.remove(file_path)
        return {
            "message": "File deleted successfully",
            "filename": filename,
            "file_type": file_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

# --- API 2: Submit a Workflow (most important step) ---
# User will send a list containing multiple Jobs

# Define data structure to receive frontend requests (Schema)
class JobCreate(BaseModel):
    name: str
    job_type: str
    branch_id: str
    parent_indices: List[int] = []  # A trick: reference which job in the list as parent node
    image_path: Optional[str] = None  # Custom image path

class WorkflowCreate(BaseModel):
    jobs: List[JobCreate]

@app.post("/workflows/")
async def create_workflow(
    workflow_data: WorkflowCreate,
    x_user_id: str = Header(..., alias="X-User-ID"),  # Require header
    session: AsyncSession = Depends(get_session)
):
    # Use ID from header
    try:
        user_id = UUID(x_user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    # Verify user exists
    user_statement = select(User).where(User.id == user_id)
    user_result = await session.exec(user_statement)
    user = user_result.first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found. Please create the user first.")
    
    # 1. Create Workflow
    new_workflow = Workflow(user_id=user_id)
    session.add(new_workflow)
    await session.commit()
    await session.refresh(new_workflow)
    
    # 2. Create Jobs and parse dependencies
    # Since Jobs haven't been saved to database yet and don't have IDs, we use list indices to temporarily store dependencies
    created_jobs = []
    
    for index, job_in in enumerate(workflow_data.jobs):
        # Debug: print image_path
        print(f"📁 Creating job with image_path: {job_in.image_path}")
        print(f"📁 Job data received: name={job_in.name}, job_type={job_in.job_type}, branch_id={job_in.branch_id}, image_path={job_in.image_path}")
        
        job = Job(
            workflow_id=new_workflow.id,
            name=job_in.name,
            job_type=job_in.job_type,
            branch_id=job_in.branch_id,
            status="PENDING",
            image_path=job_in.image_path  # Save custom image path
        )
        session.add(job)
        # Temporarily flush to get ID, but don't commit
        await session.flush()
        await session.refresh(job)
        
        print(f"📁 Job created with image_path: {job.image_path}")
        print(f"📁 Job ID: {job.id}")
        print(f"📁 Verifying image_path attribute exists: {hasattr(job, 'image_path')}")
        if hasattr(job, 'image_path'):
            print(f"📁 image_path value: '{job.image_path}'")
        else:
            print(f"⚠️  WARNING: image_path attribute not found on job object!")
        
        created_jobs.append(job)
    
    # 3. Second pass: Fill in parent_ids (now everyone has IDs)
    for index, job_in in enumerate(workflow_data.jobs):
        parent_uuids = []
        for parent_idx in job_in.parent_indices:
            if parent_idx < index:  # Prevent circular dependencies
                parent_uuids.append(str(created_jobs[parent_idx].id))
        
        # Update Job in database
        created_jobs[index].parent_ids_json = json.dumps(parent_uuids)
        session.add(created_jobs[index])
        
    await session.commit()
    
    # Update user activity time (for timeout detection)
    import redis.asyncio as redis
    from app.core.config import settings
    r = redis.from_url(settings.redis_url, decode_responses=True)
    import time
    await r.set(f"user_activity:{x_user_id}", str(time.time()), ex=600)  # 10 minutes expiration
    
    return {"workflow_id": new_workflow.id, "job_count": len(created_jobs)}

# --- New: View all Workflows ---
@app.get("/workflows/")
async def get_workflows(session: AsyncSession = Depends(get_session)):
    statement = select(Workflow)
    result = await session.exec(statement)
    workflows = result.all()
    return workflows

# --- Debug endpoint: View detailed information of all Jobs ---
@app.get("/debug/jobs")
async def get_all_jobs(session: AsyncSession = Depends(get_session)):
    # Directly query Job table to see most detailed task information
    statement = select(Job)
    result = await session.exec(statement)
    jobs = result.all()
    return jobs

# --- Get single Job result (for frontend) ---
@app.get("/jobs/{job_id}")
async def get_job_by_id(job_id: UUID, session: AsyncSession = Depends(get_session)):
    """Get detailed information of a single Job by Job ID"""
    statement = select(Job).where(Job.id == job_id)
    result = await session.exec(statement)
    job = result.first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job

# --- Export per-cell segmentation results ---
@app.get("/jobs/{job_id}/export")
async def export_cell_segmentation_results(
    job_id: UUID,
    format: str = "zarr",  # Options: "zarr", "csv"
    session: AsyncSession = Depends(get_session)
):
    """
    Export per-cell segmentation results for a completed cell_segmentation job.
    
    Returns polygon coordinates and metadata for all detected cells.
    Formats:
    - zarr: Zarr format with arrays for cell data (default)
    - csv: CSV format with cell_id, centroid_x, centroid_y, polygon_coords, area
    """
    statement = select(Job).where(Job.id == job_id)
    result = await session.exec(statement)
    job = result.first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.job_type != "cell_segmentation":
        raise HTTPException(
            status_code=400, 
            detail=f"Job type '{job.job_type}' does not support cell segmentation export. Only 'cell_segmentation' jobs can be exported."
        )
    
    if job.status != "SUCCEEDED":
        raise HTTPException(
            status_code=400,
            detail=f"Job status is '{job.status}'. Only SUCCEEDED jobs can be exported."
        )
    
    if not job.result_metadata:
        raise HTTPException(
            status_code=404,
            detail="Job result metadata not found. The job may not have completed successfully."
        )
    
    try:
        result_data = json.loads(job.result_metadata)
        cells = result_data.get("cells", [])
        
        # Create job directory in static folder
        STATIC_DIR = "static"
        job_dir = os.path.join(STATIC_DIR, str(job.id))
        os.makedirs(job_dir, exist_ok=True)
        
        if format.lower() == "csv":
            import csv
            import io
            
            # Create CSV content
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "cell_id", "centroid_x", "centroid_y", 
                "polygon_coords", "polygon_point_count", "area_pixels"
            ])
            
            # Write cell data
            for idx, cell in enumerate(cells):
                poly = cell.get("poly", [])
                # Calculate area from polygon (simplified: use bounding box area)
                if len(poly) >= 3:
                    x_coords = [p[0] for p in poly]
                    y_coords = [p[1] for p in poly]
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    area = width * height
                else:
                    area = 0
                
                # Format polygon as string: "[[x1,y1],[x2,y2],...]"
                poly_str = json.dumps(poly)
                
                writer.writerow([
                    idx + 1,  # cell_id (1-indexed)
                    cell.get("x", 0),  # centroid_x
                    cell.get("y", 0),  # centroid_y
                    poly_str,  # polygon_coords
                    len(poly),  # polygon_point_count
                    area  # area_pixels (approximate)
                ])
            
            csv_content = output.getvalue()
            output.close()
            
            # Save CSV file to job directory
            csv_filename = f"cell_segmentation_{job.id}.csv"
            csv_path = os.path.join(job_dir, csv_filename)
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(csv_content)
            print(f"✅ Saved CSV export to: {csv_path}")
            
            # Also copy to user output directory
            await copy_export_file_to_user_dir(session, job, csv_path, csv_filename)
            
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={csv_filename}",
                    "X-File-Path": f"/static/{job.id}/{csv_filename}"  # Include saved path in header
                }
            )
        
        else:  # Zarr format (default)
            import zarr
            from zarr.storage import ZipStore
            import tempfile
            
            # Create Zarr file using ZipStore (directly creates .zarr.zip file)
            zarr_filename = f"cell_segmentation_{job.id}.zarr.zip"
            zarr_path = os.path.join(job_dir, zarr_filename)
            
            # Create Zarr group using ZipStore
            store = ZipStore(zarr_path, mode='w')
            root = zarr.group(store=store)
            
            try:
                # Store metadata as attributes
                root.attrs['job_id'] = str(job.id)
                root.attrs['job_name'] = job.name
                root.attrs['job_type'] = job.job_type
                root.attrs['status'] = job.status
                root.attrs['image_path'] = job.image_path or ""
                root.attrs['total_cells'] = len(cells)
                root.attrs['pixel_size'] = float(result_data.get("pixel_size", 0.5))
                root.attrs['scale_factor'] = float(result_data.get("scale_factor", 1.0))
                root.attrs['detection_method'] = "InstanSeg"
                root.attrs['coordinate_system'] = "pixel_coordinates"
                root.attrs['units'] = "pixels"
                root.attrs['export_timestamp'] = datetime.utcnow().isoformat()
                root.attrs['format'] = "zarr"
                root.attrs['version'] = "1.0"
                
                if len(cells) > 0:
                    # Prepare arrays
                    cell_ids = []
                    centroids_x = []
                    centroids_y = []
                    polygon_counts = []
                    
                    # Store polygons - we'll use variable-length arrays
                    # Store as separate arrays for each cell (better for variable lengths)
                    polygons_group = root.create_group('polygons')
                    
                    for idx, cell in enumerate(cells):
                        cell_id = idx + 1
                        cell_ids.append(cell_id)
                        centroids_x.append(int(cell.get("x", 0)))
                        centroids_y.append(int(cell.get("y", 0)))
                        
                        poly = cell.get("poly", [])
                        polygon_counts.append(len(poly))
                        
                        # Store polygon for this cell as a 2D array (N x 2)
                        if len(poly) > 0:
                            poly_array = np.array(poly, dtype=np.int32)
                            polygons_group.create_dataset(
                                f'cell_{cell_id}',
                                data=poly_array,
                                shape=(len(poly), 2),
                                dtype=np.int32
                            )
                    
                    # Create arrays for cell data
                    root.create_dataset(
                        'cell_ids',
                        data=np.array(cell_ids, dtype=np.int32),
                        shape=(len(cell_ids),),
                        dtype=np.int32
                    )
                    
                    root.create_dataset(
                        'centroids',
                        data=np.array([centroids_x, centroids_y], dtype=np.int32).T,
                        shape=(len(cell_ids), 2),
                        dtype=np.int32
                    )
                    
                    root.create_dataset(
                        'polygon_point_counts',
                        data=np.array(polygon_counts, dtype=np.int32),
                        shape=(len(polygon_counts),),
                        dtype=np.int32
                    )
                
                # Close and flush the store
                store.close()
                
                print(f"✅ Saved Zarr export to: {zarr_path}")
                
                # Also copy to user output directory
                await copy_export_file_to_user_dir(session, job, zarr_path, zarr_filename)
                
                # Read zip file content for response
                with open(zarr_path, 'rb') as f:
                    zip_content = f.read()
                
                return Response(
                    content=zip_content,
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": f"attachment; filename={zarr_filename}",
                        "X-File-Path": f"/static/{job.id}/{zarr_filename}"
                    }
                )
            except Exception as e:
                # Clean up on error
                store.close()
                if os.path.exists(zarr_path):
                    os.remove(zarr_path)
                raise HTTPException(
                    status_code=500,
                    detail=f"Error creating Zarr export: {str(e)}"
                )
    
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse job result metadata: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error exporting results: {str(e)}"
        )

@app.get("/jobs/{job_id}/branch-jobs")
async def get_branch_jobs(job_id: UUID, session: AsyncSession = Depends(get_session)):
    """Get all jobs in the same branch (used to check if there are other types of tasks)"""
    statement = select(Job).where(Job.id == job_id)
    result = await session.exec(statement)
    current_job = result.first()
    
    if not current_job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Query all jobs in the same branch
    branch_statement = select(Job).where(
        Job.branch_id == current_job.branch_id,
        Job.workflow_id == current_job.workflow_id  # Only query jobs in the same workflow
    )
    branch_result = await session.exec(branch_statement)
    branch_jobs = branch_result.all()
    
    # Check if there are cell_segmentation and tissue_mask type jobs
    has_cell_segmentation = any(j.job_type == "cell_segmentation" and j.status == "SUCCEEDED" for j in branch_jobs)
    has_tissue_mask = any(j.job_type == "tissue_mask" and j.status == "SUCCEEDED" for j in branch_jobs)
    
    return {
        "branch_id": current_job.branch_id,
        "current_job_type": current_job.job_type,
        "has_cell_segmentation": has_cell_segmentation,
        "has_tissue_mask": has_tissue_mask,
        "branch_jobs": [
            {
                "id": str(j.id),
                "name": j.name,
                "job_type": j.job_type,
                "status": j.status
            }
            for j in branch_jobs
        ]
    }

# --- Cancel job endpoint ---
@app.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: UUID,
    x_user_id: str = Header(..., alias="X-User-ID"),  # Multi-tenant isolation
    session: AsyncSession = Depends(get_session)
):
    """Cancel a job(Can only cancel when in queue or running)"""
    try:
        user_id = UUID(x_user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    # Query job
    statement = select(Job).where(Job.id == job_id)
    result = await session.exec(statement)
    job = result.first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Multi-tenant isolation: check if job belongs to this user
    workflow_statement = select(Workflow).where(Workflow.id == job.workflow_id)
    workflow_result = await session.exec(workflow_statement)
    workflow = workflow_result.first()
    
    if not workflow or workflow.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied: Job does not belong to this user")
    
    # Can only cancel jobs with status PENDING, QUEUED, or RUNNING
    if job.status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel job with status: {job.status}"
        )
    
    # Update status to CANCELLED
    job.status = "CANCELLED"
    session.add(job)
    await session.commit()
    await session.refresh(job)
    
    return {
        "message": "Job cancelled successfully",
        "job_id": str(job.id),
        "status": job.status
    }

# --- Cancel all running or queued tasks ---
@app.post("/jobs/cancel-all")
async def cancel_all_jobs(
    x_user_id: str = Header(..., alias="X-User-ID"),  # Multi-tenant isolation
    session: AsyncSession = Depends(get_session)
):
    """
    Cancel all running or queued tasks(PENDING, QUEUED, RUNNING)
    Only cancel tasks belonging to current user(Multi-tenant isolation)
    """
    try:
        user_id = UUID(x_user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    # Query all workflows for this user
    workflow_statement = select(Workflow).where(Workflow.user_id == user_id)
    workflow_result = await session.exec(workflow_statement)
    user_workflows = workflow_result.all()
    
    if not user_workflows:
        return {
            "message": "No jobs found for this user",
            "cancelled_count": 0
        }
    
    workflow_ids = [wf.id for wf in user_workflows]
    
    # Query all cancellable tasks for this user(PENDING, QUEUED, RUNNING)
    from sqlalchemy import or_, and_
    job_statement = select(Job).where(
        and_(
            Job.workflow_id.in_(workflow_ids),
            or_(Job.status == "PENDING", Job.status == "QUEUED", Job.status == "RUNNING")
        )
    )
    job_result = await session.exec(job_statement)
    jobs_to_cancel = job_result.all()
    
    cancelled_count = 0
    for job in jobs_to_cancel:
        job.status = "CANCELLED"
        session.add(job)
        cancelled_count += 1
    
    await session.commit()
    
    return {
        "message": f"Successfully cancelled {cancelled_count} job(s)",
        "cancelled_count": cancelled_count
    }

# --- Debug endpoint: Cancel all tasks(regardless of user)---
@app.post("/debug/cancel-all-jobs")
async def cancel_all_jobs_debug(
    session: AsyncSession = Depends(get_session)
):
    """
    Cancel all running or queued tasks(PENDING, QUEUED, RUNNING)
    Debug endpoint:regardless of user,Cancel all tasks
    """
    from sqlalchemy import or_
    # ✅ Fix: Directly query PENDING, QUEUED, RUNNING status tasks(these statuses cannot be CANCELLED at the same time)
    job_statement = select(Job).where(
        or_(Job.status == "PENDING", Job.status == "QUEUED", Job.status == "RUNNING")
    )
    job_result = await session.exec(job_statement)
    jobs_to_cancel = job_result.all()
    
    cancelled_count = 0
    cancelled_jobs_info = []
    for job in jobs_to_cancel:
        # Check status again(prevent race conditions)
        if job.status in ["PENDING", "QUEUED", "RUNNING"]:
            old_status = job.status
            job.status = "CANCELLED"
            session.add(job)
            cancelled_count += 1
            cancelled_jobs_info.append({
                "id": str(job.id),
                "name": job.name,
                "branch": job.branch_id,
                "old_status": old_status
            })
    
    # ✅ Fix: Ensure all changes are committed to database
    if cancelled_count > 0:
        await session.commit()
        # ✅ Refresh all objects to ensure status update
        for job in jobs_to_cancel:
            await session.refresh(job)
        print(f"✅ Database committed,{cancelled_count} tasks status updated to CANCELLED")
        
        # ✅ Fix: After cancelling tasks, check and release active users without tasks
        import redis.asyncio as redis
        from app.core.config import settings
        from app.scheduler import release_user_if_done
        r = redis.from_url(settings.redis_url, decode_responses=True)
        
        # Get all active users
        active_users = await r.smembers("active_users")
        released_users = []
        for user_id in active_users:
            try:
                # Check if user still has pending tasks
                from sqlalchemy import or_
                from uuid import UUID
                statement = select(Job).join(Workflow).where(
                    and_(
                        Workflow.user_id == UUID(user_id),
                        Job.status.in_(["PENDING", "RUNNING", "QUEUED"])
                    )
                )
                result = await session.exec(statement)
                has_active_jobs = result.first() is not None
                
                if not has_active_jobs:
                    # User has no pending tasks, release user
                    await release_user_if_done(session, r, user_id)
                    released_users.append(user_id)
            except Exception as e:
                print(f"⚠️  Error checking user: {e}")
        
        if released_users:
            print(f"👋 Released active users without tasks")
        
        await r.aclose()
    else:
        print("ℹ️  No tasks to cancel")
    
    print(f"🛑 Cancelled tasks:")
    for info in cancelled_jobs_info[:10]:  # Only print first10
        print(f"   - {info['name']} ({info['branch']}): {info['old_status']} -> CANCELLED")
    if len(cancelled_jobs_info) > 10:
        print(f"   ... more tasks")
    
    return {
        "message": f"Successfully cancelled {cancelled_count} job(s)",
        "cancelled_count": cancelled_count,
        "cancelled_jobs": cancelled_jobs_info
    }

# --- Debug endpoint: Clear Redis active user list ---
@app.post("/debug/redis/clear")
async def clear_redis_users():
    """Clear Redis active users and waiting queue(for debugging)"""
    import redis.asyncio as redis
    from app.core.config import settings
    
    r = redis.from_url(settings.redis_url, decode_responses=True)
    
    # Get state before clearing
    active_before = await r.smembers("active_users")
    waiting_before = await r.lrange("waiting_users", 0, -1)
    
    # Clear
    await r.delete("active_users")
    await r.delete("waiting_users")
    
    return {
        "message": "Redis Clear",
        "cleared_active_users": list(active_before),
        "cleared_waiting_users": waiting_before
    }

# --- Debug endpoint: Clear all old tasks ---
@app.post("/debug/clear-all-jobs")
async def clear_all_jobs(
    clear_files: bool = True,  # Whether to also clear static files
    session: AsyncSession = Depends(get_session)
):
    """
    Clear all old tasks in database(Jobs,Workflows)
    Optional: also clear related static files(preview images, masks, etc.)
    
    Parameters:
    - clear_files: If True,Also delete all job-related files in directory
    """
    from app.models import Job, Workflow
    import os
    import shutil
    
    # 1. Count before deletion
    all_jobs_result = await session.exec(select(Job))
    all_jobs_list = all_jobs_result.all()
    job_count_before = len(all_jobs_list)
    
    all_workflows_result = await session.exec(select(Workflow))
    all_workflows_list = all_workflows_result.all()
    workflow_count_before = len(all_workflows_list)
    
    # 2. Get all job_id(for deleting files)
    job_ids = [str(job.id) for job in all_jobs_list]
    
    # 3. Delete all Jobs(First delete child tables,avoid foreign key constraint issues)
    for job in all_jobs_list:
        await session.delete(job)
    
    # 4. Delete all Workflows
    for workflow in all_workflows_list:
        await session.delete(workflow)
    
    # 5. Commit transaction
    await session.commit()
    
    # 6. Optional: clear static files
    deleted_files = []
    if clear_files:
        static_dir = "static"
        if os.path.exists(static_dir):
            for item in os.listdir(static_dir):
                item_path = os.path.join(static_dir, item)
                # Delete job-related folders(format:static/{job_id}/)
                if os.path.isdir(item_path) and item in job_ids:
                    shutil.rmtree(item_path)
                    deleted_files.append(item_path)
                # Delete job-related files(format:static/preview_{job_id}.jpg)
                elif os.path.isfile(item_path):
                    for job_id in job_ids:
                        if job_id in item:
                            os.remove(item_path)
                            deleted_files.append(item_path)
                            break
    
    return {
        "message": "All old tasks cleared",
        "deleted_jobs": job_count_before,
        "deleted_workflows": workflow_count_before,
        "deleted_files": len(deleted_files),
        "file_paths": deleted_files[:10] if len(deleted_files) > 10 else deleted_files  # Only return first10
    }

# --- Debug endpoint: Generate overlay visualization for a patch ---
@app.get("/debug/patch-overlay")
async def debug_patch_overlay(
    x: int = None,
    y: int = None,
    tile_size: int = 512,
    job_id: UUID = None,
    random: bool = False,
    session: AsyncSession = Depends(get_session)
):
    """
    Debug endpoint: Generate overlay visualization for a patch,Used for verifying segmentation results
    Parameters:
    - x, y: Top-left corner coordinates of patch(If random=True then ignore)
    - tile_size: patch size(default 512)
    - job_id: Job ID,to determine which image file to use
    - random: Whether to randomly select patch position(default False)
    """
    import numpy as np
    import openslide
    import cv2
    from PIL import Image
    from instanseg import InstanSeg
    import torch
    import random as rnd
    
    try:
        # 1. Get image path( job)
        image_path = DEFAULT_IMAGE  # default
        if job_id:
            try:
                statement = select(Job).where(Job.id == job_id)
                result = await session.execute(statement)
                job = result.scalar_one_or_none()
                if job and job.image_path:
                    image_path = job.image_path
                    print(f"🔍 [patch-overlay] Using image from job: {image_path}")
            except Exception as e:
                print(f"⚠️  [patch-overlay] Failed to get job image_path: {e}")
        
        print(f"🔍 [patch-overlay] Image Path: {image_path}")
        
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")
        
        slide = openslide.OpenSlide(image_path)
        w, h = slide.dimensions
        
        # 2.  patch 
        if random or x is None or y is None:
            # ,
            max_x = max(0, w - tile_size)
            max_y = max(0, h - tile_size)
            x = rnd.randint(0, max_x) if max_x > 0 else 0
            y = rnd.randint(0, max_y) if max_y > 0 else 0
            print(f"🎲  patch : ({x}, {y})")
        
        # 
        x = max(0, min(x, w - tile_size))
        y = max(0, min(y, h - tile_size))
        
        # 3. Read patch
        tile = slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB")
        tile_np = np.array(tile)
        
        # 🛡️ Background detection logic:Avoid generating"hallucinations"false detections
        gray = cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        std_dev = np.std(gray)
        
        is_background = mean_brightness > 220 or std_dev < 15
        
        if is_background:
            print(f"   🚫 Skipping background tile at ({x}, {y}): Brightness={mean_brightness:.1f}, StdDev={std_dev:.1f}")
            # 
            overlay_img = cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR)
            cv2.putText(overlay_img, "Background / Empty", (50, 256), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(overlay_img, f"Brightness: {mean_brightness:.1f}", (50, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(overlay_img, f"StdDev: {std_dev:.1f}", (50, 340), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            is_success, buffer = cv2.imencode(".jpg", overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not is_success:
                raise HTTPException(status_code=500, detail="Failed to encode image")
            io_buf = io.BytesIO(buffer.tobytes())
            return StreamingResponse(io_buf, media_type="image/jpeg")
        
        # 4.  InstanSeg 
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        instanseg = InstanSeg("brightfield_nuclei", image_reader="openslide", verbosity=0)
        
        # 5. (Background)
        labeled_output, _ = instanseg.eval_small_image(tile_np, pixel_size=0.25)
        
        #  numpy
        if torch.is_tensor(labeled_output):
            labeled_output = labeled_output.detach().cpu().numpy()
        if labeled_output.ndim > 2:
            labeled_output = labeled_output.squeeze()
        
        # ⚠️ :InstanSeg  label map( ID)
        #  0 = Background, > 0 =  ID
        # 
        if labeled_output.dtype not in [np.uint8, np.uint16, np.uint32, np.int32]:
            # (label map )
            labeled_output = labeled_output.astype(np.uint32)
        else:
            labeled_output = labeled_output.astype(np.uint32)
        
        # 6. Extract contours and statistics
        # InstanSeg  label map,
        # Count how many different cells(excluding background 0)
        unique_labels = np.unique(labeled_output)
        cell_labels = unique_labels[unique_labels > 0]  # excluding background 0
        num_labels = len(cell_labels)  # 
        labels_im = labeled_output.astype(np.uint32)  #  label map
        cell_count = num_labels  # 
        
        # 7. Generate overlay(original image + segmentation results + contours)
        # Create colored mask(Red indicates segmentation region)
        colored_mask = np.zeros_like(tile_np)
        colored_mask[:, :, 0] = (labels_im > 0).astype(np.uint8) * 255  # channel
        colored_mask[:, :, 1] = (labels_im > 0).astype(np.uint8) * 100  # Greenchannel
        colored_mask[:, :, 2] = 0  # channel
        
        # original image mask(original image 70%,mask 30%)
        blended = cv2.addWeighted(tile_np, 0.7, colored_mask, 0.3, 0)
        
        # 8. contours(yellow lines,clearer)
        contours_list = []
        for label_id in cell_labels:  #  label ID
            mask = (labels_im == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # contours(, 2)
                cv2.drawContours(blended, contours, -1, (0, 255, 255), 2)
                # (Green)
                M = cv2.moments(mask)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(blended, (cX, cY), 3, (0, 255, 0), -1)  # Green
                contours_list.append(contours)
        
        # 9. ()
        cv2.putText(blended, f"Patch: ({x}, {y})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(blended, f"Cells: {cell_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 10. save overlay  -  job id 
        if job_id:
            job_dir = os.path.join("static", job_id)
            os.makedirs(job_dir, exist_ok=True)
            overlay_filename = f"overlay_{x}_{y}_{tile_size}.jpg"
            overlay_path = os.path.join(job_dir, overlay_filename)
            overlay_url = f"/static/{job_id}/{overlay_filename}"
        else:
            os.makedirs("static", exist_ok=True)
            overlay_filename = f"overlay_debug_{x}_{y}_{tile_size}.jpg"
            overlay_path = os.path.join("static", overlay_filename)
            overlay_url = f"/static/{overlay_filename}"
        
        overlay_img = Image.fromarray(blended.astype(np.uint8))
        overlay_img.save(overlay_path, quality=95)
        
        return {
            "message": "Overlay generated successfully",
            "overlay_url": overlay_url,
            "patch_coords": {"x": x, "y": y, "tile_size": tile_size},
            "cells_detected": cell_count,
            "contours_count": len(contours_list),
            "image_size": f"{tile_size}x{tile_size}",
            "full_image_size": f"{w}x{h}",
            "view_url": f"http://localhost:8000{overlay_url}"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating overlay: {str(e)}")

# ---  Patch Overlay(15)---
# :generate_patch_overlays API -  verify_random_patch
# @app.get("/jobs/{job_id}/patch-overlays")
# async def generate_patch_overlays(job_id: str, count: int = 15):
    """
     job  patch overlay()
     patch  overlay  URL
    """
    import random as rnd
    import numpy as np
    import openslide
    import cv2
    from PIL import Image
    from instanseg import InstanSeg
    import torch
    from sqlalchemy.orm import sessionmaker
    from sqlmodel.ext.asyncio.session import AsyncSession
    
    results = []
    
    try:
        #  job 
        job_uuid = UUID(job_id)
        from app.db import engine
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with async_session() as session:
            statement = select(Job).where(Job.id == job_uuid)
            result = await session.exec(statement)
            job = result.first()
            if not job:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # 
        image_path = "/Users/yiling/Desktop/penn_proj/my-scheduler/data/CMU-1-Small-Region.svs"  # 
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")
        
        slide = openslide.OpenSlide(image_path)
        w, h = slide.dimensions
        tile_size = 512
        
        #  InstanSeg ()
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        instanseg = InstanSeg("brightfield_nuclei", image_reader="openslide", verbosity=0)
        
        #  job 
        job_dir = os.path.join("static", job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        #  overlay ( preview )
        if os.path.exists(job_dir):
            for filename in os.listdir(job_dir):
                if filename.startswith("overlay_") and filename.endswith(".jpg"):
                    old_file_path = os.path.join(job_dir, filename)
                    try:
                        os.remove(old_file_path)
                        print(f"🗑️  Deleted old overlay: {filename}")
                    except Exception as e:
                        print(f"⚠️  Failed to delete {filename}: {e}")
        
        #  patch
        for i in range(count):
            max_x = max(0, w - tile_size)
            max_y = max(0, h - tile_size)
            x = rnd.randint(0, max_x) if max_x > 0 else 0
            y = rnd.randint(0, max_y) if max_y > 0 else 0
            
            try:
                # Read patch
                tile = slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB")
                tile_np = np.array(tile)
                
                # 
                labeled_output, _ = instanseg.eval_small_image(tile_np, pixel_size=0.25)
                
                #  numpy
                if torch.is_tensor(labeled_output):
                    labeled_output = labeled_output.detach().cpu().numpy()
                if labeled_output.ndim > 2:
                    labeled_output = labeled_output.squeeze()
                labeled_output = labeled_output.astype(np.uint32)
                
                # Extractcontours
                unique_labels = np.unique(labeled_output)
                cell_labels = unique_labels[unique_labels > 0]
                labels_im = labeled_output
                cell_count = len(cell_labels)
                
                # Generate overlay
                colored_mask = np.zeros_like(tile_np)
                colored_mask[:, :, 0] = (labels_im > 0).astype(np.uint8) * 255
                colored_mask[:, :, 1] = (labels_im > 0).astype(np.uint8) * 100
                colored_mask[:, :, 2] = 0
                blended = cv2.addWeighted(tile_np, 0.7, colored_mask, 0.3, 0)
                
                # contours
                contours_list = []
                cell_info_list = []
                
                # 
                for label_id in cell_labels:
                    mask = (labels_im == label_id).astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        M = cv2.moments(mask)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            # 
                            x_coords = np.where(mask > 0)[1]
                            y_coords = np.where(mask > 0)[0]
                            if len(x_coords) > 0 and len(y_coords) > 0:
                                x_min, x_max = int(x_coords.min()), int(x_coords.max())
                                y_min, y_max = int(y_coords.min()), int(y_coords.max())
                                cell_info_list.append({
                                    "label_id": int(label_id),
                                    "centroid": (cX, cY),
                                    "bbox": (x_min, y_min, x_max, y_max),
                                    "contours": contours,
                                    "mask": mask
                                })
                
                # ()
                highlighted_cell = None
                if len(cell_info_list) > 0:
                    highlighted_cell = rnd.choice(cell_info_list)
                
                # contours
                for cell_info in cell_info_list:
                    is_highlighted = (highlighted_cell and cell_info["label_id"] == highlighted_cell["label_id"])
                    
                    # contours()
                    if cell_info["contours"]:
                        contour_color = (0, 0, 255) if is_highlighted else (0, 255, 255)  # ,
                        contour_thickness = 3 if is_highlighted else 2
                        cv2.drawContours(blended, cell_info["contours"], -1, contour_color, contour_thickness)
                        
                        # 
                        center_color = (0, 0, 255) if is_highlighted else (0, 255, 0)  # ,Green
                        center_radius = 4 if is_highlighted else 3
                        cv2.circle(blended, cell_info["centroid"], center_radius, center_color, -1)
                        
                        # ,
                        if is_highlighted:
                            x_min, y_min, x_max, y_max = cell_info["bbox"]
                            # ()
                            cv2.rectangle(blended, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), (0, 0, 255), 3)
                            # 
                            label_text = f"Cell #{cell_info['label_id']}"
                            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(blended, (x_min - 2, y_min - text_height - 8), 
                                        (x_min + text_width + 4, y_min - 2), (0, 0, 255), -1)
                            cv2.putText(blended, label_text, (x_min, y_min - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    contours_list.append(cell_info["contours"])
                
                # 
                cv2.putText(blended, f"Patch: ({x}, {y})", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(blended, f"Cells: {cell_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # ,
                if highlighted_cell:
                    highlight_text = f"Highlighted: Cell #{highlighted_cell['label_id']}"
                    cv2.putText(blended, highlight_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # save
                overlay_filename = f"overlay_{x}_{y}_{tile_size}.jpg"
                overlay_path = os.path.join(job_dir, overlay_filename)
                overlay_url = f"/static/{job_id}/{overlay_filename}"
                
                overlay_img = Image.fromarray(blended.astype(np.uint8))
                overlay_img.save(overlay_path, quality=95)
                
                # ()
                highlighted_cell_global = None
                segmentation_quality = None
                patch_size_info = None
                
                if highlighted_cell:
                    local_cx, local_cy = highlighted_cell["centroid"]
                    global_cx = x + local_cx
                    global_cy = y + local_cy
                    
                    # ( contour )
                    main_contour = max(highlighted_cell["contours"], key=cv2.contourArea)
                    contour_area = cv2.contourArea(main_contour)
                    contour_perimeter = cv2.arcLength(main_contour, True)
                    
                    # (circularity):4π*area/perimeter^2,1
                    if contour_perimeter > 0:
                        circularity = (4 * np.pi * contour_area) / (contour_perimeter ** 2)
                    else:
                        circularity = 0
                    
                    # (usually)
                    #  50-5000 
                    area_score = min(1.0, max(0.0, (contour_area - 20) / 2000))
                    
                    # (0-1)
                    quality_score = (circularity * 0.4 + area_score * 0.6)
                    
                    #  patch size
                    x_min, y_min, x_max, y_max = highlighted_cell["bbox"]
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    bbox_area = bbox_width * bbox_height
                    
                    #  patch size(bbox  patch )
                    patch_area = tile_size * tile_size
                    bbox_ratio = bbox_area / patch_area
                    
                    # 
                    size_status = "normal"
                    if bbox_ratio > 0.5:
                        size_status = "too_large"  # bbox  patch  50%
                    elif bbox_ratio < 0.01:
                        size_status = "too_small"  # bbox  patch  1%
                    
                    highlighted_cell_global = {
                        "label_id": highlighted_cell["label_id"],
                        "local_centroid": highlighted_cell["centroid"],
                        "global_centroid": (global_cx, global_cy),
                        "bbox": highlighted_cell["bbox"],
                        "area": float(contour_area),
                        "perimeter": float(contour_perimeter),
                        "circularity": float(circularity)
                    }
                    
                    segmentation_quality = {
                        "score": float(quality_score),
                        "circularity": float(circularity),
                        "area_score": float(area_score),
                        "area": float(contour_area),
                        "bbox_ratio": float(bbox_ratio)
                    }
                    
                    patch_size_info = {
                        "bbox_width": int(bbox_width),
                        "bbox_height": int(bbox_height),
                        "bbox_area": int(bbox_area),
                        "patch_size": tile_size,
                        "patch_area": patch_area,
                        "ratio": float(bbox_ratio),
                        "status": size_status
                    }
                
                results.append({
                    "overlay_url": overlay_url,
                    "patch_coords": {"x": x, "y": y, "tile_size": tile_size},
                    "cells_detected": cell_count,
                    "contours_count": len(contours_list),
                    "highlighted_cell": highlighted_cell_global,
                    "segmentation_quality": segmentation_quality,
                    "patch_size_info": patch_size_info
                })
            except Exception as e:
                print(f"⚠️ Failed to generate patch {i+1}: {e}")
                continue
        
        return {
            "job_id": job_id,
            "total_generated": len(results),
            "patches": results
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating patch overlays: {str(e)}")

# --- Progress tracking endpoint:  Job  ---
@app.get("/jobs/{job_id}/progress")
async def get_job_progress(job_id: UUID, session: AsyncSession = Depends(get_session)):
    """ Job """
    statement = select(Job).where(Job.id == job_id)
    result = await session.exec(statement)
    job = result.first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # 
    progress_percent = 0.0
    if job.total_tiles > 0:
        progress_percent = (job.processed_tiles / job.total_tiles) * 100.0
    
    return {
        "job_id": str(job.id),
        "name": job.name,
        "status": job.status,
        "progress_percent": round(progress_percent, 2),
        "processed_tiles": job.processed_tiles,
        "total_tiles": job.total_tiles,
        "attempt": job.attempt,
        "max_retries": job.max_retries,
        "error": job.error
    }

# --- Progress tracking endpoint:  Workflow  ---
@app.get("/workflows/{workflow_id}/progress")
async def get_workflow_progress(
    workflow_id: UUID,
    x_user_id: str = Header(..., alias="X-User-ID"),  # Multi-tenant isolation
    session: AsyncSession = Depends(get_session)
):
    """Get overall Workflow progress information(Multi-tenant isolation)"""
    from sqlalchemy.orm import selectinload
    from sqlalchemy import and_
    
    try:
        user_id = UUID(x_user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    # Only query workflows for this user(Multi-tenant isolation)
    statement = select(Workflow).where(
        and_(
            Workflow.id == workflow_id,
            Workflow.user_id == user_id
        )
    ).options(selectinload(Workflow.jobs))
    result = await session.exec(statement)
    workflow = result.first()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found or access denied")
    
    jobs = workflow.jobs
    total_jobs = len(jobs)
    
    if total_jobs == 0:
        return {
            "workflow_id": str(workflow.id),
            "status": workflow.status,
            "total_jobs": 0,
            "completed_jobs": 0,
            "progress_percent": 0.0,
            "jobs": []
        }
    
    # Count tasks by status
    completed_jobs = sum(1 for j in jobs if j.status == "SUCCEEDED")
    failed_jobs = sum(1 for j in jobs if j.status == "FAILED")
    running_jobs = sum(1 for j in jobs if j.status in ["RUNNING", "QUEUED"])
    pending_jobs = sum(1 for j in jobs if j.status == "PENDING")
    
    # Calculate overall completion percentage
    progress_percent = (completed_jobs / total_jobs) * 100.0
    
    # Calculate total tile progress for all tasks
    total_tiles_all = sum(j.total_tiles for j in jobs if j.total_tiles > 0)
    processed_tiles_all = sum(j.processed_tiles for j in jobs)
    tiles_progress_percent = 0.0
    if total_tiles_all > 0:
        tiles_progress_percent = (processed_tiles_all / total_tiles_all) * 100.0
    
    # 
    jobs_info = []
    for job in jobs:
        job_progress = 0.0
        if job.total_tiles > 0:
            job_progress = (job.processed_tiles / job.total_tiles) * 100.0
        
        jobs_info.append({
            "job_id": str(job.id),
            "name": job.name,
            "status": job.status,
            "branch_id": job.branch_id,
            "progress_percent": round(job_progress, 2),
            "processed_tiles": job.processed_tiles,
            "total_tiles": job.total_tiles
        })
    
    return {
        "workflow_id": str(workflow.id),
        "status": workflow.status,
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "running_jobs": running_jobs,
        "pending_jobs": pending_jobs,
        "progress_percent": round(progress_percent, 2),  # 
        "tiles_progress_percent": round(tiles_progress_percent, 2),  #  tile 
        "total_tiles": total_tiles_all,
        "processed_tiles": processed_tiles_all,
        "jobs": jobs_info
    }

# --- Progress tracking endpoint:  Workflows (Multi-tenant isolation)---
@app.get("/workflows/my-progress")
async def get_my_workflows_progress(
    x_user_id: str = Header(..., alias="X-User-ID"),
    session: AsyncSession = Depends(get_session)
):
    """Get progress information for all Workflows of current user(Multi-tenant isolation)"""
    from sqlalchemy.orm import selectinload
    
    try:
        user_id = UUID(x_user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    # Only query workflows for this users
    statement = select(Workflow).where(Workflow.user_id == user_id).options(selectinload(Workflow.jobs))
    result = await session.exec(statement)
    workflows = result.all()
    
    workflows_info = []
    for wf in workflows:
        jobs = wf.jobs
        total_jobs = len(jobs)
        completed_jobs = sum(1 for j in jobs if j.status == "SUCCEEDED")
        progress_percent = (completed_jobs / total_jobs * 100.0) if total_jobs > 0 else 0.0
        
        workflows_info.append({
            "workflow_id": str(wf.id),
            "status": wf.status,
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "progress_percent": round(progress_percent, 2),
            "created_at": wf.created_at.isoformat()
        })
    
    return {
        "user_id": str(user_id),
        "total_workflows": len(workflows_info),
        "workflows": workflows_info
    }

# --- DZI XML endpoint:Tell frontend how many levels the image has and its dimensions ---
@app.get("/dzi/{job_id}.dzi")
async def get_dzi_metadata(job_id: str, session: AsyncSession = Depends(get_session)):
    """
     DZI XML ,,
    In actual project, should query database by job_id to find corresponding .svs file path
    """
    # Find corresponding image path by job_id
    try:
        job_uuid = UUID(job_id)
        statement = select(Job).where(Job.id == job_uuid)
        result = await session.exec(statement)
        job = result.first()
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")
    
    # Use same image file as worker()
    # Prefer custom image path saved in job, otherwise use default
    if job.image_path:
        image_path = job.image_path
    else:
        image_path = "/Users/yiling/Desktop/penn_proj/my-scheduler/data/CMU-1-Small-Region.svs"  # default
    
    dz = get_slide_generator(image_path)
    if not dz:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Return standard DZI XML format
    content = dz.get_dzi("jpeg")
    return Response(content=content, media_type="application/xml")

# --- Tile endpoint: (level, col, row), JPG ---
@app.get("/dzi/{job_id}_files/{level}/{col}_{row}.jpeg")
async def get_dzi_tile(job_id: str, level: int, col: int, row: int, session: AsyncSession = Depends(get_session)):
    """
    Return tile image for specified level and position
    Frontend will frequently request this endpoint when zooming
    """
    # Use same image file as worker()
    # Prefer custom image path saved in job, otherwise use default
    try:
        job_uuid = UUID(job_id)
        statement = select(Job).where(Job.id == job_uuid)
        result = await session.exec(statement)
        job = result.first()
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        if job.image_path:
            image_path = job.image_path
        else:
            image_path = "/Users/yiling/Desktop/penn_proj/my-scheduler/data/CMU-1-Small-Region.svs"  # default
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")
    except Exception:
        # Fallback to default if job lookup fails
        image_path = "/Users/yiling/Desktop/penn_proj/my-scheduler/data/CMU-1-Small-Region.svs"
    
    dz = get_slide_generator(image_path)
    
    if not dz:
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        # Get small tile for specified level and position
        tile = dz.get_tile(level, (col, row))
        
        # Convert to JPEG binary stream
        buf = BytesIO()
        tile.save(buf, format="JPEG", quality=90)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except ValueError:
        raise HTTPException(status_code=404, detail="Tile coords out of bounds")

# --- Get all patches list ---
@app.get("/debug/all-patches")
async def get_all_patches(job_id: UUID, session: AsyncSession = Depends(get_session)):
    """
    Get list of all possible patch coordinates(based on tile_size and stride)
    Return patch coordinate list for frontend pagination
    
    ✅ Only return patches with tissue(filtered by tissue_mask job)
    
    Parameters:
    - job_id: Job ID, image_path
    """
    # 🔧 Get job image_path from database
    try:
        statement = select(Job).where(Job.id == job_id)
        result = await session.execute(statement)
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        #  job.image_path,default
        image_path = job.image_path or DEFAULT_IMAGE
        print(f"🔍 [get_all_patches] Job ID: {job_id}, Image Path: {image_path}")
        
        # ✅ Find tissue_mask job in the same branch
        tissue_tile_coords = None
        try:
            print(f"🔍 [get_all_patches] Searching tissue_mask job...")
            print(f"   Workflow ID: {job.workflow_id}")
            print(f"   Branch ID: {job.branch_id}")
            print(f"   Current Job Type: {job.job_type}")
            
            #  workflow  branch  job
            branch_jobs_statement = select(Job).where(
                Job.workflow_id == job.workflow_id,
                Job.branch_id == job.branch_id
            )
            branch_jobs_result = await session.execute(branch_jobs_statement)
            all_branch_jobs = branch_jobs_result.scalars().all()
            
            print(f"   Found {len(all_branch_jobs)} jobs in this branch:")
            for bj in all_branch_jobs:
                print(f"     - Job {bj.id}: type={bj.job_type}, status={bj.status}")
            
            #  tissue_mask  job
            tissue_mask_jobs = [
                bj for bj in all_branch_jobs 
                if bj.job_type == "tissue_mask" and bj.status == "SUCCEEDED"
            ]
            
            print(f"   Found {len(tissue_mask_jobs)} SUCCEEDED tissue_mask jobs")
            
            # If tissue_mask job found, extract tissue_tile_coords
            if tissue_mask_jobs:
                tissue_mask_job = tissue_mask_jobs[0]  #  tissue_mask job
                print(f"   Using tissue_mask job: {tissue_mask_job.id}")
                
                if tissue_mask_job.result_metadata:
                    result_data = json.loads(tissue_mask_job.result_metadata)
                    tissue_tile_coords = result_data.get("tissue_tile_coords", [])
                    print(f"✅ Found tissue_mask job with {len(tissue_tile_coords)} tissue tiles")
                else:
                    print(f"⚠️ tissue_mask job has no result_metadata")
            else:
                print(f"⚠️ No SUCCEEDED tissue_mask job found in this branch")
                print(f"   Will return all patches (frontend will filter by cells_detected)")
        except Exception as e:
            import traceback
            print(f"⚠️ Error fetching tissue_mask job: {e}")
            traceback.print_exc()
            print(f"   Will return all patches (frontend will filter by cells_detected)")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching job: {str(e)}")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
    
    try:
        slide = openslide.OpenSlide(image_path)
        w, h = slide.dimensions
        
        tile_size = 512
        overlap = 128
        stride = tile_size - overlap  # 384
        
        # ✅  tissue_tile_coords, patches
        if tissue_tile_coords:
            # Convert tissue_tile_coords to set for fast lookup
            tissue_coords_set = {(coord["x"], coord["y"]) for coord in tissue_tile_coords}
            print(f"🔍 Filtering patches using {len(tissue_coords_set)} tissue coordinates")
            
            patches = []
            for coord in tissue_tile_coords:
                x, y = coord["x"], coord["y"]
                # 
                if x + tile_size <= w and y + tile_size <= h:
                    patches.append({
                        "x": x,
                        "y": y,
                        "tile_size": tile_size
                    })
            
            print(f"✅ Returning {len(patches)} tissue patches (filtered from {len(tissue_tile_coords)} tissue tiles)")
        else:
            # No tissue_mask job, return all coordinates
            print(f"⚠️ No tissue_mask job found, returning all patches")
            patches = []
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    if x + tile_size <= w and y + tile_size <= h:
                        patches.append({
                            "x": x,
                            "y": y,
                            "tile_size": tile_size
                        })
        
        return {
            "total_patches": len(patches),
            "patches": patches,
            "image_path": image_path,  # 
            "filtered_by_tissue_mask": tissue_tile_coords is not None  # 
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting patches: {str(e)}")

# --- Real-time verification of patch segmentation quality ---
@app.get("/debug/verify-patch")
async def verify_random_patch(
    job_id: UUID,
    x: int = None,
    y: int = None,
    show_all: bool = False,  # If True, show all detection results(without filtering)
    return_json: bool = False,  # If True, return JSON(containing image URL and cell information)
    pixel_size: float = 0.5,  # ⚠️ Pixel size(micrometers per pixel),has great impact on detection results！Suggested test values:0.25, 0.3, 0.5, 0.7, 1.0
    session: AsyncSession = Depends(get_session)
):
    """
    Process specified patch, run model, return image with contours.
    Used for verification:1. if model works 2. if contours are aligned 3. if edge handling is correct
    
    Parameters:
    - job_id: Job ID, image_path
    - x, y: Specify patch coordinates()
    - show_all: If True, InstanSeg (without filtering)
    - return_json: If True, return JSON(containing image URL and cell information)
    """
    if debug_instanseg is None:
        raise HTTPException(status_code=500, detail="Debug model not loaded")
    
    # 🔧 Get job image_path from database
    try:
        statement = select(Job).where(Job.id == job_id)
        result = await session.execute(statement)
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        #  job.image_path,default
        image_path = job.image_path or DEFAULT_IMAGE
        print(f"🔍 [verify_patch] Job ID: {job_id}, Image Path: {image_path}, Coord: ({x}, {y})")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching job: {str(e)}")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
    
    try:
        slide = openslide.OpenSlide(image_path)
        w, h = slide.dimensions
        
        tile_size = 512
        
        # 1.  patch (x, y )
        if x is None or y is None:
            raise HTTPException(status_code=400, detail="x and y coordinates are required")
        
        # Validate coordinate validity
        if x < 0 or y < 0 or x + tile_size > w or y + tile_size > h:
            raise HTTPException(status_code=400, detail=f"Invalid coordinates: ({x}, {y})")
        
        # Read patch
        tile = slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB")
        tile_np = np.array(tile)
        
        # 🛡️ Background detection logic:Avoid generating"hallucinations"false detections
        gray = cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        std_dev = np.std(gray)
        
        # Condition to determine as background:too bright OR too flat(single color)
        # - mean_brightness > 220: whiteBackground
        # - std_dev < 15: ,Background
        is_background = mean_brightness > 220 or std_dev < 15
        
        if is_background:
            print(f"   🚫 Skipping background tile at ({x}, {y}): Brightness={mean_brightness:.1f}, StdDev={std_dev:.1f}")
            
            # 
            overlay_img = cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR)
            
            # Background
            cv2.putText(overlay_img, "Background / Empty", (50, 256), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(overlay_img, f"Brightness: {mean_brightness:.1f}", (50, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(overlay_img, f"StdDev: {std_dev:.1f}", (50, 340), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if return_json:
                #  JSON,Background
                patch_id = str(uuid.uuid4())[:8]
                patch_filename = f"background_{patch_id}.jpg"
                patch_dir = "static/quality_checks"
                os.makedirs(patch_dir, exist_ok=True)
                patch_path = os.path.join(patch_dir, patch_filename)
                patch_url = f"/static/quality_checks/{patch_filename}"
                
                cv2.imwrite(patch_path, overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                return {
                    "image_url": patch_url,
                    "patch_coords": {"x": x, "y": y, "tile_size": tile_size},
                    "pixel_size": pixel_size,
                    "is_background": True,
                    "background_analysis": {
                        "mean_brightness": float(mean_brightness),
                        "std_dev": float(std_dev),
                        "reason": "Too bright or too flat (no tissue)"
                    },
                    "cells_detected": 0,
                    "all_detected": 0,
                    "displayed": 0,
                    "filtered": {"by_area": 0, "by_edge": 0, "by_shape": 0},
                    "highlighted_cell": None,
                    "note": "This patch was skipped because it contains only background (no tissue). InstanSeg would produce false positives here."
                }
            else:
                # Return image stream
                is_success, buffer = cv2.imencode(".jpg", overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not is_success:
                    raise HTTPException(status_code=500, detail="Failed to encode image")
                io_buf = io.BytesIO(buffer.tobytes())
                return StreamingResponse(io_buf, media_type="image/jpeg")
        
        # 2. Run model(Real-time Inference)- Background
        # ⚠️ pixel_size has great impact on detection results！
        # (),:
        # - (): pixel_size (0.25, 0.3)
        # - (): pixel_size (0.5, 0.7, 1.0)
        # - : (0.25, 0.3)
        # - false detections: (0.7, 1.0)
        labeled_output, _ = debug_instanseg.eval_small_image(tile_np, pixel_size=pixel_size)
        
        #  numpy( tensor)
        if torch.is_tensor(labeled_output):
            labeled_output = labeled_output.detach().cpu().numpy()
        if labeled_output.ndim > 2:
            labeled_output = labeled_output.squeeze()
        labeled_output = labeled_output.astype(np.uint32)
        
        # 3.  OpenCV original imagecontours
        overlay_img = cv2.cvtColor(tile_np, cv2.COLOR_RGB2BGR)
        
        #  ID
        unique_labels = np.unique(labeled_output)
        all_detected_count = len(unique_labels[unique_labels > 0])
        
        # 
        filtered_by_area = 0
        filtered_by_edge = 0
        filtered_by_shape = 0
        displayed_count = 0
        
        # Parameters( worker.py )
        CROP_MARGIN = 64  # OVERLAP // 2
        MIN_AREA = 30 if not show_all else 5  #  show_all=True,
        
        # contours
        for label_id in unique_labels:
            if label_id == 0:
                continue
            
            #  Mask
            mask = (labeled_output == label_id).astype(np.uint8)
            
            # 
            M = cv2.moments(mask)
            if M["m00"] == 0:
                continue
            
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # ( show_all=False)
            if not show_all:
                if (cX < CROP_MARGIN or cX >= tile_size - CROP_MARGIN or 
                    cY < CROP_MARGIN or cY >= tile_size - CROP_MARGIN):
                    filtered_by_edge += 1
                    # 
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cv2.drawContours(overlay_img, contours, -1, (0, 0, 255), 1, cv2.LINE_4)
                    continue
            
            # contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            main_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(main_contour)
            
            # ( show_all=False)
            if not show_all:
                if contour_area < MIN_AREA:
                    filtered_by_area += 1
                    # 
                    cv2.drawContours(overlay_img, contours, -1, (0, 165, 255), 1, cv2.LINE_4)
                    continue
            
            # 
            epsilon = 0.005 * cv2.arcLength(main_contour, True)
            approx = cv2.approxPolyDP(main_contour, epsilon, True)
            if len(approx) < 3:
                filtered_by_shape += 1
                # 
                cv2.drawContours(overlay_img, contours, -1, (255, 0, 255), 1, cv2.LINE_4)
                continue
            
            # :
            displayed_count += 1
            cv2.drawContours(overlay_img, contours, -1, (0, 255, 255), 2)
            cv2.circle(overlay_img, (cX, cY), 3, (0, 255, 0), -1)
        
        # ,
        
        # 4. 
        if return_json:
            # save
            patch_id = str(uuid.uuid4())[:8]
            patch_filename = f"quality_check_{patch_id}.jpg"
            patch_dir = "static/quality_checks"
            os.makedirs(patch_dir, exist_ok=True)
            patch_path = os.path.join(patch_dir, patch_filename)
            patch_url = f"/static/quality_checks/{patch_filename}"
            
            cv2.imwrite(patch_path, overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # ()- 
            highlighted_cell_info = None
            if not show_all:
                #  highlighted cell
                max_area = 0
                best_cell = None
                for label_id in unique_labels:
                    if label_id == 0:
                        continue
                    mask = (labeled_output == label_id).astype(np.uint8)
                    M = cv2.moments(mask)
                    if M["m00"] == 0:
                        continue
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # ()
                    if (cX < CROP_MARGIN or cX >= tile_size - CROP_MARGIN or 
                        cY < CROP_MARGIN or cY >= tile_size - CROP_MARGIN):
                        continue
                    
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    main_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(main_contour)
                    if area < MIN_AREA:
                        continue
                    
                    epsilon = 0.005 * cv2.arcLength(main_contour, True)
                    approx = cv2.approxPolyDP(main_contour, epsilon, True)
                    if len(approx) < 3:
                        continue
                    
                    # 
                    if area > max_area:
                        max_area = area
                        best_cell = {
                            "label_id": int(label_id),
                            "local_centroid": (cX, cY),
                            "global_centroid": (x + cX, y + cY),
                            "area": float(area)
                        }
                
                if best_cell:
                    highlighted_cell_info = {
                        "cell_id": best_cell["label_id"],
                        "position": best_cell["global_centroid"]
                    }
            
            return {
                "image_url": patch_url,
                "patch_coords": {"x": x, "y": y, "tile_size": tile_size},
                "pixel_size": pixel_size,  # ⚠️  pixel_size,
                "cells_detected": displayed_count if not show_all else all_detected_count,
                "all_detected": all_detected_count,
                "displayed": displayed_count,
                "filtered": {
                    "by_area": filtered_by_area,
                    "by_edge": filtered_by_edge,
                    "by_shape": filtered_by_shape
                },
                "highlighted_cell": highlighted_cell_info,
                "detection_quality": {
                    "detection_rate": f"{(displayed_count / all_detected_count * 100):.1f}%" if all_detected_count > 0 else "0%",
                    "filter_rate": f"{((filtered_by_area + filtered_by_edge + filtered_by_shape) / all_detected_count * 100):.1f}%" if all_detected_count > 0 else "0%",
                    "note": " detection_rate (), pixel_size (0.25, 0.3).false detections, pixel_size (0.7, 1.0)"
                },
                # 
                "patch_info": {
                    "x": x,
                    "y": y,
                    "tile_size": tile_size
                }
            }
        else:
            # Return image stream
            is_success, buffer = cv2.imencode(".jpg", overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not is_success:
                raise HTTPException(status_code=500, detail="Failed to encode image")
            
            io_buf = io.BytesIO(buffer.tobytes())
            return StreamingResponse(io_buf, media_type="image/jpeg")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating patch: {str(e)}")

# --- Debug API: Generate Tissue Mask Overlay ---
@app.get("/debug/tissue-mask")
async def get_tissue_mask(job_id: UUID = None, session: AsyncSession = Depends(get_session)):
    """
    Generate a low-resolution tissue mask preview image (PNG format).
    Green = Tissue
    Transparent = Background
    
    Parameters:
    - job_id: Optional Job ID to determine which image file to use
    """
    from PIL import Image
    
    # Get image path from job or use default
    image_path = DEFAULT_IMAGE
    if job_id:
        try:
            statement = select(Job).where(Job.id == job_id)
            result = await session.execute(statement)
            job = result.scalar_one_or_none()
            if job and job.image_path:
                image_path = job.image_path
                print(f"🔍 [tissue-mask] Using image from job: {image_path}")
        except Exception as e:
            print(f"⚠️  [tissue-mask] Failed to get job image_path: {e}")
    
    print(f"🔍 [tissue-mask] Image Path: {image_path}")
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        slide = openslide.OpenSlide(image_path)
        
        # 1. Get thumbnail (2048px width is enough for preview, OpenSlide maintains aspect ratio)
        thumbnail = slide.get_thumbnail((2048, 2048))
        thumb_np = np.array(thumbnail)
        thumb_h, thumb_w = thumb_np.shape[:2]
        
        # 2. ✅ Improvement:Use HSV saturation threshold(more robust)
        # H&E stained slide:Backgroundwhite(,),Tissue(regardless of depth,saturation is higher)
        hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
        
        # Extract S (Saturation, ) channel
        s_channel = hsv[:, :, 1]
        
        # ✅ :Use Otsu automatic threshold(Background)
        _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # ✅ :If Otsu effect is not good,can use fixed threshold()
        # usually > 15-20 Tissue
        # _, mask = cv2.threshold(s_channel, 20, 255, cv2.THRESH_BINARY)
        
        # ✅ Improvement:Add morphological operations(fill holes,remove noise)
        # Define a kernel (Kernel),size determines filling capability,(7,7) suitable for thumbnails
        kernel = np.ones((7, 7), np.uint8)
        
        # 1. Close operation (Close): dilate then erode -> fill small internal black holes,connect disconnected regions
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 2. Open operation (Open): erode then dilate -> Backgroundin(isolated small white points)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # ✅  mask 
        if mask.shape != (thumb_h, thumb_w):
            mask_resized = np.zeros((thumb_h, thumb_w), dtype=np.uint8)
            copy_h = min(mask.shape[0], thumb_h)
            copy_w = min(mask.shape[1], thumb_w)
            mask_resized[:copy_h, :copy_w] = mask[:copy_h, :copy_w]
            mask = mask_resized
        
        # 3. Create visualization layer (RGBA)
        # ✅ : mask instead of gray(gray )
        # Tissue region (mask=255) -> Alpha = 100 (semi-transparent)
        # Background region (mask=0) -> Alpha = 0 (fully transparent)
        alpha_channel = np.zeros_like(mask)
        alpha_channel[mask == 255] = 100  # Transparency 0-255
        
        # Combine channels: [R, G, B, A]
        rgba = np.dstack((
            np.zeros_like(mask),      # R
            np.full_like(mask, 255),  # G
            np.zeros_like(mask),      # B
            alpha_channel             # A (0 or 100)
        ))
        
        # Convert back to PIL Image for PNG saving
        final_img = Image.fromarray(rgba, 'RGBA')
        
        # 4. Return PNG stream
        buf = io.BytesIO()
        final_img.save(buf, format="PNG")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating tissue mask: {str(e)}")

