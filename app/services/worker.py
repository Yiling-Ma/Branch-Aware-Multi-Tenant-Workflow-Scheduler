import asyncio
import json
import os
import sys
import numpy as np
import torch
import openslide
import cv2
from PIL import Image
from sqlmodel import select
from app.db.session import engine
from app.models.sql_models import Job, Workflow
from sqlalchemy.orm import sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from instanseg import InstanSeg
from app.core.config import settings
import shutil
from app.services.metrics import metrics
import time

# --- Configuration ---
DATA_DIR = "./data"
DEFAULT_IMAGE = "/Users/yiling/Desktop/penn_proj/my-scheduler/data/CMU-1-Small-Region.svs"  # test using small image
STATIC_DIR = "static"
USER_FILES_DIR = "user_files"

def ensure_job_static_dir(job_id):
    job_dir = os.path.join(STATIC_DIR, str(job_id))
    os.makedirs(job_dir, exist_ok=True)
    return job_dir

def save_patch_overlay(job_id, x, y, tile_np, kept_cells, edge_contours):
    """Render patch overlay image with kept cells (yellow/green) and edge-filtered (red)"""
    if tile_np is None or (not kept_cells and not edge_contours):
        return None
    overlay_img = cv2.cvtColor(tile_np.copy(), cv2.COLOR_RGB2BGR)
    
    # Draw edge-filtered contours in red
    for contour in edge_contours:
        cv2.drawContours(overlay_img, [contour], -1, (0, 0, 255), 2)
    
    # Draw kept cells in yellow + green center
    for cell in kept_cells:
        contour = cell["contour"]
        center = cell["center"]
        cv2.drawContours(overlay_img, [contour], -1, (0, 255, 255), 2)
        cv2.circle(overlay_img, center, 3, (0, 255, 0), -1)
    
    patch_dir = os.path.join(ensure_job_static_dir(job_id), "patches")
    os.makedirs(patch_dir, exist_ok=True)
    filename = f"patch_{x}_{y}.jpg"
    path = os.path.join(patch_dir, filename)
    cv2.imwrite(path, overlay_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return f"/static/{job_id}/patches/{filename}"

async def copy_job_outputs_to_user_dir(session: AsyncSession, job: Job):
    """
    Copy job output files to user's output directory.
    This ensures files persist even if static/{job_id}/ is cleaned up.
    """
    try:
        # Get workflow and user
        workflow_stmt = select(Workflow).where(Workflow.id == job.workflow_id)
        workflow_result = await session.exec(workflow_stmt)
        workflow = workflow_result.first()
        
        if not workflow:
            print(f"Workflow not found for job {job.id}, skipping file copy")
            return
        
        user_id = workflow.user_id
        
        # Create user output directory
        user_output_dir = os.path.join(USER_FILES_DIR, str(user_id), "outputs")
        os.makedirs(user_output_dir, exist_ok=True)
        
        # Source job directory
        job_dir = os.path.join(STATIC_DIR, str(job.id))
        
        if not os.path.exists(job_dir):
            print(f"Job directory not found: {job_dir}, skipping file copy")
            return
        
        # Copy all files from job directory to user output directory
        copied_files = []
        for filename in os.listdir(job_dir):
            source_path = os.path.join(job_dir, filename)
            if os.path.isfile(source_path):
                # Create unique filename: job_{job_id}_{original_filename}
                dest_filename = f"job_{job.id}_{filename}"
                dest_path = os.path.join(user_output_dir, dest_filename)
                
                # Copy file
                shutil.copy2(source_path, dest_path)
                copied_files.append(dest_filename)
        
        if copied_files:
            print(f"Copied {len(copied_files)} output file(s) to user directory: {user_output_dir}")
        else:
            print(f"No files found in job directory: {job_dir}")
            
    except Exception as e:
        print(f"Error copying job outputs to user directory: {e}")
        import traceback
        traceback.print_exc()
PIXEL_SIZE = getattr(settings, 'INSTANSEG_PIXEL_SIZE', 0.5)  # default 0.5 micrometers per pixel,can be configured in .env

# --- Performance configuration ---
BATCH_SIZE = 4       # Batch size
TILE_SIZE = 512      # Tile size
OVERLAP = 128        # Overlap
STRIDE = TILE_SIZE - OVERLAP 

# --- Helper function: Safely update Job ---
async def safe_update_job(session, job_id, updates):
    """Safely update job,avoid session expiration issues"""
    try:
        # Re-query to get latest job object
        statement = select(Job).where(Job.id == job_id)
        result = await session.exec(statement)
        job = result.first()
        if not job:
            print(f"Job {job_id} not found in database")
            return None
        
        # Apply updates
        for key, value in updates.items():
            setattr(job, key, value)
        
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job
    except Exception as e:
        await session.rollback()
        print(f"Failed to update job {job_id}: {e}")
        return None

# --- Helper function: Check if Job is cancelled ---
async def check_job_cancelled(session, job_id):
    """Check if job is cancelled"""
    try:
        statement = select(Job).where(Job.id == job_id)
        result = await session.exec(statement)
        job = result.first()
        if job and job.status == "CANCELLED":
            return True
        return False
    except Exception as e:
        print(f"Failed to check job status: {e}")
        return False

# --- Helper function:NMS-based Tile Overlap Blending ---
def calculate_polygon_area(poly):
    """Calculate polygon area using shoelace formula (fast, no shapely dependency)"""
    if not poly or len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def get_polygon_bbox(poly):
    """Return bounding box (min_x, min_y, max_x, max_y)"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)

def bbox_iou(box1, area1, box2, area2):
    """Fast IoU approximation using bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area

def merge_cells_with_nms(cells, iou_threshold=0.3):
    """
    Use Non-Maximum Suppression (NMS) merge overlapping cell detections
    Optimized with Spatial Grid Indexing + Bounding Box IoU (fast, no shapely)
    
    Args:
        cells: cell list,each cell contains {"x", "y", "poly"}
        iou_threshold: IoU threshold,default 0.3
    
    Returns:
        list: merged cell list
    """
    if len(cells) == 0:
        return []
    
    print(f"   Start NMS merge (Fast BBox)... (original cell count: {len(cells)})")
    start_time = time.time()
    
    # 1. Calculate area for each cell(for sorting and retention decision)
    cells_with_area = []
    for cell in cells:
        area = calculate_polygon_area(cell["poly"])
        bbox = get_polygon_bbox(cell["poly"])
        cells_with_area.append({
            **cell,
            "area": area,
            "bbox": bbox,
            "_suppressed": False,  # Mark if suppressed
            "_index": 0 # Placeholder
        })
    
    # 2. Sort by area descending(larger area prioritized)
    cells_with_area.sort(key=lambda c: c["area"], reverse=True)
    
    # Update index after sort
    for i, cell in enumerate(cells_with_area):
        cell["_index"] = i
    
    # 3. Spatial Grid Indexing
    # Grid size should be larger than typical cell size + overlap margin
    # 256 is safe for most cells
    GRID_SIZE = 256
    grid = {}
    
    def get_grid_key(c):
        return (int(c["x"] // GRID_SIZE), int(c["y"] // GRID_SIZE))
    
    # Populate grid
    for cell in cells_with_area:
        gk = get_grid_key(cell)
        if gk not in grid:
            grid[gk] = []
        grid[gk].append(cell)
        
    # 4. NMS algorithm with Grid Search
    final_cells = []
    check_count = 0
    
    for cell in cells_with_area:
        if cell["_suppressed"]:
            continue
        
        # Keep current cell
        final_cells.append({
            "x": cell["x"],
            "y": cell["y"],
            "poly": cell["poly"]
        })
        
        # Check neighbors in same and adjacent grid cells (3x3 area)
        gx, gy = get_grid_key(cell)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                ng_key = (gx + dx, gy + dy)
                if ng_key in grid:
                    # Check all cells in this grid bucket
                    for neighbor in grid[ng_key]:
                        # Skip if already suppressed or processed (larger area ones came before)
                        if neighbor["_suppressed"] or neighbor["_index"] <= cell["_index"]:
                            continue
                        
                        check_count += 1
                        
                        # Quick distance check (Euclidean)
                        # Optimization: use squared distance to avoid sqrt
                        dist_sq = (cell["x"] - neighbor["x"])**2 + (cell["y"] - neighbor["y"])**2
                        # TILE_SIZE/2 = 256, squared is 65536
                        if dist_sq > 65536: # (TILE_SIZE/2)^2
                            continue
                        
                        # Calculate IoU using bounding boxes (fast)
                        iou = bbox_iou(cell["bbox"], cell["area"], neighbor["bbox"], neighbor["area"])
                        
                        if iou > iou_threshold:
                            neighbor["_suppressed"] = True
    
    elapsed = time.time() - start_time
    removed_count = len(cells) - len(final_cells)
    print(f"   NMS complete in {elapsed:.2f}s: Removed {removed_count} duplicates")
    print(f"      Comparison checks: {check_count} (vs O(N^2) approx {len(cells)**2})")
    print(f"      Final cell count: {len(final_cells)}")
    
    return final_cells

# --- Task type 1: Cell segmentation ---
async def process_cell_segmentation(job, slide, w, h, coordinates, instanseg, session):
    """Process cell segmentation task"""
    # Record start time for latency calculation
    process_cell_segmentation._job_start_time = time.time()
    process_cell_segmentation._job_created_at = job.created_at.timestamp() if job.created_at else time.time()
    
    # Check if cancelled before starting processing
    job_id = job.id  # Save job_id,avoid object expiration
    if await check_job_cancelled(session, job_id):
        print(f"Task cancelled, stop processing")
        await safe_update_job(session, job_id, {"status": "CANCELLED"})
        return
    
    batch_images = []
    batch_coords = []
    batch_stats = []
    all_detected_cells = []
    CROP_MARGIN = OVERLAP // 2
    processed_count = 0  # local counter 
    
    skipped_background = 0  # Count number of skipped background tiles
    quality_checks = []  # Store per-patch quality/result metadata
    
    for i, (x, y) in enumerate(coordinates):
        # A. Read tile
        tile = slide.read_region((x, y), 0, (TILE_SIZE, TILE_SIZE)).convert("RGB")
        tile_np = np.array(tile)
        
        # Background detection logic: Avoid generating hallucination false detections
        # Use stricter detection:grayscale mean + standard deviation
        gray = cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        std_dev = np.std(gray)
        
        # Condition to determine as background:too bright OR too flat(single color)
        # - mean_brightness > 220: mostly white background
        # - std_dev < 15: very little color variation,indicates flat background region
        # This avoids misdetecting background noise as cells
        is_background = mean_brightness > 220 or std_dev < 15
        
        if is_background:
            skipped_background += 1
            processed_count += 1  # still count in progress,but skip processing
            quality_checks.append({
                "x": int(x),
                "y": int(y),
                "tile_size": TILE_SIZE,
                "is_background": True,
                "cells_detected": 0,
                "all_detected": 0,
                "filtered_by_edge": 0,
                "filtered_by_area": 0,
                "filtered_by_shape": 0,
                "mean_brightness": float(mean_brightness),
                "std_dev": float(std_dev)
            })
            # Update progress every tiles(including skipped)
            if (i + 1) % 10 == 0 or i == len(coordinates) - 1:
                updated_job = await safe_update_job(session, job_id, {
                    "processed_tiles": processed_count
                })
                if updated_job:
                    print(f"      Progress: {updated_job.processed_tiles}/{updated_job.total_tiles} (skipped background: {skipped_background}, Brightness={mean_brightness:.1f}, StdDev={std_dev:.1f})")
            continue  # Skip this background tile,do not run model
        
        batch_images.append(tile_np)
        batch_coords.append((x, y))
        batch_stats.append({
            "mean_brightness": float(mean_brightness),
            "std_dev": float(std_dev)
        })
        
        # B. Batch inference
        if len(batch_images) >= BATCH_SIZE or i == len(coordinates) - 1:
            # Batch inference
            labeled_outputs = []
            for tile_img in batch_images:
                # pixel_size has great impact on detection results！
                # If missing detections occur(especially in dense regions),try adjusting this value:
                # - High image resolution(cells look small):decrease pixel_size (0.25, 0.3)
                # - Low image resolution(cells look large):increase pixel_size (0.5, 0.7, 1.0)
                labeled_output, _ = instanseg.eval_small_image(tile_img, pixel_size=PIXEL_SIZE)
                if torch.is_tensor(labeled_output):
                    labeled_output = labeled_output.detach().cpu().numpy()
                if labeled_output.ndim > 2:
                    labeled_output = labeled_output.squeeze()
                labeled_outputs.append(labeled_output)
            
            # C. Post-processing
            for j, label_map in enumerate(labeled_outputs):
                offset_x, offset_y = batch_coords[j]
                tile_img = batch_images[j]
                label_map = label_map.astype(np.int32)
                unique_labels = np.unique(label_map)
                kept_cells = []
                edge_filtered_contours = []
                patch_stats = {
                    "x": int(offset_x),
                    "y": int(offset_y),
                    "tile_size": TILE_SIZE,
                    "is_background": False,  # update later
                    "cells_detected": 0,
                    "all_detected": int(max(len(unique_labels) - 1, 0)),
                    "filtered_by_edge": 0,
                    "filtered_by_area": 0,
                    "filtered_by_shape": 0,
                    "mean_brightness": batch_stats[j]["mean_brightness"],
                    "std_dev": batch_stats[j]["std_dev"],
                    "highlighted_cell": None,
                    "image_url": None
                }
                best_highlight = {"area": 0, "info": None}
                
                for label_id in unique_labels:
                    if label_id == 0: continue
                    
                    mask = (label_map == label_id).astype(np.uint8)
                    M = cv2.moments(mask)
                    if M["m00"] == 0: continue
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Find contours first (needed for edge filtering check)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours: continue
                    
                    main_contour = max(contours, key=cv2.contourArea)
                    
                    # Deduplication logic:Only keep center region
                    # Check if center is in edge region (after we have main_contour)
                    if (cX < CROP_MARGIN or cX >= TILE_SIZE - CROP_MARGIN or 
                        cY < CROP_MARGIN or cY >= TILE_SIZE - CROP_MARGIN):
                        patch_stats["filtered_by_edge"] += 1
                        edge_filtered_contours.append(main_contour)
                        continue
                    # Increase area threshold,Filter out too small noise(avoid over-segmentation)
                    # 10pixels too small,Change to pixels,Ensure only detecting real cells
                    if cv2.contourArea(main_contour) < 30:
                        patch_stats["filtered_by_area"] += 1
                        continue
                    
                    epsilon = 0.005 * cv2.arcLength(main_contour, True)
                    approx = cv2.approxPolyDP(main_contour, epsilon, True)
                    if len(approx) < 3:
                        patch_stats["filtered_by_shape"] += 1
                        continue
                    
                    poly_points = []
                    for point in approx:
                        px, py = point[0]
                        poly_points.append([int(px + offset_x), int(py + offset_y)])
                    
                    all_detected_cells.append({
                        "x": int(cX + offset_x),
                        "y": int(cY + offset_y),
                        "poly": poly_points 
                    })
                    patch_stats["cells_detected"] += 1
                    area = cv2.contourArea(main_contour)
                    if area > best_highlight["area"]:
                        best_highlight = {
                            "area": area,
                            "info": {
                                "cell_id": int(label_id),
                                "position": [int(cX + offset_x), int(cY + offset_y)]
                            }
                        }
                    kept_cells.append({
                        "contour": main_contour,
                        "center": (cX, cY)
                    })

                patch_stats["is_background"] = patch_stats["cells_detected"] == 0
                if best_highlight["info"]:
                    patch_stats["highlighted_cell"] = best_highlight["info"]
                overlay_url = save_patch_overlay(job.id, offset_x, offset_y, tile_img, kept_cells, edge_filtered_contours)
                patch_stats["image_url"] = overlay_url
                quality_checks.append(patch_stats)
            
            # D. Update status
            processed_count += len(batch_images)
            updated_job = await safe_update_job(session, job_id, {
                "processed_tiles": processed_count
            })
            if updated_job:
                print(f"      Progress: {updated_job.processed_tiles}/{updated_job.total_tiles}")
            else:
                print(f"      Unable to update progress (local count: {processed_count})")
            
            # Check if task is cancelled(Check after processing each batch)
            if await check_job_cancelled(session, job_id):
                print(f"Task cancelled, stop processing")
                await safe_update_job(session, job_id, {
                    "status": "CANCELLED",
                    "processed_tiles": processed_count
                })
                return  # Immediately exit processing function
            
            batch_images = []
            batch_coords = []
            batch_stats = []
    
    # New: NMS-based Tile Overlap Blending
    print(f"   Start Tile Overlap Blending...")
    print(f"      Original detected cell count: {len(all_detected_cells)}")
    
    # Apply NMS to merge duplicate detections in overlap regions
    # IoU threshold is a reasonable default value:
    # - Too low( 0.1):may miss some real duplicates
    # - Too high( 0.5+):may merge adjacent different cells
    all_detected_cells = merge_cells_with_nms(all_detected_cells, iou_threshold=0.3)
    
    # Generate preview image(without label,only show original image)
    print("   Generating preview image(without annotations)...")
    thumbnail = slide.get_thumbnail((2000, 2000))
    scale_factor = w / thumbnail.width
    thumb_cv = cv2.cvtColor(np.array(thumbnail), cv2.COLOR_RGB2BGR)
    
    # Note:Preview image does not draw any annotations(dots,contours)
    # (Load Slide) SVG overlay 
    
    # Save sparse sampling information(for frontend to display statistics)
    sampling_info = None
    if len(all_detected_cells) > 5000:
        step = len(all_detected_cells) // 5000 + 1
        sampling_info = {
            "total_cells": len(all_detected_cells),
            "displayed_cells": len(all_detected_cells[::step]),
            "sampling_step": step
        }
        print(f"   Large number of cells ({len(all_detected_cells)}),Main view will display with sparse sampling")
    
    # 
    job_dir = os.path.join(STATIC_DIR, str(job.id))
    os.makedirs(job_dir, exist_ok=True)
    preview_filename = f"preview_{job.id}.jpg"
    preview_path = os.path.join(job_dir, preview_filename)
    cv2.imwrite(preview_path, thumb_cv)
    
    # Save results
    result_data = {
        "total_cells": len(all_detected_cells),
        "preview_url": f"/static/{job.id}/{preview_filename}",
        "scale_factor": scale_factor,
        "pixel_size": PIXEL_SIZE,  # Record used pixel_size,for debugging and reproduction
        "cells": all_detected_cells,
        "sampling_info": sampling_info,  # Save sparse sampling information(if any)
        "quality_checks": quality_checks
    }
    
    # Save final results
    final_job = await safe_update_job(session, job_id, {
        "result_metadata": json.dumps(result_data),
        "status": "SUCCEEDED",
        "processed_tiles": processed_count  # Ensure final progress is correct
    })
    
    if final_job:
        # Record metrics: job latency and duration
        job_end_time = time.time()
        if hasattr(process_cell_segmentation, '_job_start_time'):
            latency = job_end_time - process_cell_segmentation._job_start_time
            metrics.record_job_latency("cell_segmentation", "SUCCEEDED", latency)
        if hasattr(process_cell_segmentation, '_job_created_at'):
            duration = job_end_time - process_cell_segmentation._job_created_at
            metrics.record_job_duration("cell_segmentation", "SUCCEEDED", duration)
        metrics.increment_job_counter("cell_segmentation", "SUCCEEDED")
        
        print(f"Cell segmentation complete！detected {len(all_detected_cells)} cells")
        print(f"   Pixel Size used: {PIXEL_SIZE} micrometers per pixel")
        print(f"   Performance optimization: skipped {skipped_background} background tiles (saved {skipped_background * 100 / len(coordinates):.1f}% processing time)")
        print(f"   Job ID: {final_job.id}")
        print(f"   Tip: If detection count is not ideal,can modify .env  INSTANSEG_PIXEL_SIZE and restart worker")
        
        # Copy output files to user directory
        await copy_job_outputs_to_user_dir(session, final_job)
    else:
        print(f"Task completed but unable to update database status")

# --- Task type 2:  ---
async def process_tissue_mask(job, slide, w, h, coordinates, session):
    """Process tissue mask generation task - Generate binary mask to skip background tiles"""
    # Record start time for latency calculation
    process_tissue_mask._job_start_time = time.time()
    process_tissue_mask._job_created_at = job.created_at.timestamp() if job.created_at else time.time()
    
    # Check if cancelled before starting processing
    job_id = job.id  # Save job_id,avoid object expiration
    if await check_job_cancelled(session, job_id):
        print(f"Task cancelled, stop processing")
        await safe_update_job(session, job_id, {"status": "CANCELLED"})
        return
    
    # Create global mask(full resolution)
    global_mask = np.zeros((h, w), dtype=np.uint8)
    tissue_tiles = []  # Record tissue tile coordinates
    
    skipped_background = 0  # Count number of skipped background tiles
    
    for i, (x, y) in enumerate(coordinates):
        # Fix:Calculate actual effective area(avoid out of bounds)
        end_x = min(x + TILE_SIZE, w)
        end_y = min(y + TILE_SIZE, h)
        actual_h = end_y - y
        actual_w = end_x - x
        
        #  tile
        tile = slide.read_region((x, y), 0, (TILE_SIZE, TILE_SIZE)).convert("RGB")
        tile_np = np.array(tile)
        
        # Fix:Ensure tile_np size matches actual area
        tile_h, tile_w = tile_np.shape[:2]
        if tile_h != actual_h or tile_w != actual_w:
            tile_np = tile_np[:actual_h, :actual_w]
        
        # Performance optimization:Quickly skip pure background tiles(before complex detection)
        # If image is mostly white (RGB > 220),skip it,greatly improve processing speed
        if tile_np.mean() > 220:
            skipped_background += 1
            # Update progress(including skipped tiles)
            if i % 10 == 0 or i == len(coordinates) - 1:
                updated_job = await safe_update_job(session, job_id, {
                    "processed_tiles": i + 1
                })
                if updated_job:
                    print(f"      Progress: {updated_job.processed_tiles}/{updated_job.total_tiles} (skipped background: {skipped_background},  tile: {len(tissue_tiles)})")
            continue  # Skip this background tile
        
        # Improvement:Use HSV saturation threshold(more robust)
        # H&E :(,),(,)
        tile_hsv = cv2.cvtColor(tile_np, cv2.COLOR_RGB2HSV)
        
        #  S (Saturation, ) 
        saturation = tile_hsv[:, :, 1]
        
        # Use Otsu automatic threshold()
        #  tile,Otsu 
        _, tissue_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tissue_mask = tissue_mask > 0  # 
        
        # Note: Otsu ,
        #  > 15-20 
        # tissue_mask = saturation > 20
        
        # Optional:(,)
        value = tile_hsv[:, :, 2]
        # ()
        not_too_dark = value > 15
        
        # :,
        combined_mask = tissue_mask & not_too_dark
        
        # Morphological operations(,)
        #  tile ,(3x3)
        kernel = np.ones((3, 3), np.uint8)
        combined_mask_uint8 = (combined_mask.astype(np.uint8)) * 255
        
        # :
        combined_mask_uint8 = cv2.morphologyEx(combined_mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # :
        combined_mask_uint8 = cv2.morphologyEx(combined_mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # 
        combined_mask = combined_mask_uint8 > 0
        
        # Fix:
        tissue_ratio = np.sum(combined_mask) / (actual_h * actual_w)
        
        #  > 10%, tile 
        if tissue_ratio > 0.1:
            # Fix:, tile
            # combined_mask , uint8
            tissue_mask_uint8 = (combined_mask.astype(np.uint8)) * 255
            
            # Fix: mask 
            if tissue_mask_uint8.shape != (actual_h, actual_w):
                tissue_mask_uint8 = tissue_mask_uint8[:actual_h, :actual_w]
            
            #  255, 0
            global_mask[y:end_y, x:end_x] = np.maximum(
                global_mask[y:end_y, x:end_x], 
                tissue_mask_uint8
            )
            
            tissue_tiles.append({
                "x": x,
                "y": y,
                "tissue_ratio": float(tissue_ratio)
            })
        
        # Update progress
        if i % 10 == 0 or i == len(coordinates) - 1:
            updated_job = await safe_update_job(session, job_id, {
                "processed_tiles": i + 1
            })
            if updated_job:
                print(f"      Progress: {updated_job.processed_tiles}/{updated_job.total_tiles} (skipped background: {skipped_background},  tile: {len(tissue_tiles)})")
            else:
                print(f"      Unable to update progress")
            
            # Check if task is cancelled( 10  tile )
            if await check_job_cancelled(session, job_id):
                print(f"Task cancelled, stop processing")
                await safe_update_job(session, job_id, {
                    "status": "CANCELLED",
                    "processed_tiles": i + 1
                })
                return  # Immediately exit processing function
    
    # Generate preview image()
    print("   Generating preview image...")
    thumbnail = slide.get_thumbnail((2000, 2000))
    thumb_np = np.array(thumbnail)
    thumb_h, thumb_w = thumb_np.shape[:2]
    
    # ()
    scale_factor_w = w / thumb_w
    scale_factor_h = h / thumb_h
    
    thumb_cv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2BGR)
    
    # Fix: resize 
    # ,
    mask_h, mask_w = global_mask.shape
    print(f"   Mask size: ({mask_h}, {mask_w})")
    print(f"   Thumbnail size: ({thumb_h}, {thumb_w})")
    
    #  resize (Note:cv2.resize  (width, height))
    thumb_mask = cv2.resize(global_mask, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
    
    # Check:
    if thumb_mask.shape != (thumb_h, thumb_w):
        print(f"   Warning:thumb_mask size mismatch, Fix...")
        print(f"      : ({thumb_h}, {thumb_w}), : {thumb_mask.shape}")
        #  thumb_cv 
        thumb_mask_resized = np.zeros((thumb_h, thumb_w), dtype=np.uint8)
        #  thumb_mask 
        copy_h = min(thumb_mask.shape[0], thumb_h)
        copy_w = min(thumb_mask.shape[1], thumb_w)
        thumb_mask_resized[:copy_h, :copy_w] = thumb_mask[:copy_h, :copy_w]
        thumb_mask = thumb_mask_resized
        print(f"   Fix: {thumb_mask.shape}")
    
    # Check:
    if thumb_mask.shape != (thumb_h, thumb_w):
        raise ValueError(f"thumb_mask  ({thumb_h}, {thumb_w}), {thumb_mask.shape}")
    if thumb_cv.shape[:2] != (thumb_h, thumb_w):
        raise ValueError(f"thumb_cv  ({thumb_h}, {thumb_w}, 3), {thumb_cv.shape}")
    
    print(f"   Verified: thumb_mask={thumb_mask.shape}, thumb_cv={thumb_cv.shape}")
    
    # Improvement: preview  mask()
    #  mask:
    # 1.  Load Slide  "Tissue Mask" 
    # 2.  "View full mask image"  mask 
    # 3. Preview only show original image,
    
    # Optional: mask ,
    # ( mask)
    
    # ( mask)
    job_dir = os.path.join(STATIC_DIR, str(job.id))
    os.makedirs(job_dir, exist_ok=True)
    preview_filename = f"preview_{job.id}.jpg"
    preview_path = os.path.join(job_dir, preview_filename)
    cv2.imwrite(preview_path, thumb_cv)
    
    # Optional: mask ()
    preview_with_mask_filename = f"preview_with_mask_{job.id}.jpg"
    preview_with_mask_path = os.path.join(job_dir, preview_with_mask_filename)
    
    # ()
    thumb_cv_with_mask = thumb_cv.copy()
    mask_colored = np.zeros_like(thumb_cv_with_mask)
    mask_colored[:, :, 1] = thumb_mask  # 
    thumb_cv_with_mask = cv2.addWeighted(thumb_cv_with_mask, 0.7, mask_colored, 0.3, 0)
    cv2.imwrite(preview_with_mask_path, thumb_cv_with_mask)
    
    # 
    mask_filename = f"tissue_mask_{job.id}.png"
    mask_path = os.path.join(job_dir, mask_filename)
    cv2.imwrite(mask_path, global_mask)
    
    # Save results
    result_data = {
        "job_type": "tissue_mask",
        "total_tiles": len(coordinates),
        "tissue_tiles": len(tissue_tiles),
        "background_tiles": len(coordinates) - len(tissue_tiles),
        "tissue_ratio": len(tissue_tiles) / len(coordinates) if coordinates else 0,
        "preview_url": f"/static/{job.id}/{preview_filename}",  # ( mask)
        "preview_with_mask_url": f"/static/{job.id}/{preview_with_mask_filename}",  #  mask 
        "mask_url": f"/static/{job.id}/{mask_filename}",  #  mask 
        "tissue_tile_coords": tissue_tiles,  #  tile 
        "image_size": {"width": w, "height": h}
    }
    
    # Save final results
    final_job = await safe_update_job(session, job_id, {
        "result_metadata": json.dumps(result_data),
        "status": "SUCCEEDED",
        "processed_tiles": len(coordinates)  # Ensure final progress is correct
    })
    
    if final_job:
        print(f"Tissue mask generation complete！")
        print(f"   Total tiles: {len(coordinates)}")
        print(f"   Tissue tiles: {len(tissue_tiles)}")
        print(f"   Skipped background: {skipped_background} (saved {skipped_background * 100 / len(coordinates):.1f}% processing time)")
        print(f"   ⚪  tile: {len(coordinates) - len(tissue_tiles) - skipped_background}")
        print(f"   Job ID: {final_job.id}")
        
        # Copy output files to user directory
        await copy_job_outputs_to_user_dir(session, final_job)
    else:
        print(f"Task completed but unable to update database status")

async def run_worker(worker_id=1):
    print(f"Worker-{worker_id} started (Batch Size: {BATCH_SIZE})...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading InstanSeg model (Device: {device})...")
    print(f"   Pixel Size: {PIXEL_SIZE} micrometers per pixel")
    print(f"   Tip:If missing detections occur(especially in dense regions),decrease pixel_size")
    print(f"      ,increase pixel_size")
    print(f"       .env  INSTANSEG_PIXEL_SIZE ")
    
    # 1.  (, settings )
    try:
        #  standard nucleus model
        instanseg = InstanSeg("brightfield_nuclei", verbosity=0, device=device)
        print("InstanSeg brightfield_nuclei model loaded")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    while True:
        try:
            async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with async_session() as session:
                
                # 2.  (Use SELECT FOR UPDATE SKIP LOCKED )
                from sqlalchemy import select as sa_select
                statement = sa_select(Job).where(
                    Job.status == "QUEUED"
                ).order_by(Job.created_at).limit(1).with_for_update(skip_locked=True)
                
                result = await session.execute(statement)
                job = result.scalar_one_or_none()
                
                if not job:
                    await asyncio.sleep(1)
                    continue
                
                print(f"Worker-{worker_id} Process task: {job.name}")
                print(f"   Job ID: {job.id}")
                
                # Fix: image_path()
                #  job ,
                await session.refresh(job)
                
                #  image_path
                saved_image_path = None
                if hasattr(job, 'image_path') and job.image_path:
                    saved_image_path = job.image_path.strip()
                    print(f"   Job image_path (from job object): '{saved_image_path}'")
                else:
                    print(f"   Job object has no image_path attribute")
                    print(f"   Job object attributes: {dir(job)}")
                    # 
                    db_statement = sa_select(Job.image_path).where(Job.id == job.id)
                    db_result = await session.execute(db_statement)
                    db_image_path = db_result.scalar_one_or_none()
                    if db_image_path:
                        saved_image_path = db_image_path.strip()
                        print(f"   Database image_path: '{saved_image_path}'")
                
                # Check if cancelled before starting processing
                if await check_job_cancelled(session, job.id):
                    print(f"Task cancelled, skip processing")
                    continue
                
                # Check:Update status, saved_image_path ()
                if not saved_image_path:
                    # ,
                    db_statement = sa_select(Job.image_path).where(Job.id == job.id)
                    db_result = await session.execute(db_statement)
                    db_image_path = db_result.scalar_one_or_none()
                    if db_image_path:
                        saved_image_path = db_image_path.strip()
                        print(f"   Database image_path: '{saved_image_path}'")
                
                # Record job start time for latency calculation
                job_start_time = time.time()
                job_created_at = job.created_at.timestamp() if job.created_at else job_start_time
                
                # Record job start time for latency calculation
                job_start_time = time.time()
                job_created_at = job.created_at.timestamp() if job.created_at else job_start_time
                
                #  job  RUNNING( image_path)
                job = await safe_update_job(session, job.id, {"status": "RUNNING"})
                if not job:
                    print(f"Failed to update job status")
                    continue
                
                # Update metrics: active jobs
                metrics.update_active_jobs(worker_id, job.job_type, "RUNNING", 1)
                
                # Update metrics: active jobs
                metrics.update_active_jobs(worker_id, job.job_type, "RUNNING", 1)
                
                # Get image_path(,)
                await session.refresh(job)
                if hasattr(job, 'image_path') and job.image_path:
                    # ,()
                    db_image_path_value = job.image_path.strip()
                    if db_image_path_value:
                        saved_image_path = db_image_path_value
                        print(f"   Job image_path (from job object): '{saved_image_path}'")
                    elif saved_image_path:
                        # ,,
                        print(f"   Warning: image_path mismatch, using: '{saved_image_path}'")
                elif saved_image_path:
                    #  job  image_path ,,
                    print(f"   Warning: Job object has no image_path attribute, using: '{saved_image_path}'")
                
                # Check if cancelled (Update status)
                if await check_job_cancelled(session, job.id):
                    print(f"Task cancelled, stop processing")
                    await safe_update_job(session, job.id, {"status": "CANCELLED"})
                    continue
                
                try:
                    # 3. Prepare large image
                    # Fix:Prefer custom image path saved in job
                    if saved_image_path:
                        # Convert relative path to absolute path if needed
                        if os.path.isabs(saved_image_path):
                            image_path = saved_image_path
                        else:
                            # Get project root directory
                            # Try __file__ first (when running as module)
                            try:
                                current_file = os.path.abspath(__file__)
                                project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
                            except:
                                # Fallback to current working directory (usually project root)
                                project_root = os.getcwd()
                            
                            image_path = os.path.join(project_root, saved_image_path)
                            # Normalize path to handle any redundant separators
                            image_path = os.path.normpath(image_path)
                        print(f"   Use custom image path saved in job: {image_path}")
                    else:
                        # If no custom path,default
                        if os.path.isabs(DEFAULT_IMAGE):
                            image_path = DEFAULT_IMAGE
                        else:
                            image_path = os.path.join(DATA_DIR, DEFAULT_IMAGE)
                        print(f"   Using default: {image_path}")
                        print(f"   Warning: job.image_path not set, using default！")
                    
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Image not found: {image_path}. Please ensure the image file exists.")
                    
                    print(f"   Image path: {image_path}")
                    slide = openslide.OpenSlide(image_path)
                    w, h = slide.dimensions
                    
                    # 
                    coordinates = []
                    for y in range(0, h, STRIDE):
                        for x in range(0, w, STRIDE):
                            if x + TILE_SIZE > w or y + TILE_SIZE > h: continue 
                            coordinates.append((x, y))
                    
                    #  tile 
                    job = await safe_update_job(session, job.id, {
                        "total_tiles": len(coordinates),
                        "processed_tiles": 0
                    })
                    if not job:
                        raise Exception(" job tile ")
                    
                    print(f"   Total tiles: {len(coordinates)}")
                    print(f"   Job Type: {job.job_type}")
                    
                    #  job_type 
                    if job.job_type in ["cell_segmentation", "inference", "segmentation"]:
                        # Task type 1: Cell segmentation
                        await process_cell_segmentation(job, slide, w, h, coordinates, instanseg, session)
                    elif job.job_type in ["tissue_mask", "tissue_mask_generation", "mask"]:
                        # Task type 2: 
                        await process_tissue_mask(job, slide, w, h, coordinates, session)
                    else:
                        raise ValueError(f"Unknown job_type: {job.job_type}. Supported types: 'cell_segmentation', 'tissue_mask'")
                    
                except Exception as e:
                    print(f"Processing failed: {e}")
                    job_id = job.id if job else None
                    print(f"   Job ID: {job_id}")
                    import traceback
                    traceback.print_exc()
                    
                    #  job  FAILED
                    if job_id:
                        # Record metrics: failed job
                        job_end_time = time.time()
                        if 'job_start_time' in locals():
                            latency = job_end_time - job_start_time
                            metrics.record_job_latency(job.job_type if job else "unknown", "FAILED", latency)
                        if 'job_created_at' in locals():
                            duration = job_end_time - job_created_at
                            metrics.record_job_duration(job.job_type if job else "unknown", "FAILED", duration)
                        metrics.increment_job_counter(job.job_type if job else "unknown", "FAILED")
                        
                        failed_job = await safe_update_job(session, job_id, {
                            "status": "FAILED",
                            "error": str(e)[:500]  # 
                        })
                        if not failed_job:
                            print(f"Failed to update job status")

        except Exception as e:
            print(f"Worker error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(run_worker())

