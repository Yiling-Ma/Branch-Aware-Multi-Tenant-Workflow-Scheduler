# How to Test Export Functionality

This guide explains how to test the cell segmentation export functionality.

## Quick Test

Run the automated test script:

```bash
python test_export.py
```

This script will:
1. Check if the server is running
2. Find a completed cell_segmentation job
3. Test JSON export
4. Test CSV export
5. Validate the exported data
6. Save sample export files

## Manual Testing Steps

### Step 1: Ensure Server is Running

```bash
./start_server.sh 4
```

Or check if it's already running:
```bash
curl http://127.0.0.1:8000/
```

### Step 2: Find a Completed Job

#### Option A: Using the Web UI

1. Open http://127.0.0.1:8000/ in your browser
2. Click "Test Scheduler"
3. Create a cell_segmentation job
4. Wait for it to complete (status: SUCCEEDED)
5. Note the Job ID

#### Option B: Using API

```bash
# List all jobs
curl http://127.0.0.1:8000/debug/jobs | jq '.[] | select(.job_type=="cell_segmentation" and .status=="SUCCEEDED") | {id, name, status}'
```

### Step 3: Test JSON Export

```bash
# Replace {job_id} with your actual job ID
JOB_ID="your-job-id-here"

# Export as JSON
curl -X GET "http://127.0.0.1:8000/jobs/${JOB_ID}/export?format=json" \
  -H "Accept: application/json" \
  -o test_export.json

# View the result
cat test_export.json | jq '.total_cells'
cat test_export.json | jq '.cells[0]'
```

**Expected Output:**
```json
{
  "job_id": "...",
  "job_name": "Cell Segmentation",
  "total_cells": 1234,
  "cells": [
    {
      "cell_id": 1,
      "centroid": {"x": 100, "y": 200},
      "polygon": [[95,195], [105,195], [105,205], [95,205]],
      "polygon_point_count": 4,
      "metadata": {...}
    }
  ]
}
```

### Step 4: Test CSV Export

```bash
# Export as CSV
curl -X GET "http://127.0.0.1:8000/jobs/${JOB_ID}/export?format=csv" \
  -H "Accept: text/csv" \
  -o test_export.csv

# View the result
head -5 test_export.csv
```

**Expected Output:**
```csv
cell_id,centroid_x,centroid_y,polygon_coords,polygon_point_count,area_pixels
1,100,200,"[[95,195],[105,195],[105,205],[95,205]]",4,100
2,150,250,"[[145,245],[155,245],[155,255],[145,255]]",4,100
```

### Step 5: Validate Exported Data

#### Validate JSON

```python
import json

with open("test_export.json", "r") as f:
    data = json.load(f)

# Check structure
assert "job_id" in data
assert "total_cells" in data
assert "cells" in data

# Check first cell
cell = data["cells"][0]
assert "cell_id" in cell
assert "centroid" in cell
assert "polygon" in cell
assert len(cell["polygon"]) >= 3

print(f"✅ Valid JSON export with {data['total_cells']} cells")
```

#### Validate CSV

```python
import csv
import json

with open("test_export.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Check structure
assert len(rows) > 0
row = rows[0]
assert "cell_id" in row
assert "centroid_x" in row
assert "polygon_coords" in row

# Validate polygon can be parsed
polygon = json.loads(row["polygon_coords"])
assert isinstance(polygon, list)
assert len(polygon) >= 3

print(f"✅ Valid CSV export with {len(rows)} rows")
```

## Testing Error Cases

### Test 1: Invalid Job ID

```bash
curl "http://127.0.0.1:8000/jobs/invalid-id/export?format=json"
```

**Expected:** 400 or 422 error

### Test 2: Non-existent Job

```bash
curl "http://127.0.0.1:8000/jobs/00000000-0000-0000-0000-000000000000/export?format=json"
```

**Expected:** 404 error

### Test 3: Wrong Job Type

```bash
# Try to export a tissue_mask job
curl "http://127.0.0.1:8000/jobs/{tissue_mask_job_id}/export?format=json"
```

**Expected:** 400 error with message about job type

### Test 4: Job Not Completed

```bash
# Try to export a RUNNING job
curl "http://127.0.0.1:8000/jobs/{running_job_id}/export?format=json"
```

**Expected:** 400 error with message about job status

## Using Python for Testing

```python
import requests
import json

BASE_URL = "http://127.0.0.1:8000"
JOB_ID = "your-job-id-here"

# Test JSON export
response = requests.get(
    f"{BASE_URL}/jobs/{JOB_ID}/export",
    params={"format": "json"}
)

if response.status_code == 200:
    data = response.json()
    print(f"✅ JSON export successful!")
    print(f"   Total cells: {data['total_cells']}")
    print(f"   First cell: {data['cells'][0]}")
else:
    print(f"❌ Export failed: {response.status_code}")
    print(f"   {response.text}")

# Test CSV export
response = requests.get(
    f"{BASE_URL}/jobs/{JOB_ID}/export",
    params={"format": "csv"}
)

if response.status_code == 200:
    print(f"✅ CSV export successful!")
    print(f"   Size: {len(response.text)} bytes")
    print(f"   First 200 chars:\n{response.text[:200]}")
else:
    print(f"❌ Export failed: {response.status_code}")
```

## Using Swagger UI

1. Open http://127.0.0.1:8000/docs
2. Find the `GET /jobs/{job_id}/export` endpoint
3. Click "Try it out"
4. Enter a job_id
5. Select format (json or csv)
6. Click "Execute"
7. Review the response

## Expected Results

### Successful Export

- Status code: 200
- Content-Type: `application/json` (JSON) or `text/csv` (CSV)
- Response includes:
  - Job metadata (job_id, job_name, etc.)
  - Total cell count
  - Array of cells with:
    - cell_id
    - centroid (x, y)
    - polygon (array of [x, y] coordinates)
    - metadata

### Error Cases

- **404**: Job not found
- **400**: Invalid job type or status
- **500**: Server error (check logs)

## Troubleshooting

### "No completed jobs found"

**Solution:**
1. Create a cell_segmentation job via the UI
2. Wait for it to complete
3. Check job status: `curl http://127.0.0.1:8000/debug/jobs | jq '.[] | select(.id=="your-job-id")'`

### "Export failed: 400"

**Possible causes:**
- Job type is not `cell_segmentation`
- Job status is not `SUCCEEDED`
- Job has no result_metadata

**Solution:**
- Check job details: `curl http://127.0.0.1:8000/jobs/{job_id}`
- Verify job completed successfully

### "Export failed: 404"

**Solution:**
- Verify job_id is correct
- Check if job exists: `curl http://127.0.0.1:8000/jobs/{job_id}`

### "Server not responding"

**Solution:**
```bash
# Check if server is running
ps aux | grep uvicorn

# Restart server
./start_server.sh 4
```

## Sample Test Output

```
============================================================
  Export Functionality Test
============================================================

✅ Server is running

1. Searching for completed cell_segmentation jobs...
   ✅ Found completed job: Cell Segmentation
      Job ID: 550e8400-e29b-41d4-a716-446655440000
      Status: SUCCEEDED

2. Testing JSON export for job 550e8400-e29b-41d4-a716-446655440000...
   ✅ JSON export successful!
      Total cells: 1234
      Pixel size: 0.5
      First cell ID: 1
      First cell centroid: (1024, 2048)
      First cell polygon points: 4
      Saved to: export_test_550e8400.json

3. Testing CSV export for job 550e8400-e29b-41d4-a716-446655440000...
   ✅ CSV export successful!
      Total rows: 1234
      First cell ID: 1
      First cell centroid: (1024, 2048)
      First cell polygon points: 4
      First cell area: 100.0 pixels
      Saved to: export_test_550e8400.csv

4. Testing error handling...
   ✅ Correctly returns 404 for invalid job ID
   ✅ Correctly rejects invalid format

============================================================
  Test Summary
============================================================
JSON Export: ✅ PASS
CSV Export: ✅ PASS

✅ All export tests passed!

📁 Exported files:
   - export_test_550e8400.json
   - export_test_550e8400.csv
```

