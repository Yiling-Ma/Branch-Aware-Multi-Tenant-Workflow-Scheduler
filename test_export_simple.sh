#!/bin/bash
# Simple test script for export functionality using curl

BASE_URL="http://127.0.0.1:8000"

echo "============================================================"
echo "  Simple Export Test (using curl)"
echo "============================================================"
echo ""

# Check if server is running
echo "1. Checking if server is running..."
if curl -s "${BASE_URL}/" > /dev/null 2>&1; then
    echo "   ✅ Server is running"
else
    echo "   ❌ Server is not running"
    echo "      Start with: ./start_server.sh"
    exit 1
fi

# Find a completed cell_segmentation job
echo ""
echo "2. Finding completed cell_segmentation jobs..."
JOBS_RESPONSE=$(curl -s "${BASE_URL}/debug/jobs")
JOB_ID=$(echo "$JOBS_RESPONSE" | python3 -c "
import sys, json
try:
    jobs = json.load(sys.stdin)
    for job in jobs:
        if job.get('job_type') == 'cell_segmentation' and job.get('status') == 'SUCCEEDED':
            print(job['id'])
            sys.exit(0)
    print('NONE')
except:
    print('ERROR')
")

if [ "$JOB_ID" = "NONE" ] || [ "$JOB_ID" = "ERROR" ] || [ -z "$JOB_ID" ]; then
    echo "   ⚠️  No completed cell_segmentation jobs found"
    echo ""
    echo "   To create a test job:"
    echo "   1. Open http://127.0.0.1:8000/ in browser"
    echo "   2. Use 'Test Scheduler' to create a cell_segmentation job"
    echo "   3. Wait for the job to complete"
    echo "   4. Run this script again"
    exit 1
fi

echo "   ✅ Found job: ${JOB_ID}"

# Test JSON export
echo ""
echo "3. Testing JSON export..."
JSON_RESPONSE=$(curl -s -w "\n%{http_code}" "${BASE_URL}/jobs/${JOB_ID}/export?format=json")
HTTP_CODE=$(echo "$JSON_RESPONSE" | tail -n1)
JSON_BODY=$(echo "$JSON_RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ JSON export successful (HTTP 200)"
    echo "$JSON_BODY" > "test_export_${JOB_ID:0:8}.json"
    TOTAL_CELLS=$(echo "$JSON_BODY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total_cells', 0))" 2>/dev/null || echo "unknown")
    echo "   Total cells: ${TOTAL_CELLS}"
    echo "   Saved to: test_export_${JOB_ID:0:8}.json"
else
    echo "   ❌ JSON export failed (HTTP ${HTTP_CODE})"
    echo "   Response: ${JSON_BODY:0:200}"
fi

# Test CSV export
echo ""
echo "4. Testing CSV export..."
CSV_RESPONSE=$(curl -s -w "\n%{http_code}" "${BASE_URL}/jobs/${JOB_ID}/export?format=csv")
HTTP_CODE=$(echo "$CSV_RESPONSE" | tail -n1)
CSV_BODY=$(echo "$CSV_RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ CSV export successful (HTTP 200)"
    echo "$CSV_BODY" > "test_export_${JOB_ID:0:8}.csv"
    ROW_COUNT=$(echo "$CSV_BODY" | wc -l | tr -d ' ')
    echo "   Total rows: $((ROW_COUNT - 1))"  # Subtract header
    echo "   Saved to: test_export_${JOB_ID:0:8}.csv"
    echo ""
    echo "   First few lines:"
    echo "$CSV_BODY" | head -3
else
    echo "   ❌ CSV export failed (HTTP ${HTTP_CODE})"
    echo "   Response: ${CSV_BODY:0:200}"
fi

echo ""
echo "============================================================"
echo "  Test Complete"
echo "============================================================"

