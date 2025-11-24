#!/bin/bash
# Download OpenSlide test data

DATA_DIR="./data"
BASE_URL="https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio"

# Create data directory
mkdir -p "$DATA_DIR"

echo "=========================================="
echo "  Downloading OpenSlide test data"
echo "=========================================="
echo ""

# Download CMU-1-Small-Region.svs (1.85 MB) - Recommended for testing
echo "📥 Downloading CMU-1-Small-Region.svs (1.85 MB)..."
curl -L -o "$DATA_DIR/CMU-1-Small-Region.svs" "$BASE_URL/CMU-1-Small-Region.svs"

if [ $? -eq 0 ]; then
    echo "✅ CMU-1-Small-Region.svs downloaded successfully"
    ls -lh "$DATA_DIR/CMU-1-Small-Region.svs"
else
    echo "❌ Download failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "  Optional: Download larger test files"
echo "=========================================="
echo ""
echo "If you need larger test files, you can run:"
echo "  curl -L -o $DATA_DIR/CMU-1.svs $BASE_URL/CMU-1.svs"
echo "  curl -L -o $DATA_DIR/CMU-2.svs $BASE_URL/CMU-2.svs"
echo ""
echo "Note: Large files (100+ MB) may take a long time to download"
echo ""
