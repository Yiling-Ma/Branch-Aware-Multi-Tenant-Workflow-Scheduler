#!/bin/bash
# Script to compress video for GitHub upload

INPUT="/Users/yiling/Desktop/Screen Recording 2025-11-24 at 1.08.00 PM.mov"
OUTPUT="demo_compressed.mp4"

echo "📹 Compressing video..."
echo "Input: $INPUT"
echo "Output: $OUTPUT"

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg is not installed."
    echo "Install with: brew install ffmpeg"
    exit 1
fi

# Compress video (target size ~50MB)
ffmpeg -i "$INPUT" \
  -c:v libx264 \
  -crf 28 \
  -preset slow \
  -vf "scale=1920:-1" \
  -c:a aac \
  -b:a 128k \
  "$OUTPUT"

if [ $? -eq 0 ]; then
    echo "✅ Compression complete!"
    echo "Original size: $(du -h "$INPUT" | cut -f1)"
    echo "Compressed size: $(du -h "$OUTPUT" | cut -f1)"
    echo ""
    echo "📤 You can now upload $OUTPUT to GitHub"
else
    echo "❌ Compression failed"
    exit 1
fi

