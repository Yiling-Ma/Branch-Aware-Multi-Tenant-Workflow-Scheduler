#!/bin/bash
# Script to help upload video to GitHub Releases
# This script provides instructions and checks prerequisites

VIDEO_FILE="/Users/yiling/Desktop/Screen Recording 2025-11-24 at 1.08.00 PM.mov"

echo "📹 GitHub Release Video Upload Helper"
echo "======================================"
echo ""

# Check if file exists
if [ ! -f "$VIDEO_FILE" ]; then
    echo "❌ Video file not found: $VIDEO_FILE"
    exit 1
fi

# Get file size
FILE_SIZE=$(du -h "$VIDEO_FILE" | cut -f1)
echo "📁 Video file: $VIDEO_FILE"
echo "📊 File size: $FILE_SIZE"
echo ""

# Check if gh CLI is installed
if command -v gh &> /dev/null; then
    echo "✅ GitHub CLI (gh) is installed"
    echo ""
    echo "You can upload using GitHub CLI:"
    echo "  1. Create a release: gh release create v1.0.0 --title 'Demo Video' --notes 'Screen recording demo'"
    echo "  2. Upload video: gh release upload v1.0.0 '$VIDEO_FILE'"
    echo ""
else
    echo "ℹ️  GitHub CLI (gh) is not installed"
    echo "   Install with: brew install gh"
    echo ""
fi

echo "📝 Manual Upload Steps:"
echo "======================"
echo ""
echo "1. Go to your GitHub repository"
echo "2. Click 'Releases' → 'Create a new release'"
echo "3. Tag version: v1.0.0"
echo "4. Release title: Demo Video"
echo "5. Description: Screen recording demo of the WSI Cell Segmentation Scheduler"
echo "6. Click 'Publish release'"
echo "7. Edit the release → Scroll to 'Attach binaries'"
echo "8. Drag and drop: $VIDEO_FILE"
echo "9. Wait for upload → Click 'Update release'"
echo ""
echo "🔗 After uploading, update README.md with:"
echo "   [📹 Watch Demo Video](https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0/Screen-Recording-2025-11-24-at-1.08.00-PM.mov)"
echo ""

