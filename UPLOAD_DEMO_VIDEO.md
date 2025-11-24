# How to Upload Demo Video to GitHub

Your video file is **217MB**, which exceeds GitHub's 100MB file size limit. Here are several options:

## Option 1: GitHub Releases (Recommended) ⭐

GitHub Releases allows files up to **2GB** per file.

### Steps:

1. **Create a release on GitHub:**
   - Go to your repository on GitHub
   - Click "Releases" → "Create a new release"
   - Tag version: `v1.0.0` (or any version)
   - Release title: `Demo Video`
   - Description: `Screen recording demo of the WSI Cell Segmentation Scheduler`
   - Click "Publish release"

2. **Upload video to the release:**
   - After creating the release, click "Edit release"
   - Scroll down to "Attach binaries"
   - Drag and drop your video file: `Screen Recording 2025-11-24 at 1.08.00 PM.mov`
   - Wait for upload to complete
   - Click "Update release"

3. **Add link to README.md:**
   ```markdown
   ## Demo
   
   [📹 Watch Demo Video](https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0/Screen-Recording-2025-11-24-at-1.08.00-PM.mov)
   ```

## Option 2: Compress Video First

Reduce file size before uploading:

### Using ffmpeg (if installed):
```bash
# Convert to MP4 with compression
ffmpeg -i "/Users/yiling/Desktop/Screen Recording 2025-11-24 at 1.08.00 PM.mov" \
  -c:v libx264 -crf 28 -preset slow \
  -c:a aac -b:a 128k \
  "demo_compressed.mp4"
```

### Using HandBrake (GUI tool):
1. Download HandBrake: https://handbrake.fr/
2. Open video file
3. Select "Fast 1080p30" preset
4. Click "Start Encode"
5. Upload the compressed file to GitHub

## Option 3: Upload to YouTube/Vimeo (Best for Large Files)

1. **Upload to YouTube:**
   - Go to https://www.youtube.com/upload
   - Upload your video
   - Set visibility to "Unlisted" (or "Public")
   - Copy the video URL

2. **Add to README.md:**
   ```markdown
   ## Demo
   
   [📹 Watch Demo on YouTube](https://youtube.com/watch?v=YOUR_VIDEO_ID)
   
   Or embed directly:
   [![Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://youtube.com/watch?v=YOUR_VIDEO_ID)
   ```

## Option 4: GitHub LFS (Large File Storage)

For files that need to be in the repository:

1. **Install Git LFS:**
   ```bash
   brew install git-lfs  # macOS
   # or download from: https://git-lfs.github.com/
   ```

2. **Initialize Git LFS:**
   ```bash
   cd my-scheduler
   git lfs install
   ```

3. **Track video files:**
   ```bash
   git lfs track "*.mov"
   git add .gitattributes
   ```

4. **Add and commit:**
   ```bash
   git add "Screen Recording 2025-11-24 at 1.08.00 PM.mov"
   git commit -m "Add demo video"
   git push
   ```

⚠️ **Note:** GitHub LFS has storage limits (1GB for free accounts).

## Recommended Approach

For a demo video, I recommend **Option 1 (GitHub Releases)** or **Option 3 (YouTube)**:
- GitHub Releases: Easy, no external dependencies
- YouTube: Better for sharing, supports embedding, no size limits

