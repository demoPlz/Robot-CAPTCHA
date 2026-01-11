# Demo Video Hosting Strategy

## Problem
Demo videos were compressed to ~200kbps (resulting in blurry videos) to reduce bandwidth through the free Cloudflare tunnel.

## Solution
Host uncompressed videos on Netlify CDN for high-quality delivery.

## Setup

### 1. Copy Uncompressed Videos
```bash
./scripts/copy_uncompressed_demos.sh
```

This copies original videos from `data/prompts/drawer/demos` to `public/demos_hq/` (108MB total).

### 2. Configure Frontend URL
In `backend/crowd_interface_config.py`, set your Netlify URL:

```python
self.frontend_url = "https://your-site.netlify.app"  # Your Netlify deployment URL
```

### 3. Deploy to Netlify
```bash
git add public/demos_hq
git commit -m "Add uncompressed demo videos for CDN hosting"
git push
```

Netlify will automatically deploy the videos.

## How It Works

- **With `frontend_url` set**: Backend returns URLs like `https://your-site.netlify.app/demos_hq/1.webm`
- **Without `frontend_url`**: Backend returns URLs like `/demos/1.webm` (served through tunnel, compressed)

The frontend automatically handles both URL formats.

## File Sizes

- **Compressed** (`public/demos`): ~1-2MB per video, 480p, 200kbps
- **Uncompressed** (`public/demos_hq`): ~2-18MB per video, original quality

## Cost Considerations

Netlify free tier includes:
- 100GB bandwidth/month
- With 108MB of videos, you can serve ~900 full video sets per month on free tier
- If you exceed, Netlify offers paid plans with higher bandwidth

Cloudflare tunnel remains free for backend API calls (much lower bandwidth).
