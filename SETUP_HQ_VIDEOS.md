# Quick Setup: High-Quality Demo Videos on Netlify

## What I've done:
1. ✅ Copied uncompressed videos to `public/demos_hq/` (108MB total)
2. ✅ Updated backend code to use Netlify URLs when configured
3. ✅ Added configuration option in `crowd_interface_config.py`

## What YOU need to do:

### Step 1: Find your Netlify URL
Go to your Netlify dashboard and find your site's URL (e.g., `https://your-project.netlify.app`)

### Step 2: Update the config
Edit `backend/crowd_interface_config.py` line 53:

```python
# Change this:
self.frontend_url: str | None = None

# To this (use YOUR Netlify URL):
self.frontend_url: str | None = "https://your-project.netlify.app"
```

### Step 3: Commit and deploy to Netlify
```bash
git add .
git commit -m "Add high-quality demo videos hosted on Netlify CDN"
git push
```

Netlify will automatically rebuild and deploy the uncompressed videos.

### Step 4: Restart your backend
After deploying, restart your backend server so it picks up the new `frontend_url` config.

## That's it!

Videos will now be served from Netlify's CDN in full quality instead of compressed through the tunnel.

## Verification
Once deployed, check the browser console. The video URL should look like:
- ✅ `https://your-project.netlify.app/demos_hq/1.webm` (using CDN)
- ❌ Not `/demos/1.webm` (compressed through tunnel)
