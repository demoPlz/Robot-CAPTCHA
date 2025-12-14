# Deployment Guide

## Quick Setup

```bash
# 1. Start tunnel (in tmux for persistence)
./start_tunnel.sh

# 2. Start backend
conda activate csui
python backend/collect_data.py <your-args>

# 3. Deploy to Netlify (once per tunnel restart)
./deploy_with_tunnel.sh
```

## Access

- **Local dev**: `npm run dev` → `localhost:5173`
- **Remote access**: Share your Netlify URL

## When to Redeploy

✅ **Redeploy when:**
- Tunnel restarts
- PC reboots

❌ **No need to redeploy when:**
- Backend restarts
- Code changes (backend only)
- Daily usage with tunnel running in tmux

## How It Works

1. `start_tunnel.sh` - Creates cloudflared tunnel, logs URL to `/tmp/cloudflared.log`
2. `deploy_with_tunnel.sh` - Reads tunnel URL, updates `public/backend-config.json`, builds & deploys
3. Remote users access Netlify → Frontend reads config → Connects via tunnel to your local backend

## Troubleshooting

**Tunnel URL changed but forgot to redeploy?**
```bash
./deploy_with_tunnel.sh
```

**Check if tunnel is running:**
```bash
ps aux | grep cloudflared
# or
tmux attach
```

**Test tunnel detection:**
```bash
./test_tunnel_detection.sh
```
