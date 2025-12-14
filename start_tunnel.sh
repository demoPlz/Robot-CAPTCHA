#!/bin/bash
# Start cloudflared tunnel with automatic URL logging
# This script starts a cloudflared tunnel and logs the URL to /tmp/cloudflared.log
# The Flask backend will read this file to automatically discover the tunnel URL

PORT=${1:-9000}

echo "Starting cloudflared tunnel for port $PORT..."
echo "The tunnel URL will be automatically detected by the application."

# Kill any existing cloudflared processes
pkill -f cloudflared

# Start cloudflared and log output
cloudflared tunnel --url http://localhost:$PORT 2>&1 | tee /tmp/cloudflared.log &

# Wait a moment for the tunnel to start
sleep 3

# Extract and display the URL
if [ -f /tmp/cloudflared.log ]; then
    TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared.log | head -1)
    if [ -n "$TUNNEL_URL" ]; then
        echo ""
        echo "✅ Tunnel is running at: $TUNNEL_URL"
        echo "✅ Your frontend will automatically use this URL"
        echo ""
    else
        echo "⏳ Waiting for tunnel URL to appear in logs..."
    fi
else
    echo "⚠️  Could not find cloudflared log file"
fi

echo "To stop the tunnel, run: pkill -f cloudflared"
