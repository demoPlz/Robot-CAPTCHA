#!/bin/bash
# Check if cloudflare tunnel is running and display the URL

# Check if cloudflared process is running
if pgrep -f "cloudflared tunnel" > /dev/null; then
    echo "Cloudflare tunnel is RUNNING"
    
    # Try to get the URL from the log file
    if [ -f /tmp/cloudflared.log ]; then
        URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared.log | tail -1)
        if [ -n "$URL" ]; then
            echo "Tunnel URL: $URL"
        else
            echo "Could not find URL in /tmp/cloudflared.log"
        fi
    else
        echo "Log file /tmp/cloudflared.log not found"
    fi
    
    exit 0
else
    echo "Cloudflare tunnel is NOT running"
    echo "Start it with: ./start_tunnel.sh"
    exit 1
fi
