#!/bin/bash
# Deploy to Netlify with the current cloudflared tunnel URL

set -e

echo "🚀 Deploy to Netlify with Auto-Detected Tunnel URL"
echo "===================================================="
echo ""

echo "1. Detecting cloudflared tunnel URL..."
TUNNEL_OUTPUT=$(./scripts/check_tunnel.sh)

if ! echo "$TUNNEL_OUTPUT" | grep -q "tunnel is RUNNING"; then
    echo "❌ No cloudflared tunnel is running"
    echo "   Start it with: ./scripts/start_tunnel.sh"
    exit 1
fi

TUNNEL_URL=$(echo "$TUNNEL_OUTPUT" | grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com')

if [ -z "$TUNNEL_URL" ]; then
    echo "❌ Could not extract tunnel URL"
    exit 1
fi

echo "   ✅ Detected tunnel URL: $TUNNEL_URL"
echo ""

# Update the config file
echo "2. Updating public/backend-config.json..."
cat > public/backend-config.json << EOF
{
  "backendUrl": "$TUNNEL_URL",
  "updatedAt": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo "   ✅ Config file updated"
echo ""

# Build the site
echo "3. Building site..."
npm run build
echo "   ✅ Build complete"
echo ""

# Deploy to Netlify
echo "4. Deploying to Netlify..."
npx netlify deploy --prod --dir=dist

echo ""
echo "===================================================="
echo "✅ Deployment complete!"
echo ""
echo "Your site will use backend: $TUNNEL_URL"
