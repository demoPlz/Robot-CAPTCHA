#!/bin/bash
# Deploy to Netlify with the current cloudflared tunnel URL

set -e

echo "🚀 Deploy to Netlify with Auto-Detected Tunnel URL"
echo "===================================================="
echo ""

# Check if Flask backend is running
if ! curl -s http://localhost:9000/api/test > /dev/null 2>&1; then
    echo "❌ Backend is not running on port 9000"
    echo "   Please start your Flask backend first"
    exit 1
fi

# Get the cloudflared URL from the backend
echo "1. Fetching cloudflared tunnel URL from backend..."
TUNNEL_INFO=$(curl -s http://localhost:9000/api/cloudflared-url)
TUNNEL_URL=$(echo "$TUNNEL_INFO" | grep -oP '"url":\s*"\K[^"]+' || echo "")

if [ -z "$TUNNEL_URL" ]; then
    echo "❌ Could not get tunnel URL from backend"
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
