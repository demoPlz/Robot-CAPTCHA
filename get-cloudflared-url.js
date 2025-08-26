#!/usr/bin/env node

// Simple script to extract cloudflared URL from log and update Netlify environment
const fs = require('fs');
const { execSync } = require('child_process');

try {
    // Read the cloudflared log
    const log = fs.readFileSync('cloudflared.log', 'utf8');
    
    // Extract the tunnel URL
    const match = log.match(/https:\/\/[a-z0-9-]+\.trycloudflare\.com/);
    
    if (match) {
        const tunnelUrl = match[0];
        console.log('Current cloudflared URL:', tunnelUrl);
        
        // Update Netlify environment variable
        try {
            execSync(`netlify env:set BACKEND_URL ${tunnelUrl} --force`, { stdio: 'inherit' });
            console.log('✅ Updated Netlify environment variable');
        } catch (error) {
            console.error('❌ Failed to update Netlify environment:', error.message);
        }
        
        // Also output for manual use
        console.log('\nTo manually update your files, use this URL:');
        console.log(tunnelUrl);
        
    } else {
        console.error('❌ Could not find cloudflared URL in log file');
        process.exit(1);
    }
    
} catch (error) {
    console.error('❌ Error reading cloudflared.log:', error.message);
    console.log('\nMake sure cloudflared is running and cloudflared.log exists');
    process.exit(1);
}
