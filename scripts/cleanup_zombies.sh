#!/bin/bash
# Cleanup script to kill all zombie processes from previous crowdsourcing-ui runs
# This kills Flask servers, Isaac Sim workers, pose estimation workers, and other related processes

set -e

echo "🧹 Cleaning up zombie processes from previous runs..."
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

KILLED_COUNT=0

# Function to kill processes by pattern with confirmation
kill_by_pattern() {
    local pattern=$1
    local description=$2
    
    echo -e "${YELLOW}Searching for: ${description}${NC}"
    
    # Find PIDs matching the pattern
    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    
    if [ -n "$pids" ]; then
        echo "  Found PIDs: $pids"
        for pid in $pids; do
            # Get process info for confirmation
            local proc_info=$(ps -p $pid -o pid,ppid,cmd --no-headers 2>/dev/null || echo "")
            if [ -n "$proc_info" ]; then
                echo "  → Killing PID $pid: $(echo $proc_info | cut -c 1-100)"
                kill -9 $pid 2>/dev/null || true
                KILLED_COUNT=$((KILLED_COUNT + 1))
            fi
        done
        echo -e "${GREEN}  ✓ Killed processes${NC}"
    else
        echo "  No processes found"
    fi
    echo ""
}

# Function to kill processes on specific port
kill_by_port() {
    local port=$1
    local description=$2
    
    echo -e "${YELLOW}Checking port ${port}: ${description}${NC}"
    
    # Find process using the port
    local pid=$(lsof -ti:$port 2>/dev/null || true)
    
    if [ -n "$pid" ]; then
        echo "  Found PID $pid using port $port"
        local proc_info=$(ps -p $pid -o pid,ppid,cmd --no-headers 2>/dev/null || echo "")
        if [ -n "$proc_info" ]; then
            echo "  → Killing PID $pid: $(echo $proc_info | cut -c 1-100)"
            kill -9 $pid 2>/dev/null || true
            KILLED_COUNT=$((KILLED_COUNT + 1))
            echo -e "${GREEN}  ✓ Port $port freed${NC}"
        fi
    else
        echo "  Port $port is free"
    fi
    echo ""
}

# 1. Kill Flask server on port 9000
kill_by_port 9000 "Flask server"

# 2. Kill Flask/Werkzeug servers
kill_by_pattern "werkzeug.*9000" "Werkzeug servers"

# 3. Kill collect_data.py processes
kill_by_pattern "collect_data.py" "Main data collection script"

# 4. Kill flask_app.py processes
kill_by_pattern "flask_app.py" "Flask app processes"

# 5. Kill Isaac Sim worker processes
kill_by_pattern "isaac_sim_worker.py" "Isaac Sim workers"
kill_by_pattern "persistent_isaac_sim_worker.py" "Persistent Isaac Sim workers"
kill_by_pattern "isaac_sim_worker_manager.py" "Isaac Sim worker managers"

# 6. Kill any Isaac Sim / Omniverse processes
kill_by_pattern "omni.*isaac" "Omniverse Isaac Sim processes"
kill_by_pattern "kit.*isaac" "Isaac Sim Kit processes"

# 7. Kill pose estimation workers
kill_by_pattern "pose_worker.py" "Pose estimation workers"
kill_by_pattern "estimate_pose.py" "Pose estimation processes"

# 8. Kill any Python processes in the backend directory
kill_by_pattern "python.*backend/.*\.py" "Backend Python processes"

# 9. Kill cloudflared tunnel if running
kill_by_pattern "cloudflared.*tunnel" "Cloudflared tunnel"

# 10. Clean up any stale temporary files
echo -e "${YELLOW}Cleaning temporary files...${NC}"
if [ -d "/tmp/crowd_obs_cache" ]; then
    echo "  Removing /tmp/crowd_obs_cache"
    rm -rf /tmp/crowd_obs_cache 2>/dev/null || true
fi

if [ -f "/tmp/cloudflared.log" ]; then
    echo "  Removing /tmp/cloudflared.log"
    rm -f /tmp/cloudflared.log 2>/dev/null || true
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $KILLED_COUNT -gt 0 ]; then
    echo -e "${GREEN}✓ Cleanup complete: Killed $KILLED_COUNT process(es)${NC}"
else
    echo -e "${GREEN}✓ No zombie processes found - system is clean${NC}"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "You can now run your data collection script."
