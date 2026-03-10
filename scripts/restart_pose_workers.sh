#!/usr/bin/env bash
# ============================================================================
# restart_pose_workers.sh — Kill and restart pose estimation workers in-place.
#
# Safe to run while collect_data.py is running. The main process communicates
# with workers purely via files (inbox/ → outbox/), so killing and restarting
# workers is transparent.
#
# Usage:
#   bash scripts/restart_pose_workers.sh
#
# What it does:
#   1. Kills existing pose_worker.py processes (the actual python, not conda)
#   2. Kills their conda-run wrappers
#   3. Moves any orphaned jobs from tmp/ back to inbox/
#   4. Relaunches workers with the same arguments in the any6d conda env
# ============================================================================
set -euo pipefail

JOBS_DIR="/tmp/crowd_obs_cache/pose_jobs"
POSE_ENV="${POSE_ENV:-any6d}"
WORKER_SCRIPT="/home/yilong/crowdsourcing-ui/backend/any6d/pose_worker.py"

# CUDA library paths for the any6d environment
CONDA_PREFIX="$HOME/miniconda3/envs/$POSE_ENV"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

echo "🔧 === RESTARTING POSE WORKERS ==="
echo ""

# ── Step 1: Find and kill existing workers ──────────────────────────────────
echo "🛑 Step 1: Killing existing pose workers..."

# Find the actual python pose_worker.py processes
WORKER_PIDS=$(pgrep -f "pose_worker.py.*--jobs-dir" 2>/dev/null || true)

if [ -z "$WORKER_PIDS" ]; then
    echo "   No running pose workers found."
else
    for pid in $WORKER_PIDS; do
        CMDLINE=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' || echo "unknown")
        echo "   Killing PID $pid: $CMDLINE"
        kill -TERM "$pid" 2>/dev/null || true
    done

    # Wait briefly for graceful shutdown
    sleep 2

    # Force kill if any are still alive
    for pid in $WORKER_PIDS; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Force killing PID $pid..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    # Also kill any orphaned conda-run wrappers
    CONDA_PIDS=$(pgrep -f "conda run.*pose_worker" 2>/dev/null || true)
    for pid in $CONDA_PIDS; do
        echo "   Killing conda wrapper PID $pid"
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 1
fi

echo "   ✅ Workers stopped."
echo ""

# ── Step 2: Recover orphaned jobs from tmp/ ─────────────────────────────────
echo "🔄 Step 2: Recovering orphaned jobs from tmp/..."

RECOVERED=0
shopt -s nullglob
for f in "$JOBS_DIR/tmp/"*.json; do
    BASENAME=$(basename "$f")
    mv "$f" "$JOBS_DIR/inbox/$BASENAME"
    echo "   Recovered: $BASENAME"
    RECOVERED=$((RECOVERED + 1))
done
shopt -u nullglob

if [ "$RECOVERED" -gt 0 ]; then
    echo "   ✅ Recovered $RECOVERED orphaned job(s)."
else
    echo "   No orphaned jobs found."
fi

PENDING=$(find "$JOBS_DIR/inbox/" -name '*.json' 2>/dev/null | wc -l)
echo "   📬 $PENDING pending job(s) in inbox."
echo ""

# ── Step 3: Relaunch workers ────────────────────────────────────────────────
echo "🚀 Step 3: Launching fresh pose workers..."

# Worker definitions: object|mesh_path|prompt
# Edit these if your objects change
WORKERS=(
    "container|/home/yilong/crowdsourcing-ui/public/assets/container.stl|Teal Cube Container|0.001"
    "cup|/home/yilong/crowdsourcing-ui/public/assets/cup.stl|Red Cylinder|0.001"
)

for entry in "${WORKERS[@]}"; do
    IFS='|' read -r OBJ MESH PROMPT SCALE <<< "$entry"

    echo "   Starting worker: object=$OBJ, mesh=$MESH, prompt='$PROMPT', scale=$SCALE"

    conda run --no-capture-output -n "$POSE_ENV" \
        python "$WORKER_SCRIPT" \
        --jobs-dir "$JOBS_DIR" \
        --object "$OBJ" \
        --mesh "$MESH" \
        --prompt "$PROMPT" \
        --mesh-scale "$SCALE" \
        > >(while IFS= read -r line; do echo "[$OBJ] $line"; done) \
        2>&1 &

    NEW_PID=$!
    echo "   ✅ Worker '$OBJ' launched (background PID $NEW_PID)"
done

echo ""

# Wait a few seconds and check they're alive
echo "⏳ Waiting for workers to initialize..."
sleep 5

ALIVE=0
DEAD=0
for entry in "${WORKERS[@]}"; do
    IFS='|' read -r OBJ MESH PROMPT SCALE <<< "$entry"
    PID=$(pgrep -f "pose_worker.py.*--object $OBJ" 2>/dev/null | tail -1 || true)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "   ✅ $OBJ worker alive (PID $PID)"
        ALIVE=$((ALIVE + 1))
    else
        echo "   ❌ $OBJ worker NOT running!"
        DEAD=$((DEAD + 1))
    fi
done

echo ""
if [ "$DEAD" -eq 0 ]; then
    echo "✅ === ALL POSE WORKERS RESTARTED SUCCESSFULLY ==="
else
    echo "⚠️  === $DEAD worker(s) failed to start, check logs above ==="
fi
echo ""
echo "The main collect_data.py process will automatically pick up results from"
echo "the restarted workers via the outbox/ directory. No further action needed."
