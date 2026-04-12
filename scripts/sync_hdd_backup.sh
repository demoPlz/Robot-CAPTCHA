#!/usr/bin/env bash
set -euo pipefail

# Daily mirror from SSD source to HDD backup destination.
# This creates an exact copy at destination (including deletions).
#
# Defaults:
#   SOURCE: ~/crowdmaps_results/lerobot
#   DEST:   /mnt/data_drive/crowdmaps/backup/lerobot
#
# Usage:
#   ./scripts/sync_hdd_backup.sh
#   ./scripts/sync_hdd_backup.sh --dry-run
#   ./scripts/sync_hdd_backup.sh --source /path/to/src --dest /path/to/dst

SOURCE_DEFAULT="$HOME/crowdmaps_results/lerobot"
DEST_DEFAULT="/mnt/data_drive/crowdmaps/backup/lerobot"
LOG_DIR_DEFAULT="$HOME/crowdmaps_results/sync_logs"

SOURCE="$SOURCE_DEFAULT"
DEST="$DEST_DEFAULT"
LOG_DIR="$LOG_DIR_DEFAULT"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --dest)
      DEST="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --source <path>   Source directory (default: $SOURCE_DEFAULT)
  --dest <path>     Destination directory (default: $DEST_DEFAULT)
  --log-dir <path>  Log directory (default: $LOG_DIR_DEFAULT)
  --dry-run         Show changes without writing
  -h, --help        Show this help
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

SOURCE="$(readlink -f "$SOURCE")"
DEST_PARENT="$(dirname "$DEST")"
DEST_BASE="$(basename "$DEST")"
DEST_PARENT="$(readlink -f "$DEST_PARENT")"
DEST="$DEST_PARENT/$DEST_BASE"

mkdir -p "$LOG_DIR"
mkdir -p "$DEST"

if [[ ! -d "$SOURCE" ]]; then
  echo "ERROR: Source directory does not exist: $SOURCE" >&2
  exit 1
fi

if [[ "$SOURCE" == "$DEST" ]]; then
  echo "ERROR: Source and destination resolve to the same path: $SOURCE" >&2
  exit 1
fi

if [[ "$DEST" == "/" || "$DEST" == "$HOME" || "$DEST" == "$HOME/" ]]; then
  echo "ERROR: Destination is unsafe: $DEST" >&2
  exit 1
fi

if [[ "$DEST" != /mnt/data_drive/* ]]; then
  echo "ERROR: Destination must be under /mnt/data_drive for safety. Got: $DEST" >&2
  exit 1
fi

if ! mountpoint -q /mnt/data_drive; then
  echo "ERROR: /mnt/data_drive is not mounted. Aborting to avoid writing elsewhere." >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/sync_hdd_backup_$TIMESTAMP.log"
LOCK_FILE="/tmp/sync_hdd_backup.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: Another sync appears to be running (lock: $LOCK_FILE)" >&2
  exit 1
fi

RSYNC_ARGS=(
  -aHAX
  --numeric-ids
  --delete
  --delete-excluded
  --partial
  --inplace
  --human-readable
  --stats
)

if [[ $DRY_RUN -eq 1 ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

echo "[$(date -Is)] Starting sync"
echo "Source: $SOURCE"
echo "Dest:   $DEST"
echo "DryRun: $DRY_RUN"
echo "Log:    $LOG_FILE"

{
  echo "[$(date -Is)] rsync start"
  rsync "${RSYNC_ARGS[@]}" "$SOURCE/" "$DEST/"
  echo "[$(date -Is)] rsync done"
} | tee "$LOG_FILE"

echo "[$(date -Is)] Sync completed successfully"
