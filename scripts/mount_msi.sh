#!/usr/bin/env bash
# Mount MSI (agate.msi.umn.edu) via SSHFS to ~/mnt/msi
# Uses the "msi" host from ~/.ssh/config (song0837@agate.msi.umn.edu)
#
# Usage:
#   ./mount_msi.sh          # mount
#   ./mount_msi.sh unmount  # unmount

MOUNT_POINT="$HOME/mnt/msi"
REMOTE_HOST="msi"            # SSH config alias
REMOTE_PATH="/projects/standard/ztchen/shared/yilong"

mkdir -p "$MOUNT_POINT"

case "${1:-mount}" in
  mount)
    # Check if already mounted
    if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
      echo "✓ Already mounted at $MOUNT_POINT"
      exit 0
    fi

    echo "Mounting ${REMOTE_HOST}:${REMOTE_PATH} → ${MOUNT_POINT} ..."
    sshfs "${REMOTE_HOST}:${REMOTE_PATH}" "$MOUNT_POINT" \
      -o reconnect \
      -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 \
      -o follow_symlinks \
      -o cache=yes \
      -o kernel_cache \
      -o auto_cache \
      -o compression=no

    if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
      echo "✓ Mounted successfully at $MOUNT_POINT"
    else
      echo "✗ Mount failed" >&2
      exit 1
    fi
    ;;

  unmount|umount)
    if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
      echo "Unmounting $MOUNT_POINT ..."
      fusermount -u "$MOUNT_POINT"
      echo "✓ Unmounted"
    else
      echo "Not currently mounted."
    fi
    ;;

  *)
    echo "Usage: $0 [mount|unmount]" >&2
    exit 1
    ;;
esac
