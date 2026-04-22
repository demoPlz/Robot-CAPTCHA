#!/bin/bash
# Copy uncompressed demo videos to public directory for Netlify hosting

TASK_NAME="sorting"
SOURCE_DIR="data/prompts/${TASK_NAME}/demos"
DEST_DIR="public/demos_hq"

echo "Copying uncompressed demo videos to Netlify public directory..."
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo ""

mkdir -p "$DEST_DIR"
rm -f "$DEST_DIR"/*

# Copy all videos
cp -v "$SOURCE_DIR"/*.webm "$DEST_DIR/" 2>/dev/null || echo "No .webm files found"
cp -v "$SOURCE_DIR"/*.mp4 "$DEST_DIR/" 2>/dev/null || echo "No .mp4 files found"

echo ""
echo "Done! Copied files:"
ls -lh "$DEST_DIR"
echo ""
echo "Total size:"
du -sh "$DEST_DIR"
echo ""
echo "⚠️  Remember to commit and deploy to Netlify for changes to take effect!"
