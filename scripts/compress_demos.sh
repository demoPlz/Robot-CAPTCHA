#!/bin/bash
# Compress demo videos for fast loading (target: ~1-2MB per video)

SOURCE_DIR="data/prompts/drawer/demos"
DEST_DIR="public/demos"

echo "Compressing demo videos..."
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo ""

mkdir -p "$DEST_DIR"

for video in "$SOURCE_DIR"/*.webm; do
    filename=$(basename "$video")
    output="$DEST_DIR/$filename"
    
    echo "Processing $filename..."
    
    # Aggressive compression for instant loading:
    # - Scale to 480p (enough for demo reference)
    # - Very low bitrate (200k video + 32k audio)
    # - Fast encoding
    ffmpeg -i "$video" \
        -vf "scale=480:-2" \
        -c:v libvpx-vp9 \
        -b:v 200k \
        -c:a libopus \
        -b:a 32k \
        -deadline realtime \
        -cpu-used 4 \
        -y "$output" 2>&1 | grep -E "Duration|time=|size=" | tail -3
    
    # Show size comparison
    original_size=$(du -h "$video" | cut -f1)
    compressed_size=$(du -h "$output" | cut -f1)
    echo "  $original_size → $compressed_size"
    echo ""
done

echo "Done! Total size:"
du -sh "$DEST_DIR"
