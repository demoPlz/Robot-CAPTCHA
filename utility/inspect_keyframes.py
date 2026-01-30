#!/usr/bin/env python
"""Inspect dataset and visualize keyframes"""
import pandas as pd
import json
from pathlib import Path
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_keyframes.py <dataset_path>")
    sys.exit(1)

dataset_path = Path(sys.argv[1])
dataset_name = dataset_path.name
output_file = Path(f"/home/yilong/crowdsourcing-ui/keyframe_inspection_{dataset_name}.txt")

# Open output file
with open(output_file, 'w') as out:
    # Read episodes info
    with open(dataset_path / "meta" / "episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f]

    out.write(f"Dataset: {dataset_name}\n")
    out.write(f"Total episodes: {len(episodes)}\n")
    out.write(f"{'='*80}\n\n")

    # Read each episode's parquet file
    for ep_idx, episode in enumerate(episodes):
        parquet_file = dataset_path / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        
        if not parquet_file.exists():
            out.write(f"Episode {ep_idx}: File not found\n")
            continue
        
        df = pd.read_parquet(parquet_file)
        
        out.write(f"\n{'='*80}\n")
        out.write(f"EPISODE {ep_idx}\n")
        out.write(f"{'='*80}\n")
        out.write(f"Total frames: {len(df)}\n")
        out.write(f"Columns: {df.columns.tolist()}\n\n")
        
        # Check for frame_type column
        if 'frame_type' in df.columns:
            # Count keyframes
            keyframe_count = 0
            keyframe_indices = []
            
            for idx, row in df.iterrows():
                frame_type = row['frame_type']
                # Convert to int if it's a list/array
                if isinstance(frame_type, (list, tuple)):
                    frame_type = frame_type[0] if len(frame_type) > 0 else 0
                
                if frame_type == 1:
                    keyframe_count += 1
                    keyframe_indices.append(idx)
            
            out.write(f"Keyframes: {keyframe_count} / {len(df)} frames ({100*keyframe_count/len(df):.1f}%)\n")
            out.write(f"Keyframe indices: {keyframe_indices}\n\n")
            
            # Show detailed view around keyframes
            out.write("KEYFRAME DETAILS:\n")
            out.write("-" * 80 + "\n")
            
            for kf_idx in keyframe_indices:
                # Show context: 2 frames before and after
                start_idx = max(0, kf_idx - 2)
                end_idx = min(len(df), kf_idx + 3)
                
                out.write(f"\nKeyframe at frame {kf_idx} (context {start_idx}-{end_idx-1}):\n")
                
                for idx in range(start_idx, end_idx):
                    row = df.iloc[idx]
                    frame_type = row['frame_type']
                    if isinstance(frame_type, (list, tuple)):
                        frame_type = frame_type[0] if len(frame_type) > 0 else 0
                    
                    marker = " <-- KEYFRAME" if frame_type == 1 else ""
                    
                    # Get action values
                    if 'action' in row:
                        action = row['action']
                        if hasattr(action, '__len__') and len(action) > 0:
                            action_list = action.tolist() if hasattr(action, 'tolist') else list(action)
                            action_str = '[' + ', '.join(f'{x:7.4f}' for x in action_list) + ']'
                        else:
                            action_str = str(action)
                    else:
                        action_str = 'N/A'
                    
                    timestamp = row['timestamp'] if 'timestamp' in row else 'N/A'
                    out.write(f"  Frame {idx:4d}: frame_type={frame_type}  t={timestamp:6.2f}s  action={action_str}{marker}\n")
        else:
            out.write("WARNING: No 'frame_type' column found in this episode!\n")
            out.write("This dataset may not have been recorded with keyframe marking enabled.\n")
            out.write(f"Available columns: {df.columns.tolist()}\n")

print(f"Keyframe analysis written to: {output_file}")
