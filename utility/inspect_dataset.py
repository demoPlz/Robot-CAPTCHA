#!/usr/bin/env python
import pandas as pd
import json
from pathlib import Path
import sys

dataset_path = Path("/home/yilong/.cache/huggingface/lerobot/yilong/async_sess1_10_train")
output_file = Path("/home/yilong/crowdsourcing-ui/dataset_inspection.txt")

# Open output file
with open(output_file, 'w') as out:
    # Read episodes info
    with open(dataset_path / "meta" / "episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f]

    out.write(f"Total episodes: {len(episodes)}\n")
    out.write(f"Episodes: {episodes}\n\n")

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
        out.write(f"Columns: {df.columns.tolist()}\n")
        
        if 'action_type' in df.columns:
            out.write(f"\nAction type distribution:\n")
            out.write(f"{df['action_type'].value_counts().sort_index()}\n")
            
            out.write(f"\nFrame-by-frame breakdown:\n")
            for idx, row in df.iterrows():
                action_type = row['action_type'] if 'action_type' in row else 'N/A'
                # Convert action_type to int if it's a list
                if isinstance(action_type, (list, tuple)):
                    action_type = action_type[0] if len(action_type) > 0 else 'N/A'
                
                action_type_name = {0: 'non_critical', 1: 'critical_final', 2: 'critical_crowd', 3: 'critical_single_mode'}.get(action_type, f'unknown({action_type})')
                
                # Get full action values with 3 decimal places
                if 'action' in row:
                    action = row['action']
                    if hasattr(action, '__len__') and len(action) > 0:
                        action_list = action.tolist() if hasattr(action, 'tolist') else list(action)
                        action_str = '[' + ', '.join(f'{x:.3f}' for x in action_list) + ']'
                    else:
                        action_str = str(action)
                else:
                    action_str = 'N/A'
                
                out.write(f"  Frame {idx:3d}: action_type={action_type_name:25s} action={action_str}\n")
        else:
            out.write("\nNo 'action_type' column found in this episode!\n")
            out.write(f"First few rows:\n")
            out.write(f"{df.head()}\n")

print(f"Output written to: {output_file}")
