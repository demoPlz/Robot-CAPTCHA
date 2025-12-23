#!/usr/bin/env python3
"""Visualize crowdsourced dataset as a spreadsheet.

Displays dataset frames with action selection metadata in a readable table format.
Shows executed actions, propensities, approvals, and final executed actions.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "lerobot_trossen"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def load_dataset(repo_id: str, root: str = "data") -> LeRobotDataset:
    """Load a LeRobot dataset."""
    print(f"Loading dataset: {repo_id}")
    dataset = LeRobotDataset(repo_id, root=root)
    print(f"Loaded {len(dataset)} frames across {dataset.meta.total_episodes} episodes")
    return dataset


def extract_frame_data(dataset: LeRobotDataset, frame_idx: int) -> dict:
    """Extract relevant data from a single frame."""
    frame = dataset[frame_idx]
    
    # Basic info
    episode_idx = dataset.episode_data_index["episode_index"][frame_idx].item()
    frame_in_episode = dataset.episode_data_index["frame_index"][frame_idx].item()
    task = frame.get("task", [""])[0] if "task" in frame else ""
    
    # Action data
    action = frame.get("action", None)
    action_str = _format_tensor(action) if action is not None else "N/A"
    
    # Execution history
    executed_actions = frame.get("executed_actions", None)
    executed_propensities = frame.get("executed_propensities", None)
    executed_approvals = frame.get("executed_approvals", None)
    final_executed_action = frame.get("final_executed_action", None)
    
    # Count executions (non-NaN propensities)
    num_executions = 0
    if executed_propensities is not None:
        propensities = executed_propensities.numpy() if hasattr(executed_propensities, "numpy") else executed_propensities
        num_executions = int(np.sum(~np.isnan(propensities)))
    
    # Format execution details
    execution_details = []
    if executed_actions is not None and executed_propensities is not None:
        actions_arr = executed_actions.numpy() if hasattr(executed_actions, "numpy") else executed_actions
        propensities_arr = executed_propensities.numpy() if hasattr(executed_propensities, "numpy") else executed_propensities
        approvals_arr = executed_approvals.numpy() if hasattr(executed_approvals, "numpy") else executed_approvals
        
        # Determine action dimension
        total_len = len(actions_arr)
        num_slots = len(propensities_arr)
        action_dim = total_len // num_slots if num_slots > 0 else 0
        
        for i in range(num_slots):
            if not np.isnan(propensities_arr[i]):
                start_idx = i * action_dim
                end_idx = start_idx + action_dim
                action_vec = actions_arr[start_idx:end_idx]
                
                approval_val = approvals_arr[i] if i < len(approvals_arr) else np.nan
                if np.isnan(approval_val):
                    approval_str = "pending"
                elif approval_val > 0:
                    approval_str = "approved"
                else:
                    approval_str = "rejected"
                
                execution_details.append({
                    "exec_idx": i,
                    "action": _format_tensor(action_vec, max_vals=3),
                    "propensity": f"{propensities_arr[i]:.4f}",
                    "approval": approval_str
                })
    
    return {
        "frame_idx": frame_idx,
        "episode": episode_idx,
        "frame_in_ep": frame_in_episode,
        "task": task,
        "action": action_str,
        "num_executions": num_executions,
        "execution_details": execution_details,
        "final_action": _format_tensor(final_executed_action, max_vals=3) if final_executed_action is not None else "N/A"
    }


def _format_tensor(tensor, max_vals: int = 5) -> str:
    """Format a tensor for display."""
    if tensor is None:
        return "N/A"
    
    arr = tensor.numpy() if hasattr(tensor, "numpy") else np.array(tensor)
    
    # Check if all NaN
    if np.all(np.isnan(arr)):
        return "N/A"
    
    # Format values
    if len(arr) <= max_vals:
        vals = ", ".join(f"{v:.3f}" for v in arr)
    else:
        vals = ", ".join(f"{v:.3f}" for v in arr[:max_vals]) + "..."
    
    return f"[{vals}]"


def create_summary_dataframe(dataset: LeRobotDataset, max_frames: int = None) -> pd.DataFrame:
    """Create a summary dataframe of the dataset."""
    num_frames = len(dataset) if max_frames is None else min(len(dataset), max_frames)
    
    rows = []
    for frame_idx in range(num_frames):
        data = extract_frame_data(dataset, frame_idx)
        
        # Create base row
        row = {
            "Frame": data["frame_idx"],
            "Episode": data["episode"],
            "Frame_in_Ep": data["frame_in_ep"],
            "Task": data["task"][:50] + "..." if len(data["task"]) > 50 else data["task"],
            "Num_Executions": data["num_executions"],
            "Final_Action": data["final_action"],
        }
        
        # Add execution details as separate columns
        for exec_detail in data["execution_details"][:3]:  # Show up to 3 executions
            exec_idx = exec_detail["exec_idx"]
            row[f"Exec{exec_idx}_Action"] = exec_detail["action"]
            row[f"Exec{exec_idx}_Prop"] = exec_detail["propensity"]
            row[f"Exec{exec_idx}_Approval"] = exec_detail["approval"]
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def display_frame_details(dataset: LeRobotDataset, frame_idx: int):
    """Display detailed information about a specific frame."""
    data = extract_frame_data(dataset, frame_idx)
    
    print("\n" + "="*80)
    print(f"Frame {data['frame_idx']} Details")
    print("="*80)
    print(f"Episode: {data['episode']}")
    print(f"Frame in Episode: {data['frame_in_ep']}")
    print(f"Task: {data['task']}")
    print(f"\nFinal Action: {data['final_action']}")
    print(f"\nNumber of Executions: {data['num_executions']}")
    
    if data['execution_details']:
        print("\nExecution History:")
        print("-" * 80)
        for exec_detail in data['execution_details']:
            print(f"  Execution {exec_detail['exec_idx']}:")
            print(f"    Action:     {exec_detail['action']}")
            print(f"    Propensity: {exec_detail['propensity']}")
            print(f"    Approval:   {exec_detail['approval']}")
    else:
        print("\nNo execution history recorded")
    
    print("="*80 + "\n")


def export_to_csv(df: pd.DataFrame, output_path: str):
    """Export dataframe to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize crowdsourced LeRobot dataset")
    parser.add_argument("repo_id", help="Dataset repository ID (e.g., username/dataset_name)")
    parser.add_argument("--root", default="data", help="Root directory for datasets (default: data)")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to display")
    parser.add_argument("--frame", type=int, help="Show detailed view of specific frame")
    parser.add_argument("--export", help="Export to CSV file")
    parser.add_argument("--episode", type=int, help="Show only frames from specific episode")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args.repo_id, args.root)
    
    # Check if dataset has crowd fields
    has_crowd_fields = (
        "executed_actions" in dataset.features or
        "executed_propensities" in dataset.features or
        "final_executed_action" in dataset.features
    )
    
    if not has_crowd_fields:
        print("\nWARNING: This dataset does not appear to have crowd annotation fields.")
        print("It may have been created without the crowd interface.")
        print("Continuing with standard fields...\n")
    
    # Show specific frame details
    if args.frame is not None:
        if args.frame >= len(dataset):
            print(f"Error: Frame {args.frame} does not exist (dataset has {len(dataset)} frames)")
            return
        display_frame_details(dataset, args.frame)
        return
    
    # Create summary dataframe
    print("\nGenerating dataset summary...")
    df = create_summary_dataframe(dataset, args.max_frames)
    
    # Filter by episode if requested
    if args.episode is not None:
        df = df[df["Episode"] == args.episode]
        print(f"\nFiltered to episode {args.episode}: {len(df)} frames")
    
    # Display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 30)
    
    print("\n" + "="*80)
    print("Dataset Summary")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Statistics
    print("\nStatistics:")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Episodes: {dataset.meta.total_episodes}")
    print(f"  Frames shown: {len(df)}")
    
    if "Num_Executions" in df.columns:
        print(f"  Avg executions per frame: {df['Num_Executions'].mean():.2f}")
        print(f"  Max executions: {df['Num_Executions'].max()}")
        print(f"  Frames with executions: {(df['Num_Executions'] > 0).sum()}")
    
    # Export if requested
    if args.export:
        export_to_csv(df, args.export)
    
    print("\nTip: Use --frame <idx> to see detailed execution history for a specific frame")


if __name__ == "__main__":
    main()
