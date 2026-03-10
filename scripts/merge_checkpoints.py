#!/usr/bin/env python3
"""Merge specific episodes from multiple Phase 1 checkpoints into a single checkpoint.

Usage:
    # Extract episodes 0,2,4 from checkpoint A and episodes 1,3 from checkpoint B:
    python scripts/merge_checkpoints.py \
        --source /path/to/checkpointA.json --episodes 0 2 4 \
        --source /path/to/checkpointB.json --episodes 1 3 \
        --output /path/to/merged_checkpoint/

    # Extract ALL episodes from checkpoint A and specific ones from B:
    python scripts/merge_checkpoints.py \
        --source /path/to/checkpointA.json --episodes all \
        --source /path/to/checkpointB.json --episodes 0 5 7 \
        --output /path/to/merged_checkpoint/

    # Dry run (no files written):
    python scripts/merge_checkpoints.py \
        --source /path/to/checkpointA.json --episodes 0 1 \
        --dry-run

Then run Phase 2 on the merged checkpoint:
    ./scripts/run_phase2_only.sh /path/to/merged_checkpoint/phase1_checkpoint.json
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def parse_source_args(args: list[str]) -> list[tuple[Path, list[int] | None]]:
    """Parse --source / --episodes pairs from raw args.
    
    Returns list of (checkpoint_path, episode_list_or_None_for_all).
    """
    sources = []
    i = 0
    while i < len(args):
        if args[i] == "--source":
            if i + 1 >= len(args):
                print("❌ --source requires a path argument")
                sys.exit(1)
            cp_path = Path(args[i + 1])
            i += 2
            
            # Check for --episodes
            episodes = None  # None = all
            if i < len(args) and args[i] == "--episodes":
                i += 1
                ep_list = []
                while i < len(args) and not args[i].startswith("--"):
                    if args[i].lower() == "all":
                        episodes = None
                        i += 1
                        break
                    try:
                        ep_list.append(int(args[i]))
                    except ValueError:
                        print(f"❌ Invalid episode number: {args[i]}")
                        sys.exit(1)
                    i += 1
                if ep_list:
                    episodes = ep_list
            
            sources.append((cp_path, episodes))
        else:
            i += 1
    
    return sources


def load_checkpoint(path: Path) -> dict:
    """Load and validate a checkpoint JSON."""
    if not path.exists():
        print(f"❌ Checkpoint not found: {path}")
        sys.exit(1)
    
    with open(path) as f:
        cp = json.load(f)
    
    if cp.get("version") != 1:
        print(f"❌ Unsupported checkpoint version: {cp.get('version')} in {path}")
        sys.exit(1)
    
    return cp


def get_episodes_in_checkpoint(cp: dict) -> set[str]:
    """Get all episode keys present in a checkpoint."""
    episodes = set()
    for bucket in ["pending_states_by_episode", "completed_states_by_episode", "completed_states_buffer_by_episode"]:
        episodes.update(cp.get(bucket, {}).keys())
    return episodes


def remap_obs_paths(states: dict, src_obs_dir: Path, dst_obs_dir: Path, new_ep_id: int, dry_run: bool = False) -> tuple[dict, int, int]:
    """Update obs_path entries in states to point to merged obs_cache, and copy files.
    
    Returns (updated_states, files_copied, files_failed).
    """
    copied = 0
    failed = 0
    
    for sid, state in states.items():
        obs_path = state.get("obs_path")
        if obs_path:
            src_file = Path(obs_path)
            # Generate new filename with remapped episode ID
            new_filename = f"ep{new_ep_id}_state{state.get('state_id', sid)}.pt"
            dst_file = dst_obs_dir / new_filename
            
            if not dry_run:
                if src_file.exists():
                    try:
                        shutil.copy2(str(src_file), str(dst_file))
                        state["obs_path"] = str(dst_file)
                        copied += 1
                    except Exception as e:
                        print(f"  ⚠️  Failed to copy {src_file.name}: {e}")
                        failed += 1
                else:
                    print(f"  ⚠️  Source obs missing: {src_file}")
                    failed += 1
            else:
                state["obs_path"] = str(dst_file)
                copied += 1
        
        # Update episode_id in state
        state["episode_id"] = new_ep_id
    
    return states, copied, failed


def merge_checkpoints(sources: list[tuple[Path, list[int] | None]], output_dir: Path, dry_run: bool = False):
    """Merge episodes from multiple checkpoints into one."""
    
    print(f"{'='*60}")
    print(f"🔀 Merging Phase 1 Checkpoints")
    print(f"{'='*60}")
    
    # Validate all sources first
    checkpoints = []
    for cp_path, episodes in sources:
        cp = load_checkpoint(cp_path)
        available = get_episodes_in_checkpoint(cp)
        
        if episodes is None:
            selected = sorted(available, key=lambda x: int(x) if x.isdigit() else x)
            print(f"\n📂 {cp_path}")
            print(f"   Selecting ALL {len(selected)} episodes: {', '.join(selected)}")
        else:
            selected = [str(e) for e in episodes]
            missing = [e for e in selected if e not in available]
            if missing:
                print(f"\n❌ Episodes not found in {cp_path}: {missing}")
                print(f"   Available episodes: {sorted(available, key=lambda x: int(x) if x.isdigit() else x)}")
                sys.exit(1)
            print(f"\n📂 {cp_path}")
            print(f"   Selecting {len(selected)} episodes: {', '.join(selected)}")
        
        checkpoints.append((cp_path, cp, selected))
    
    # Count total
    total_episodes = sum(len(sel) for _, _, sel in checkpoints)
    print(f"\n📊 Total episodes to merge: {total_episodes}")
    
    if dry_run:
        print(f"\n🔍 DRY RUN — no files will be written\n")
    
    # Build merged checkpoint
    # Use first checkpoint's config as base
    base_cp = checkpoints[0][1]
    merged = {
        "version": 1,
        "saved_at": __import__("time").time(),
        "saved_at_iso": __import__("datetime").datetime.now().isoformat(),
        "dataset_config": base_cp.get("dataset_config"),
        "pending_states_by_episode": {},
        "completed_states_by_episode": {},
        "completed_states_buffer_by_episode": {},
        "episode_start_times": {},
        "episode_start_times_iso": {},
        "episodes_marked_as_end": [],
        "next_state_id": 0,
        "config": base_cp.get("config", {}),
    }
    
    # Prepare output
    output_dir = Path(output_dir)
    obs_cache_dir = output_dir / "obs_cache"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        obs_cache_dir.mkdir(parents=True, exist_ok=True)
    
    new_ep_id = 0
    max_state_id = 0
    total_copied = 0
    total_failed = 0
    
    for cp_path, cp, selected_eps in checkpoints:
        src_obs_dir = cp_path.parent / "obs_cache"
        
        for old_ep_key in selected_eps:
            new_ep_key = str(new_ep_id)
            
            for bucket in ["pending_states_by_episode", "completed_states_by_episode", "completed_states_buffer_by_episode"]:
                states = cp.get(bucket, {}).get(old_ep_key, {})
                if states:
                    # Deep copy to avoid mutating original
                    import copy
                    states_copy = copy.deepcopy(states)
                    
                    states_copy, copied, failed = remap_obs_paths(
                        states_copy, src_obs_dir, obs_cache_dir, new_ep_id, dry_run
                    )
                    total_copied += copied
                    total_failed += failed
                    
                    # Track max state_id for next_state_id
                    for sid_str in states_copy:
                        sid = int(sid_str) if isinstance(sid_str, str) and sid_str.isdigit() else 0
                        max_state_id = max(max_state_id, sid + 1)
                    
                    merged[bucket][new_ep_key] = states_copy
            
            # Copy episode metadata
            if old_ep_key in cp.get("episode_start_times", {}):
                merged["episode_start_times"][new_ep_key] = cp["episode_start_times"][old_ep_key]
            if old_ep_key in cp.get("episode_start_times_iso", {}):
                merged["episode_start_times_iso"][new_ep_key] = cp["episode_start_times_iso"][old_ep_key]
            if old_ep_key in [str(e) for e in cp.get("episodes_marked_as_end", [])] or \
               (old_ep_key.isdigit() and int(old_ep_key) in cp.get("episodes_marked_as_end", [])):
                merged["episodes_marked_as_end"].append(new_ep_id)
            
            src_label = f"ep {old_ep_key} from {cp_path.parent.name}"
            pending_count = len(merged["pending_states_by_episode"].get(new_ep_key, {}))
            completed_count = len(merged["completed_states_by_episode"].get(new_ep_key, {}))
            print(f"   ✅ {src_label} → merged ep {new_ep_id} ({pending_count} pending, {completed_count} completed)")
            
            new_ep_id += 1
    
    merged["next_state_id"] = max_state_id
    
    # Summary
    total_pending = sum(len(s) for s in merged["pending_states_by_episode"].values())
    total_completed = sum(len(s) for s in merged["completed_states_by_episode"].values())
    
    print(f"\n{'='*60}")
    print(f"📊 Merge Summary")
    print(f"   Episodes: {new_ep_id}")
    print(f"   Pending states: {total_pending}")
    print(f"   Completed states: {total_completed}")
    print(f"   Obs files: {total_copied} copied, {total_failed} failed")
    
    if dry_run:
        print(f"\n🔍 DRY RUN complete. No files were written.")
        print(f"   To execute, remove --dry-run")
        return
    
    # Write merged checkpoint
    output_path = output_dir / "phase1_checkpoint.json"
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    
    print(f"\n✅ Merged checkpoint saved to: {output_path}")
    print(f"   📁 obs_cache contains {len(list(obs_cache_dir.glob('*.pt')))} files")
    print(f"\n🚀 Run Phase 2:")
    print(f"   ./scripts/run_phase2_only.sh {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge episodes from multiple Phase 1 checkpoints",
        usage="%(prog)s --source <checkpoint.json> --episodes <ep_ids...> [--source ...] --output <dir> [--dry-run]",
    )
    parser.add_argument("--output", "-o", type=str, help="Output directory for merged checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be merged without writing files")
    
    # We parse --source/--episodes manually since argparse can't handle repeated grouped args well
    known, remaining = parser.parse_known_args()
    
    # Parse --source/--episodes from sys.argv (include both known and unknown)
    sources = parse_source_args(sys.argv[1:])
    
    if not sources:
        parser.print_help()
        print("\n❌ No --source arguments provided")
        sys.exit(1)
    
    if not known.output and not known.dry_run:
        print("❌ --output directory required (or use --dry-run)")
        sys.exit(1)
    
    output_dir = Path(known.output) if known.output else Path("/tmp/merge_preview")
    merge_checkpoints(sources, output_dir, dry_run=known.dry_run)


if __name__ == "__main__":
    main()
