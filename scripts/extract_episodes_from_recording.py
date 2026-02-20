#!/usr/bin/env python3
"""Extract individual episode clips from continuous benchmark recordings.

Uses the event log JSON to identify episode boundaries and extracts clips
from the continuous video chunks using ffmpeg.

Usage:
    python scripts/extract_episodes_from_recording.py <recording_dir> [--output-dir OUTPUT_DIR] [--episodes 0,1,2]
    
Examples:
    # Extract all episodes
    python scripts/extract_episodes_from_recording.py ~/.cache/huggingface/lerobot/benchmark_recordings/eval_test_logging
    
    # Extract specific episodes
    python scripts/extract_episodes_from_recording.py ~/.cache/huggingface/lerobot/benchmark_recordings/eval_test_logging --episodes 0,5,10
    
    # Specify output directory
    python scripts/extract_episodes_from_recording.py ~/.cache/huggingface/lerobot/benchmark_recordings/eval_test_logging --output-dir ./extracted_clips
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class EpisodeBoundary:
    """Episode start and end information."""
    episode_index: int
    start_frame: int
    end_frame: int
    start_chunk: int
    end_chunk: int
    start_chunk_frame: int
    end_chunk_frame: int
    start_timestamp: float
    end_timestamp: float


def load_event_log(recording_dir: Path) -> dict:
    """Load the event log JSON file from recording directory."""
    json_files = list(recording_dir.glob("*_events.json"))
    if not json_files:
        raise FileNotFoundError(f"No event log found in {recording_dir}")
    if len(json_files) > 1:
        # Use the most recent one
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        print(f"Warning: Multiple event logs found, using: {json_files[0].name}")
    
    with open(json_files[0]) as f:
        return json.load(f)


def get_chunk_files(recording_dir: Path, session_start: str) -> list[Path]:
    """Get sorted list of chunk video files."""
    # Match pattern: <session_name>_<session_start>_chunk####.mp4
    chunks = sorted(recording_dir.glob(f"*_{session_start}_chunk*.mp4"))
    return chunks


def parse_episode_boundaries(events: list[dict]) -> list[EpisodeBoundary]:
    """Parse events to find episode start/end boundaries."""
    episodes = {}
    
    for event in events:
        event_type = event["event_type"]
        episode_idx = event.get("episode_index")
        
        if episode_idx is None:
            continue
        
        if episode_idx not in episodes:
            episodes[episode_idx] = {"start": None, "end": None}
        
        if event_type == "episode_start":
            episodes[episode_idx]["start"] = event
        elif event_type == "episode_end":
            episodes[episode_idx]["end"] = event
    
    boundaries = []
    for ep_idx in sorted(episodes.keys()):
        ep_data = episodes[ep_idx]
        if ep_data["start"] is None or ep_data["end"] is None:
            print(f"Warning: Episode {ep_idx} has incomplete boundaries, skipping")
            continue
        
        start = ep_data["start"]
        end = ep_data["end"]
        
        boundaries.append(EpisodeBoundary(
            episode_index=ep_idx,
            start_frame=start["frame_number"],
            end_frame=end["frame_number"],
            start_chunk=start["chunk_index"],
            end_chunk=end["chunk_index"],
            start_chunk_frame=start["chunk_frame_number"],
            end_chunk_frame=end["chunk_frame_number"],
            start_timestamp=start["elapsed_seconds"],
            end_timestamp=end["elapsed_seconds"],
        ))
    
    return boundaries


def get_chunk_frame_counts(chunk_files: list[Path]) -> list[int]:
    """Get frame count for each chunk using ffprobe."""
    counts = []
    for chunk_path in chunk_files:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-count_frames",
                    "-show_entries", "stream=nb_read_frames",
                    "-of", "csv=p=0",
                    str(chunk_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            count = int(result.stdout.strip())
            counts.append(count)
        except Exception as e:
            print(f"Warning: Could not get frame count for {chunk_path}: {e}")
            counts.append(0)
    return counts


def extract_episode_ffmpeg(
    episode: EpisodeBoundary,
    chunk_files: list[Path],
    chunk_frame_counts: list[int],
    fps: int,
    output_path: Path,
) -> bool:
    """Extract episode using ffmpeg.
    
    Handles episodes that span multiple chunks by:
    1. If single chunk: direct extraction
    2. If multiple chunks: extract from each, then concatenate
    """
    
    if episode.start_chunk == episode.end_chunk:
        # Simple case: episode is within one chunk
        chunk_path = chunk_files[episode.start_chunk]
        start_time = episode.start_chunk_frame / fps
        duration = (episode.end_chunk_frame - episode.start_chunk_frame) / fps
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", str(chunk_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error extracting episode {episode.episode_index}: {e.stderr.decode()}")
            return False
    
    else:
        # Episode spans multiple chunks - need to extract and concatenate
        temp_files = []
        concat_list_path = output_path.parent / f"_concat_ep{episode.episode_index}.txt"
        
        try:
            for chunk_idx in range(episode.start_chunk, episode.end_chunk + 1):
                chunk_path = chunk_files[chunk_idx]
                temp_path = output_path.parent / f"_temp_ep{episode.episode_index}_chunk{chunk_idx}.mp4"
                temp_files.append(temp_path)
                
                if chunk_idx == episode.start_chunk:
                    # First chunk: start from start_chunk_frame to end
                    start_time = episode.start_chunk_frame / fps
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", str(start_time),
                        "-i", str(chunk_path),
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "23",
                        str(temp_path)
                    ]
                elif chunk_idx == episode.end_chunk:
                    # Last chunk: from beginning to end_chunk_frame
                    duration = episode.end_chunk_frame / fps
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(chunk_path),
                        "-t", str(duration),
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "23",
                        str(temp_path)
                    ]
                else:
                    # Middle chunk: use entire file
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(chunk_path),
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "23",
                        str(temp_path)
                    ]
                
                subprocess.run(cmd, capture_output=True, check=True)
            
            # Create concat file
            with open(concat_list_path, "w") as f:
                for temp_path in temp_files:
                    f.write(f"file '{temp_path}'\n")
            
            # Concatenate
            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_path),
                "-c", "copy",
                str(output_path)
            ]
            subprocess.run(concat_cmd, capture_output=True, check=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error extracting episode {episode.episode_index}: {e.stderr.decode() if e.stderr else e}")
            return False
            
        finally:
            # Cleanup temp files
            for temp_path in temp_files:
                if temp_path.exists():
                    temp_path.unlink()
            if concat_list_path.exists():
                concat_list_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Extract episode clips from continuous benchmark recordings"
    )
    parser.add_argument(
        "recording_dir",
        type=Path,
        help="Directory containing continuous recording chunks and event log"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for extracted clips (default: <recording_dir>/extracted)"
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Comma-separated list of episode indices to extract (default: all)"
    )
    parser.add_argument(
        "--include-resets",
        action="store_true",
        help="Also extract reset periods between episodes"
    )
    
    args = parser.parse_args()
    
    recording_dir = args.recording_dir.expanduser().resolve()
    if not recording_dir.exists():
        print(f"Error: Recording directory not found: {recording_dir}")
        sys.exit(1)
    
    # Load event log
    print(f"Loading event log from {recording_dir}...")
    try:
        log_data = load_event_log(recording_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    session_start = log_data["session_start"]
    fps = log_data["recording_fps"]
    events = log_data["events"]
    
    print(f"Session: {log_data['session_name']}")
    print(f"FPS: {fps}, Resolution: {log_data['resolution']}")
    print(f"Total frames: {log_data['total_frames']}, Chunks: {log_data['total_chunks']}")
    
    # Get chunk files
    chunk_files = get_chunk_files(recording_dir, session_start)
    if not chunk_files:
        print(f"Error: No chunk files found for session {session_start}")
        sys.exit(1)
    
    print(f"Found {len(chunk_files)} chunk files")
    
    # Parse episode boundaries
    boundaries = parse_episode_boundaries(events)
    print(f"Found {len(boundaries)} episodes")
    
    # Filter episodes if specified
    if args.episodes:
        selected = set(int(x.strip()) for x in args.episodes.split(","))
        boundaries = [b for b in boundaries if b.episode_index in selected]
        print(f"Extracting {len(boundaries)} selected episodes: {sorted(selected)}")
    
    if not boundaries:
        print("No episodes to extract")
        sys.exit(0)
    
    # Setup output directory
    output_dir = args.output_dir or recording_dir / "extracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get chunk frame counts (for multi-chunk episodes)
    print("Analyzing chunk files...")
    chunk_frame_counts = get_chunk_frame_counts(chunk_files)
    
    # Extract each episode
    print("\nExtracting episodes...")
    success_count = 0
    for boundary in boundaries:
        output_path = output_dir / f"episode_{boundary.episode_index:04d}.mp4"
        duration_s = boundary.end_timestamp - boundary.start_timestamp
        
        print(f"  Episode {boundary.episode_index}: frames {boundary.start_frame}-{boundary.end_frame} "
              f"({duration_s:.1f}s, chunks {boundary.start_chunk}-{boundary.end_chunk})")
        
        if extract_episode_ffmpeg(boundary, chunk_files, chunk_frame_counts, fps, output_path):
            print(f"    -> {output_path.name}")
            success_count += 1
        else:
            print(f"    -> FAILED")
    
    print(f"\nDone! Extracted {success_count}/{len(boundaries)} episodes to {output_dir}")


if __name__ == "__main__":
    main()
