#!/usr/bin/env python3
"""Standalone script to find and kill orphaned worker processes.

This can be run manually to clean up any Isaac Sim or pose estimation workers
that were left running after the main program crashed or was killed.

Usage:
    python backend/cleanup_workers.py [--dry-run]

Options:
    --dry-run    Show what would be killed without actually killing
"""

import argparse
import os
import signal
import sys
import time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil not installed - install with: pip install psutil")
    sys.exit(1)


def find_orphaned_workers():
    """Find orphaned worker processes (Isaac Sim, pose workers).
    
    Returns:
        List of (pid, cmdline, cpu_percent, memory_mb) tuples
    """
    orphaned = []
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid', 'cpu_percent', 'memory_info']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if not cmdline:
                    continue
                
                cmdline_str = ' '.join(cmdline)
                
                # Look for Isaac Sim workers
                if 'isaac_sim_worker.py' in cmdline_str or 'persistent_isaac_sim_worker.py' in cmdline_str:
                    # Check if it's a zombie (parent is init or doesn't exist)
                    ppid = proc.info.get('ppid', 0)
                    is_orphan = False
                    
                    if ppid == 1:
                        is_orphan = True  # Parent is init (process was orphaned)
                    else:
                        # Check if parent still exists
                        try:
                            parent = psutil.Process(ppid)
                            parent_cmd = ' '.join(parent.cmdline())
                            # If parent is bash or similar shell, likely orphaned
                            if not ('collect_data.py' in parent_cmd or 'python' in parent.name()):
                                is_orphan = True
                        except psutil.NoSuchProcess:
                            is_orphan = True  # Parent doesn't exist
                    
                    if is_orphan:
                        mem_mb = proc.info.get('memory_info').rss / (1024 * 1024) if proc.info.get('memory_info') else 0
                        cpu = proc.cpu_percent(interval=0.1)
                        orphaned.append((proc.info['pid'], cmdline_str, cpu, mem_mb))
                        continue
                
                # Look for pose workers (any6d environment)
                if 'pose_worker.py' in cmdline_str and 'any6d' in cmdline_str:
                    ppid = proc.info.get('ppid', 0)
                    is_orphan = False
                    
                    if ppid == 1:
                        is_orphan = True
                    else:
                        try:
                            parent = psutil.Process(ppid)
                            parent_cmd = ' '.join(parent.cmdline())
                            if not ('collect_data.py' in parent_cmd or 'python' in parent.name()):
                                is_orphan = True
                        except psutil.NoSuchProcess:
                            is_orphan = True
                    
                    if is_orphan:
                        mem_mb = proc.info.get('memory_info').rss / (1024 * 1024) if proc.info.get('memory_info') else 0
                        cpu = proc.cpu_percent(interval=0.1)
                        orphaned.append((proc.info['pid'], cmdline_str, cpu, mem_mb))
            
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    
    except Exception as e:
        print(f"⚠️  Error scanning for orphaned workers: {e}")
    
    return orphaned


def kill_workers(worker_list, dry_run=False):
    """Kill the specified worker processes.
    
    Args:
        worker_list: List of (pid, cmdline, cpu, mem) tuples
        dry_run: If True, only show what would be killed
    """
    if not worker_list:
        print("✅ No orphaned workers found")
        return
    
    print(f"{'🔍' if dry_run else '🧹'} Found {len(worker_list)} orphaned worker process(es):")
    print()
    
    total_cpu = 0
    total_mem = 0
    
    for pid, cmdline, cpu, mem_mb in worker_list:
        # Truncate long command lines
        short_cmd = cmdline[:80] + '...' if len(cmdline) > 80 else cmdline
        
        # Extract worker type
        worker_type = "Isaac Sim" if "isaac_sim_worker" in cmdline else "Pose Worker"
        
        print(f"  PID {pid} ({worker_type}):")
        print(f"    CPU: {cpu:.1f}%  |  Memory: {mem_mb:.1f} MB")
        print(f"    Command: {short_cmd}")
        print()
        
        total_cpu += cpu
        total_mem += mem_mb
    
    print(f"📊 Total resource usage: CPU {total_cpu:.1f}%, Memory {total_mem:.1f} MB")
    print()
    
    if dry_run:
        print("🔍 Dry run mode - no processes killed")
        return
    
    # Kill them
    killed_count = 0
    for pid, cmdline, cpu, mem_mb in worker_list:
        worker_type = "Isaac Sim" if "isaac_sim_worker" in cmdline else "Pose Worker"
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
                print(f"✓ Terminated {worker_type} PID {pid}")
                killed_count += 1
            except psutil.TimeoutExpired:
                proc.kill()
                print(f"✓ Force killed {worker_type} PID {pid}")
                killed_count += 1
        except psutil.NoSuchProcess:
            print(f"✓ {worker_type} PID {pid} already gone")
            killed_count += 1
        except Exception as e:
            print(f"⚠️  Failed to kill {worker_type} PID {pid}: {e}")
    
    print()
    print(f"✅ Cleanup complete: {killed_count}/{len(worker_list)} workers killed")


def main():
    parser = argparse.ArgumentParser(
        description="Find and kill orphaned worker processes (Isaac Sim, pose workers)"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be killed without actually killing"
    )
    
    args = parser.parse_args()
    
    print("🔍 Scanning for orphaned worker processes...")
    print()
    
    orphaned = find_orphaned_workers()
    kill_workers(orphaned, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
