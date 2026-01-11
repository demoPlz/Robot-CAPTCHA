#!/usr/bin/env python3
"""View and analyze user approval statistics from logs.

This script reads the user_approval_log.jsonl file and displays:
- Per-state approval statistics
- Per-user overall acceptance rates
- Episode summaries

Usage:
    python scripts/view_user_approval_stats.py <dataset_path>
    python scripts/view_user_approval_stats.py outputs/lerobot_datasets/my_dataset

"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_user_approval_log(dataset_path: Path):
    """Load and parse the user approval log."""
    # Look for log files matching the pattern user_approval_log_*.jsonl
    log_files = list(dataset_path.glob("user_approval_log_*.jsonl"))
    
    if not log_files:
        # Try old naming convention
        old_log_path = dataset_path / "user_approval_log.jsonl"
        if old_log_path.exists():
            log_files = [old_log_path]
        else:
            print(f"❌ No log files found in: {dataset_path}")
            print(f"   Looked for: user_approval_log_*.jsonl")
            print(f"   Make sure the dataset path is correct and episodes have been saved.")
            return []
    
    # Use the most recent log file if multiple exist
    log_path = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"📖 Reading log file: {log_path.name}")
    
    entries = []
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    return entries


def print_state_approvals(entries):
    """Print per-state approval information."""
    state_entries = [e for e in entries if e['type'] == 'state_approval']
    
    if not state_entries:
        print("\n📝 No state approval entries found.")
        return
    
    print("\n" + "="*80)
    print("📝 PER-STATE APPROVAL DETAILS")
    print("="*80)
    
    for entry in state_entries:
        ep = entry['episode_index']
        sid = entry['state_id']
        accepted = entry['accepted_users']
        rejected = entry['rejected_users']
        n_accepted = entry['num_accepted']
        n_rejected = entry['num_rejected']
        rate = entry['acceptance_rate']
        
        print(f"\n🔹 Episode {ep}, State {sid}:")
        print(f"   Accepted ({n_accepted}):")
        for user in accepted:
            print(f"      • {user['name']} ({user['email']})")
        
        print(f"   Rejected ({n_rejected}):")
        for user in rejected:
            print(f"      • {user['name']} ({user['email']})")
        
        if rate is not None:
            print(f"   Acceptance Rate: {n_accepted}/{n_accepted + n_rejected} ({rate:.1%})")


def print_episode_summaries(entries):
    """Print per-episode summary statistics."""
    episode_entries = [e for e in entries if e['type'] == 'episode_summary']
    
    if not episode_entries:
        print("\n📊 No episode summaries found.")
        return
    
    print("\n" + "="*80)
    print("📊 EPISODE SUMMARIES")
    print("="*80)
    
    for entry in episode_entries:
        ep = entry['episode_index']
        user_stats = entry['user_stats']
        overall_accepted = entry['overall_accepted']
        overall_rejected = entry['overall_rejected']
        overall_rate = entry['overall_acceptance_rate']
        
        print(f"\n🎬 Episode {ep}:")
        print(f"   Per-User Statistics:")
        for user in user_stats:
            name = user['name']
            email = user['email']
            accepted = user['accepted']
            total = user['total']
            rate = user['acceptance_rate']
            if rate is not None:
                print(f"      • {name} ({email}): {accepted}/{total} ({rate:.1%})")
            else:
                print(f"      • {name} ({email}): {accepted}/{total} (N/A)")
        
        if overall_rate is not None:
            print(f"   Overall: {overall_accepted}/{overall_accepted + overall_rejected} ({overall_rate:.1%})")


def print_overall_user_stats(entries):
    """Compute and print overall statistics across all episodes."""
    episode_entries = [e for e in entries if e['type'] == 'episode_summary']
    
    if not episode_entries:
        print("\n📈 No data for overall statistics.")
        return
    
    # Aggregate across all episodes
    user_totals = defaultdict(lambda: {'name': '', 'accepted': 0, 'rejected': 0})
    
    for entry in episode_entries:
        for user in entry['user_stats']:
            email = user['email']
            user_totals[email]['name'] = user['name']
            user_totals[email]['accepted'] += user['accepted']
            user_totals[email]['rejected'] += user['rejected']
    
    # Compute rates
    user_rates = []
    total_accepted = 0
    total_rejected = 0
    
    for email, stats in user_totals.items():
        total = stats['accepted'] + stats['rejected']
        rate = stats['accepted'] / total if total > 0 else None
        user_rates.append({
            'name': stats['name'],
            'email': email,
            'accepted': stats['accepted'],
            'rejected': stats['rejected'],
            'total': total,
            'rate': rate,
        })
        total_accepted += stats['accepted']
        total_rejected += stats['rejected']
    
    # Sort by acceptance rate (descending)
    user_rates.sort(key=lambda x: x['rate'] if x['rate'] is not None else 0, reverse=True)
    
    print("\n" + "="*80)
    print("📈 OVERALL USER STATISTICS (Across All Episodes)")
    print("="*80)
    
    print("\n🏆 Leaderboard (by acceptance rate):")
    for i, user in enumerate(user_rates, 1):
        name = user['name']
        email = user['email']
        accepted = user['accepted']
        total = user['total']
        rate = user['rate']
        
        if rate is not None:
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"   {emoji} #{i}: {name} ({email})")
            print(f"        {accepted}/{total} accepted ({rate:.1%})")
        else:
            print(f"      #{i}: {name} ({email})")
            print(f"        {accepted}/{total} (N/A)")
    
    # Overall acceptance rate
    overall_total = total_accepted + total_rejected
    if overall_total > 0:
        overall_rate = total_accepted / overall_total
        print(f"\n🌐 Overall Acceptance Rate: {total_accepted}/{overall_total} ({overall_rate:.1%})")
    else:
        print(f"\n🌐 Overall Acceptance Rate: No data")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/view_user_approval_stats.py <dataset_path>")
        print("Example: python scripts/view_user_approval_stats.py outputs/lerobot_datasets/my_dataset")
        sys.exit(1)
    
    dataset_path = Path(sys.argv[1])
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    print(f"📁 Loading user approval logs from: {dataset_path}")
    entries = load_user_approval_log(dataset_path)
    
    if not entries:
        print("❌ No log entries found.")
        sys.exit(1)
    
    print(f"✅ Loaded {len(entries)} log entries")
    
    # Print all sections
    print_state_approvals(entries)
    print_episode_summaries(entries)
    print_overall_user_stats(entries)
    
    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
