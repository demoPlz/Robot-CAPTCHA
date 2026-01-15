"""
Async Mode User Activity Logger

Tracks detailed metrics for each user during async mode data collection:
- Submission timing and duration
- Animation usage
- Approval/rejection rates
- Overall session metrics

Separate from sync mode logging.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime


class AsyncUserLogger:
    """Logger for async mode user submissions and statistics."""

    def __init__(self, output_dir: Path):
        """Initialize async user logger.
        
        Args:
            output_dir: Directory to write log files (typically dataset root)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.submission_log_path = self.output_dir / "async_user_submissions.jsonl"
        self.summary_log_path = self.output_dir / "async_user_summary.json"
        
        # In-memory tracking for summary generation
        self.user_stats: Dict[str, Dict] = {}  # email -> stats
        
        # Track when users first/last interacted (for wall clock time)
        self.user_first_activity: Dict[str, float] = {}  # email -> timestamp
        self.user_last_activity: Dict[str, float] = {}   # email -> timestamp
        
        print(f"📊 Async user logger initialized: {self.submission_log_path}")

    def log_submission(
        self,
        user_email: str,
        user_name: str,
        episode_id: int,
        state_id: int,
        duration_seconds: float,
        used_animation: bool,
        approval_status: Optional[int] = None,  # 1=approved, -1=rejected, None=pending
        current_approval_rate: Optional[float] = None,
        current_approval_count: Optional[int] = None,
        current_total_count: Optional[int] = None,
    ):
        """Log a single submission immediately after approval/rejection.
        
        Args:
            user_email: User's email
            user_name: User's name
            episode_id: Episode ID
            state_id: State ID
            duration_seconds: Time spent on this submission
            used_animation: Whether user clicked animation for this submission
            approval_status: 1 (approved), -1 (rejected), None (pending)
            current_approval_rate: User's approval rate so far (0.0-1.0)
            current_approval_count: Number of approved submissions so far
            current_total_count: Total number of reviewed submissions so far
        """
        timestamp = time.time()
        timestamp_iso = datetime.now().isoformat()
        
        # Update user activity tracking
        if user_email not in self.user_first_activity:
            self.user_first_activity[user_email] = timestamp
        self.user_last_activity[user_email] = timestamp
        
        # Update in-memory stats
        if user_email not in self.user_stats:
            self.user_stats[user_email] = {
                "name": user_name,
                "email": user_email,
                "submissions": [],
                "approved_count": 0,
                "rejected_count": 0,
                "pending_count": 0,
                "total_duration_seconds": 0.0,
                "animation_usage_count": 0,
                "total_submissions": 0,
            }
        
        stats = self.user_stats[user_email]
        stats["total_submissions"] += 1
        stats["total_duration_seconds"] += duration_seconds
        
        if used_animation:
            stats["animation_usage_count"] += 1
        
        if approval_status == 1:
            stats["approved_count"] += 1
        elif approval_status == -1:
            stats["rejected_count"] += 1
        else:
            stats["pending_count"] += 1
        
        # Log entry
        log_entry = {
            "type": "async_user_submission",
            "timestamp": timestamp,
            "timestamp_iso": timestamp_iso,
            "user_name": user_name,
            "user_email": user_email,
            "episode_id": episode_id,
            "state_id": state_id,
            "duration_seconds": round(duration_seconds, 2),
            "used_animation": used_animation,
            "approval_status": approval_status,  # 1=approved, -1=rejected, None=pending
            "approval_status_str": (
                "approved" if approval_status == 1 else
                "rejected" if approval_status == -1 else
                "pending"
            ),
            "current_approval_rate": round(current_approval_rate, 3) if current_approval_rate is not None else None,
            "current_approval_count": current_approval_count,
            "current_total_count": current_total_count,
        }
        
        # Append to log file
        with open(self.submission_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Console output
        status_emoji = "✅" if approval_status == 1 else "❌" if approval_status == -1 else "⏳"
        approval_str = f"{current_approval_count}/{current_total_count} ({current_approval_rate*100:.1f}%)" if current_approval_rate is not None else "N/A"
        anim_str = "🎬 yes" if used_animation else "no"
        
        print(f"{status_emoji} Async submission logged: {user_name} ({user_email})")
        print(f"   State: ep={episode_id}, state={state_id} | Duration: {duration_seconds:.1f}s | Animation: {anim_str}")
        print(f"   Approval rate: {approval_str}")

    def generate_final_summary(self):
        """Generate and save final summary of all users' performance.
        
        Call this after async data collection is complete.
        """
        if not self.user_stats:
            print("⚠️  No async user submissions to summarize")
            return
        
        # Compute per-user summary
        user_summaries = []
        
        for email, stats in self.user_stats.items():
            reviewed_count = stats["approved_count"] + stats["rejected_count"]
            approval_rate = stats["approved_count"] / reviewed_count if reviewed_count > 0 else None
            
            avg_duration = (
                stats["total_duration_seconds"] / stats["total_submissions"]
                if stats["total_submissions"] > 0
                else 0.0
            )
            
            animation_usage_rate = (
                stats["animation_usage_count"] / stats["total_submissions"]
                if stats["total_submissions"] > 0
                else 0.0
            )
            
            # Wall clock time (first to last activity)
            wall_clock_seconds = None
            if email in self.user_first_activity and email in self.user_last_activity:
                wall_clock_seconds = self.user_last_activity[email] - self.user_first_activity[email]
            
            user_summary = {
                "name": stats["name"],
                "email": email,
                "total_submissions": stats["total_submissions"],
                "approved": stats["approved_count"],
                "rejected": stats["rejected_count"],
                "pending": stats["pending_count"],
                "reviewed_count": reviewed_count,
                "approval_rate": round(approval_rate, 3) if approval_rate is not None else None,
                "approval_rate_percent": round(approval_rate * 100, 1) if approval_rate is not None else None,
                "average_duration_seconds": round(avg_duration, 2),
                "animation_usage_count": stats["animation_usage_count"],
                "animation_usage_rate": round(animation_usage_rate, 3),
                "animation_usage_percent": round(animation_usage_rate * 100, 1),
                "wall_clock_seconds": round(wall_clock_seconds, 1) if wall_clock_seconds is not None else None,
                "wall_clock_minutes": round(wall_clock_seconds / 60, 1) if wall_clock_seconds is not None else None,
                "first_activity_iso": datetime.fromtimestamp(self.user_first_activity[email]).isoformat() if email in self.user_first_activity else None,
                "last_activity_iso": datetime.fromtimestamp(self.user_last_activity[email]).isoformat() if email in self.user_last_activity else None,
            }
            
            user_summaries.append(user_summary)
        
        # Sort by approval rate (descending)
        user_summaries.sort(key=lambda x: x["approval_rate"] if x["approval_rate"] is not None else 0, reverse=True)
        
        # Overall aggregated stats
        total_submissions = sum(s["total_submissions"] for s in user_summaries)
        total_approved = sum(s["approved"] for s in user_summaries)
        total_rejected = sum(s["rejected"] for s in user_summaries)
        total_reviewed = total_approved + total_rejected
        overall_approval_rate = total_approved / total_reviewed if total_reviewed > 0 else None
        
        summary = {
            "type": "async_user_summary",
            "timestamp": time.time(),
            "timestamp_iso": datetime.now().isoformat(),
            "total_users": len(user_summaries),
            "total_submissions": total_submissions,
            "total_approved": total_approved,
            "total_rejected": total_rejected,
            "total_reviewed": total_reviewed,
            "overall_approval_rate": round(overall_approval_rate, 3) if overall_approval_rate is not None else None,
            "overall_approval_rate_percent": round(overall_approval_rate * 100, 1) if overall_approval_rate is not None else None,
            "users": user_summaries,
        }
        
        # Write summary to file
        with open(self.summary_log_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*80)
        print("📊 ASYNC MODE USER SUMMARY (Final)")
        print("="*80)
        print(f"\n👥 Total users: {len(user_summaries)}")
        print(f"📝 Total submissions: {total_submissions}")
        print(f"✅ Approved: {total_approved} | ❌ Rejected: {total_rejected}")
        if overall_approval_rate is not None:
            print(f"📈 Overall approval rate: {total_approved}/{total_reviewed} ({overall_approval_rate*100:.1f}%)")
        
        print("\n🏆 User Leaderboard (by approval rate):")
        print("-" * 80)
        
        for i, user in enumerate(user_summaries, 1):
            name = user["name"]
            email = user["email"]
            approved = user["approved"]
            reviewed = user["reviewed_count"]
            rate = user["approval_rate_percent"]
            avg_time = user["average_duration_seconds"]
            anim_rate = user["animation_usage_percent"]
            wall_clock = user["wall_clock_minutes"]
            
            rate_str = f"{rate:.1f}%" if rate is not None else "N/A"
            wall_str = f"{wall_clock:.1f} min" if wall_clock is not None else "N/A"
            
            print(f"{i:2d}. {name} ({email})")
            print(f"    ✅ Approval: {approved}/{reviewed} ({rate_str})")
            print(f"    ⏱️  Avg time: {avg_time:.1f}s | 🎬 Animation: {anim_rate:.1f}% | 🕐 Wall clock: {wall_str}")
        
        print("="*80)
        print(f"📁 Summary saved to: {self.summary_log_path}")
        print("="*80)

    def track_user_activity_start(self, user_email: str):
        """Track when user first becomes active (loads first state).
        
        Args:
            user_email: User's email
        """
        if user_email not in self.user_first_activity:
            self.user_first_activity[user_email] = time.time()
            print(f"🕐 Started tracking wall clock time for {user_email}")

    def track_user_activity_end(self, user_email: str):
        """Track when user finishes (submits last state).
        
        Args:
            user_email: User's email
        """
        self.user_last_activity[user_email] = time.time()
