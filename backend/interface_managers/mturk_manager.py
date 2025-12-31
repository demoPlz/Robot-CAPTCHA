"""MTurk Manager Module.

Manages Amazon Mechanical Turk HIT creation and lifecycle for crowdsourced robot data collection.
Creates HITs when critical states need labeling, tracks HIT status, and auto-approves submissions.
"""

import json
import time
from pathlib import Path
from threading import Lock, Thread
from typing import Optional

import boto3
from botocore.exceptions import ClientError


class MTurkManager:
    """Manages MTurk HIT creation and lifecycle for critical states.

    Responsibilities:
    - Create HITs for critical states that need labeling
    - Track HIT IDs and their associated states
    - Auto-approve all submitted assignments
    - Clean up expired/completed HITs
    - Provide HIT status monitoring

    Attributes:
        mturk_client: Boto3 MTurk client
        hit_tracking: Maps (episode_id, state_id) -> HIT metadata
        hit_lock: Lock protecting HIT tracking data
        sandbox: Whether using sandbox environment

    """

    def __init__(
        self,
        sandbox: bool = True,
        reward: float = 0.50,
        assignment_duration_seconds: int = 600,
        lifetime_seconds: int = 3600,
        auto_approval_delay_seconds: int = 60,
        title: str = "Control a robot arm to complete a manipulation task",
        description: str = "View a robot simulation and specify the next position for the robot to move to",
        keywords: str = "robot, manipulation, annotation, simulation",
        external_url: Optional[str] = None,
        num_expert_workers: int = 0,
        required_responses_per_critical_state: int = 2,
        output_dir: str = "output/mturk_stats",
        require_masters: bool = False,
        min_approval_rate: int = 95,
        min_approved_hits: int = 100,
        require_location: Optional[list[str]] = None,
    ):
        """Initialize MTurk manager.

        Args:
            sandbox: Use MTurk sandbox (True) or production (False)
            reward: Payment per assignment in USD
            assignment_duration_seconds: Time allowed per assignment
            lifetime_seconds: How long HIT remains available
            auto_approval_delay_seconds: Auto-approve delay (minimum 60 seconds)
            title: HIT title shown to workers
            description: HIT description
            keywords: Search keywords
            external_url: Public URL where frontend is hosted (e.g., cloudflare tunnel)
            num_expert_workers: Number of expert workers labeling via localhost (reduces MTurk assignments)
            required_responses_per_critical_state: Total responses needed per critical state
            output_dir: Directory to save timing statistics
            require_masters: Require MTurk Masters qualification (premium workers, higher quality)
            min_approval_rate: Minimum approval rate percentage (0-100, default 95)
            min_approved_hits: Minimum number of approved HITs (default 100)
            require_location: List of country codes to restrict workers (e.g., ['US', 'CA', 'GB'])

        """
        self.sandbox = sandbox
        self.reward = reward
        self.assignment_duration_seconds = assignment_duration_seconds
        self.lifetime_seconds = lifetime_seconds
        self.auto_approval_delay_seconds = max(60, auto_approval_delay_seconds)  # Minimum 60s
        self.title = title
        self.description = description
        self.keywords = keywords
        self.external_url = external_url
        self.num_expert_workers = num_expert_workers
        self.required_responses_per_critical_state = required_responses_per_critical_state
        self.keywords = keywords
        self.external_url = external_url
        
        # Worker qualification settings
        self.require_masters = require_masters
        self.min_approval_rate = min_approval_rate
        self.min_approved_hits = min_approved_hits
        self.require_location = require_location
        
        # Setup output directory and timing file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        env_suffix = "sandbox" if sandbox else "production"
        self.timing_file = self.output_dir / f"hit_timing_{env_suffix}_{timestamp}.json"

        # Initialize MTurk client
        endpoint_url = (
            "https://mturk-requester-sandbox.us-east-1.amazonaws.com"
            if sandbox
            else "https://mturk-requester.us-east-1.amazonaws.com"
        )

        self.mturk_client = boto3.client(
            "mturk",
            endpoint_url=endpoint_url,
            region_name="us-east-1",
        )

        # Verify credentials and get account balance
        try:
            balance = self.mturk_client.get_account_balance()
            env = "sandbox" if sandbox else "production"
            print(f"✅ MTurk {env} initialized. Balance: {balance['AvailableBalance']}")
        except ClientError as e:
            print(f"❌ Failed to initialize MTurk client: {e}")
            raise

        # Track HITs: (episode_id, state_id) -> HIT metadata
        self.hit_tracking: dict[tuple[int, int], dict] = {}
        self.hit_lock = Lock()
        
        # Track HIT completion times: episode_id -> state_id -> {created_at, completed_at, max_assignments, duration_seconds}
        self.hit_timing_stats: dict[int, dict[int, dict]] = {}
        self.timing_lock = Lock()
        
        # Initialize timing file
        self._save_timing_stats()

        # Start background thread for auto-approval
        self.running = True
        self.approval_thread = Thread(target=self._approval_worker, daemon=True)
        self.approval_thread.start()
    
    def _save_timing_stats(self):
        """Save timing statistics to JSON file."""
        with self.timing_lock:
            # Convert to serializable format with computed statistics
            output_data = {
                "metadata": {
                    "sandbox": self.sandbox,
                    "reward": self.reward,
                    "num_expert_workers": self.num_expert_workers,
                    "required_responses_per_critical_state": self.required_responses_per_critical_state,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "hits": {},
                "summary": {}
            }
            
            all_durations = []
            
            for episode_id, states in self.hit_timing_stats.items():
                ep_key = f"episode_{episode_id}"
                output_data["hits"][ep_key] = {}
                
                for state_id, timing in states.items():
                    state_key = f"state_{state_id}"
                    output_data["hits"][ep_key][state_key] = {
                        "max_assignments": timing["max_assignments"],
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timing["created_at"])),
                        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timing["completed_at"])) if timing["completed_at"] else None,
                        "duration_seconds": round(timing["duration_seconds"], 1) if timing["duration_seconds"] else None,
                        "status": "complete" if timing["completed_at"] else "pending"
                    }
                    
                    if timing["duration_seconds"] is not None:
                        all_durations.append(timing["duration_seconds"])
            
            # Compute summary statistics
            if all_durations:
                output_data["summary"] = {
                    "total_hits_completed": len(all_durations),
                    "average_duration_seconds": round(sum(all_durations) / len(all_durations), 1),
                    "average_duration_minutes": round(sum(all_durations) / len(all_durations) / 60, 1),
                    "min_duration_seconds": round(min(all_durations), 1),
                    "max_duration_seconds": round(max(all_durations), 1)
                }
            
            # Write to file
            with open(self.timing_file, 'w') as f:
                json.dump(output_data, f, indent=2)
    
    def _build_qualification_requirements(self) -> list:
        """Build qualification requirements for HIT filtering.
        
        Returns:
            List of QualificationRequirement dictionaries
            
        """
        qualifications = []
        
        # 1. Approval rate requirement (e.g., >= 95%)
        if self.min_approval_rate > 0:
            qualifications.append({
                'QualificationTypeId': '000000000000000000L0',  # Approval rate system qualification
                'Comparator': 'GreaterThanOrEqualTo',
                'IntegerValues': [self.min_approval_rate],
                'RequiredToPreview': True,
            })
        
        # 2. Minimum approved HITs (e.g., >= 100)
        if self.min_approved_hits > 0:
            qualifications.append({
                'QualificationTypeId': '00000000000000000040',  # Number of approved HITs
                'Comparator': 'GreaterThanOrEqualTo',
                'IntegerValues': [self.min_approved_hits],
                'RequiredToPreview': True,
            })
        
        # 3. MTurk Masters qualification (premium workers)
        if self.require_masters:
            # Masters qualification IDs differ between sandbox and production
            masters_id = '2ARFPLSP75KLA8M8DH1HTEQVJT3SY6' if self.sandbox else '2F1QJWKUDD8XADTFD2Q0G6UTO95ALH'
            qualifications.append({
                'QualificationTypeId': masters_id,
                'Comparator': 'Exists',
                'RequiredToPreview': True,
            })
        
        # 4. Location requirement (e.g., US, CA, GB only)
        if self.require_location:
            qualifications.append({
                'QualificationTypeId': '00000000000000000071',  # Location system qualification
                'Comparator': 'In',
                'LocaleValues': [{'Country': code} for code in self.require_location],
                'RequiredToPreview': True,
            })
        
        return qualifications

    def create_hit_for_state(
        self,
        episode_id: int,
        state_id: int,
        state_data: dict,
    ) -> Optional[str]:
        """Create an MTurk HIT for a critical state.

        Args:
            episode_id: Episode ID
            state_id: State ID
            state_data: State information to embed in HIT (images, task, etc.)

        Returns:
            HIT ID if successful, None otherwise

        """
        # Calculate max_assignments: leave slots for expert workers
        max_assignments = max(1, self.required_responses_per_critical_state - self.num_expert_workers)
        
        if self.num_expert_workers > 0:
            print(f"📊 Creating HIT with {max_assignments} MTurk assignments (reserving {self.num_expert_workers} slots for expert workers)")
        
        if not self.external_url:
            print("❌ Cannot create HIT: mturk_external_url not configured")
            return None

        try:
            # Build ExternalQuestion with state data embedded as URL parameters
            # MTurk workers will load: https://your-url.com/sim_mturk.html?episode_id=X&state_id=Y
            # Note: & must be XML-escaped as &amp; in the ExternalQuestion XML
            hit_url = f"{self.external_url.rstrip('/')}/sim_mturk.html?episode_id={episode_id}&amp;state_id={state_id}"

            external_question = f"""
            <ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">
                <ExternalURL>{hit_url}</ExternalURL>
                <FrameHeight>800</FrameHeight>
            </ExternalQuestion>
            """

            # Build qualification requirements
            qualifications = self._build_qualification_requirements()
            
            # Log qualification requirements being applied
            if qualifications:
                print(f"🔒 Applying {len(qualifications)} qualification requirement(s):")
                if self.min_approval_rate > 0:
                    print(f"   • Approval rate >= {self.min_approval_rate}%")
                if self.min_approved_hits > 0:
                    print(f"   • Approved HITs >= {self.min_approved_hits}")
                if self.require_masters:
                    print(f"   • MTurk Masters required")
                if self.require_location:
                    print(f"   • Location: {', '.join(self.require_location)}")

            # Create HIT
            response = self.mturk_client.create_hit(
                Title=self.title,
                Description=self.description,
                Keywords=self.keywords,
                Reward=str(self.reward),
                MaxAssignments=max_assignments,
                LifetimeInSeconds=self.lifetime_seconds,
                AssignmentDurationInSeconds=self.assignment_duration_seconds,
                AutoApprovalDelayInSeconds=self.auto_approval_delay_seconds,
                Question=external_question,
                QualificationRequirements=qualifications,
            )

            hit_id = response["HIT"]["HITId"]
            hit_type_id = response["HIT"]["HITTypeId"]
            
            # Verify HIT was created successfully by fetching it
            # Add small delay to allow MTurk backend to index the HIT
            import time as time_module
            time_module.sleep(0.5)
            
            try:
                hit_info = self.mturk_client.get_hit(HITId=hit_id)
                hit_status = hit_info["HIT"]["HITStatus"]
                hit_review_status = hit_info["HIT"].get("HITReviewStatus", "N/A")
                num_available = hit_info["HIT"]["NumberOfAssignmentsAvailable"]
                expiration = hit_info["HIT"]["Expiration"]
                
                print(f"🔍 HIT Verification:")
                print(f"   Status: {hit_status}")
                print(f"   Review Status: {hit_review_status}")
                print(f"   Assignments Available: {num_available}")
                print(f"   Expires: {expiration}")
                
                if hit_status != "Assignable":
                    print(f"⚠️  WARNING: HIT status is '{hit_status}' - it may not be visible to workers!")
                    
            except ClientError as verify_error:
                print(f"⚠️  Could not verify HIT creation: {verify_error}")
            
            # Track HIT
            created_at = time.time()
            with self.hit_lock:
                self.hit_tracking[(episode_id, state_id)] = {
                    "hit_id": hit_id,
                    "hit_type_id": hit_type_id,
                    "max_assignments": max_assignments,
                    "assignments_submitted": 0,
                    "assignments_approved": 0,
                    "assignments_pending": 0,
                    "pending_since": None,  # Track when assignments became pending
                    "created_at": created_at,
                    "status": "active",
                }
            
            # Track timing for statistics
            with self.timing_lock:
                if episode_id not in self.hit_timing_stats:
                    self.hit_timing_stats[episode_id] = {}
                self.hit_timing_stats[episode_id][state_id] = {
                    "created_at": created_at,
                    "completed_at": None,
                    "max_assignments": max_assignments,
                    "duration_seconds": None,
                }
            
            # Save timing stats to file
            self._save_timing_stats()

            env = "sandbox" if self.sandbox else "production"
            base_url = "workersandbox.mturk.com" if self.sandbox else "worker.mturk.com"
            requester_url = f"https://requester{'sandbox' if self.sandbox else ''}.mturk.com/manage"
            
            # Format creation time
            from datetime import datetime
            creation_time = datetime.fromtimestamp(created_at).strftime("%H:%M:%S")
            
            print(f"✅ Created MTurk HIT ({env}): {hit_id} for episode={episode_id}, state={state_id}")
            print(f"   🕐 Created at: {creation_time}")
            print(f"   🔧 Requester manage: {requester_url}")
            print(f"   👤 Workers can find this HIT by searching on https://{base_url}")
            print(f"      Search for: \"{self.title[:50]}...\" or filter by 'Available HITs'")
            
            # List recent HITs for debugging
            self.list_recent_hits(max_results=5)

            return hit_id

        except ClientError as e:
            print(f"❌ Failed to create MTurk HIT: {e}")
            return None

    def get_hit_status(self, episode_id: int, state_id: int) -> Optional[dict]:
        """Get status of HIT for a specific state.

        Args:
            episode_id: Episode ID
            state_id: State ID

        Returns:
            HIT metadata dict or None if not found

        """
        with self.hit_lock:
            return self.hit_tracking.get((episode_id, state_id))

    def update_hit_assignment_count(self, episode_id: int, state_id: int):
        """Update assignment count when backend receives a submission.

        Called by Flask endpoint after MTurk worker submits response (not expert workers).
        Auto-expires HIT when max assignments reached.

        Args:
            episode_id: Episode ID
            state_id: State ID

        """
        with self.hit_lock:
            key = (episode_id, state_id)
            if key in self.hit_tracking:
                self.hit_tracking[key]["assignments_submitted"] += 1
                print(
                    f"📊 MTurk HIT assignment submitted: episode={episode_id}, state={state_id} "
                    f"({self.hit_tracking[key]['assignments_submitted']}/{self.hit_tracking[key]['max_assignments']})"
                )

                # Mark as complete and expire HIT if all assignments submitted
                if self.hit_tracking[key]["assignments_submitted"] >= self.hit_tracking[key]["max_assignments"]:
                    self.hit_tracking[key]["status"] = "complete"
                    hit_id = self.hit_tracking[key]["hit_id"]
                    
                    # Mark HIT as completed in timing stats
                    self._mark_hit_completed(episode_id, state_id)
                    
                    # Expire the HIT immediately to prevent more workers from accepting
                    try:
                        self.mturk_client.update_expiration_for_hit(
                            HITId=hit_id,
                            ExpireAt=0,
                        )
                        print(f"⏱️  Expired HIT {hit_id} (all assignments received)")
                    except ClientError as e:
                        print(f"⚠️  Failed to expire HIT {hit_id}: {e}")

    def _mark_hit_completed(self, episode_id: int, state_id: int):
        """Mark HIT as completed and compute duration."""
        with self.timing_lock:
            if episode_id in self.hit_timing_stats and state_id in self.hit_timing_stats[episode_id]:
                timing = self.hit_timing_stats[episode_id][state_id]
                if timing["completed_at"] is None:  # Only mark once
                    timing["completed_at"] = time.time()
                    timing["duration_seconds"] = timing["completed_at"] - timing["created_at"]
                    
                    # Log completion
                    print(f"HIT completed: Episode {episode_id}, State {state_id} - "
                          f"{timing['duration_seconds']:.1f}s for {timing['max_assignments']} assignments")
        
        # Save updated timing stats to file
        self._save_timing_stats()
    
    def _approval_worker(self):
        """Background thread that auto-approves all submitted assignments."""
        while self.running:
            try:
                with self.hit_lock:
                    active_hits = [
                        (key, meta)
                        for key, meta in self.hit_tracking.items()
                        if meta["status"] in ["active", "complete"]
                    ]

                # Check each HIT for new assignments
                for (episode_id, state_id), meta in active_hits:
                    try:
                        hit_id = meta["hit_id"]

                        # Check HIT status to monitor pending assignments
                        try:
                            hit_info = self.mturk_client.get_hit(HITId=hit_id)
                            num_pending = hit_info["HIT"]["NumberOfAssignmentsPending"]
                            num_available = hit_info["HIT"]["NumberOfAssignmentsAvailable"]
                            max_assignments = hit_info["HIT"]["MaxAssignments"]
                            
                            # Track and alert when workers accept assignments (slots get taken)
                            with self.hit_lock:
                                prev_pending = self.hit_tracking[(episode_id, state_id)]["assignments_pending"]
                                
                                if num_pending > prev_pending:
                                    # Worker(s) just accepted assignment(s)
                                    from datetime import datetime
                                    accept_time = datetime.now().strftime("%H:%M:%S")
                                    new_workers = num_pending - prev_pending
                                    self.hit_tracking[(episode_id, state_id)]["assignments_pending"] = num_pending
                                    print(f"👤 Worker(s) accepted {new_workers} assignment(s) for episode={episode_id}, state={state_id}")
                                    print(f"   🕐 Accepted at: {accept_time}")
                                    print(f"   📊 Status: {num_pending} pending, {num_available} available, {max_assignments} max")
                                    
                                    # Alert and start timeout timer ONLY when all slots are taken
                                    if num_available == 0:
                                        print(f"🎯 All assignment slots taken for episode={episode_id}, state={state_id} - HIT no longer searchable by workers")
                                        # Start timeout timer only when fully blocked
                                        if self.hit_tracking[(episode_id, state_id)]["pending_since"] is None:
                                            self.hit_tracking[(episode_id, state_id)]["pending_since"] = time.time()
                                            print(f"   ⏱️  Timeout timer started (will expire if blocked for {self.assignment_duration_seconds}s)")
                                
                                elif num_pending < prev_pending:
                                    # Worker abandoned or timed out (submitted assignments are caught separately)
                                    abandoned = prev_pending - num_pending
                                    self.hit_tracking[(episode_id, state_id)]["assignments_pending"] = num_pending
                                    # Clear pending timer if slots become available again
                                    if num_available > 0:
                                        self.hit_tracking[(episode_id, state_id)]["pending_since"] = None
                                    print(f"🚪 Worker(s) abandoned/timed out {abandoned} assignment(s) for episode={episode_id}, state={state_id}")
                                    print(f"   📊 Status: {num_pending} pending, {num_available} available, {max_assignments} max")
                                    
                                    # Alert if slots become available again
                                    if num_available > 0:
                                        print(f"♻️  Assignment slot(s) now available again for episode={episode_id}, state={state_id} - HIT is searchable")
                                
                                # Check for timeout - expire HIT ONLY if all slots blocked for too long
                                pending_since = self.hit_tracking[(episode_id, state_id)]["pending_since"]
                                if pending_since and num_pending > 0 and num_available == 0:
                                    elapsed = time.time() - pending_since
                                    if elapsed > self.assignment_duration_seconds:
                                        print(f"⏰ TIMEOUT: All slots blocked for {elapsed:.0f}s (limit: {self.assignment_duration_seconds}s)")
                                        print(f"   Expiring HIT to kick out non-responsive workers")
                                        try:
                                            # Expire HIT - this forces MTurk to return pending assignments
                                            self.mturk_client.update_expiration_for_hit(HITId=hit_id, ExpireAt=0)
                                            
                                            # Remove from tracking entirely - this allows system to create new HIT
                                            # Even if workers submit after expiration, we want fresh HIT for remaining slots
                                            del self.hit_tracking[(episode_id, state_id)]
                                            
                                            print(f"   ✅ Expired HIT {hit_id} and removed from tracking")
                                            print(f"   System will create new HIT for this state (workers may have submitted late)")
                                            
                                        except ClientError as expire_error:
                                            print(f"   ❌ Failed to expire HIT: {expire_error}")
                                    
                        except ClientError as e:
                            if "does not exist" not in str(e):
                                print(f"⚠️  Failed to check HIT status {hit_id}: {e}")

                        # Get assignments for this HIT
                        response = self.mturk_client.list_assignments_for_hit(
                            HITId=hit_id,
                            AssignmentStatuses=["Submitted"],
                        )

                        # Auto-approve all submitted assignments
                        for assignment in response.get("Assignments", []):
                            assignment_id = assignment["AssignmentId"]
                            try:
                                self.mturk_client.approve_assignment(
                                    AssignmentId=assignment_id,
                                    RequesterFeedback="Thank you for your contribution!",
                                )

                                with self.hit_lock:
                                    self.hit_tracking[(episode_id, state_id)]["assignments_approved"] += 1

                                print(f"✅ Auto-approved MTurk assignment: {assignment_id}")

                            except ClientError as e:
                                # Assignment may already be approved
                                if "already been approved" not in str(e):
                                    print(f"⚠️  Failed to approve assignment {assignment_id}: {e}")

                    except ClientError as e:
                        print(f"⚠️  Failed to check HIT {hit_id}: {e}")

            except Exception as e:
                print(f"❌ Error in MTurk approval worker: {e}")

            # Clean up completed HITs
            self._cleanup_completed_hits()

            # Check every 30 seconds
            time.sleep(30)

    def delete_hit(self, episode_id: int, state_id: int) -> bool:
        """Delete a HIT (mark as reviewable and dispose).

        Args:
            episode_id: Episode ID
            state_id: State ID

        Returns:
            True if successful, False otherwise

        """
        with self.hit_lock:
            key = (episode_id, state_id)
            if key not in self.hit_tracking:
                return False

            hit_id = self.hit_tracking[key]["hit_id"]

        try:
            # Expire the HIT to prevent new assignments
            self.mturk_client.update_expiration_for_hit(
                HITId=hit_id,
                ExpireAt=0,  # Expire immediately
            )

            # Delete the HIT (only works if all assignments are approved/rejected)
            self.mturk_client.delete_hit(HITId=hit_id)

            with self.hit_lock:
                self.hit_tracking[key]["status"] = "deleted"

            print(f"🗑️  Deleted MTurk HIT: {hit_id}")
            return True

        except ClientError as e:
            print(f"⚠️  Failed to delete HIT {hit_id}: {e}")
            return False

    def _cleanup_completed_hits(self):
        """Auto-delete HITs that are complete and all assignments approved.
        
        Called periodically by approval worker thread.
        """
        with self.hit_lock:
            completed_hits = [
                (key, meta)
                for key, meta in self.hit_tracking.items()
                if meta["status"] == "complete"
                and meta["assignments_approved"] >= meta["max_assignments"]
            ]

        for (episode_id, state_id), meta in completed_hits:
            hit_id = meta["hit_id"]
            print(f"🧹 Auto-deleting completed HIT: {hit_id} (episode={episode_id}, state={state_id})")
            self.delete_hit(episode_id, state_id)

    def get_all_hits_status(self) -> dict:
        """Get status of all tracked HITs.

        Returns:
            Dict mapping (episode_id, state_id) to HIT metadata

        """
        with self.hit_lock:
            return {
                f"ep{ep}_state{sid}": meta.copy()
                for (ep, sid), meta in self.hit_tracking.items()
            }

    def cleanup_old_hits(self, max_age_seconds: int = 7200):
        """Clean up HITs older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds (default: 2 hours)

        """
        current_time = time.time()
        with self.hit_lock:
            old_hits = [
                (key, meta)
                for key, meta in self.hit_tracking.items()
                if current_time - meta["created_at"] > max_age_seconds
                and meta["status"] in ["active", "complete"]
            ]

        for (episode_id, state_id), meta in old_hits:
            print(f"🧹 Cleaning up old HIT: episode={episode_id}, state={state_id}")
            self.delete_hit(episode_id, state_id)

    def print_timing_summary(self):
        """Print location of timing statistics file."""
        print(f"\nMTurk HIT timing statistics saved to: {self.timing_file}")
        
        with self.timing_lock:
            if not self.hit_timing_stats:
                print("No HIT timing data available.")
                return
            
            # Count completed HITs
            completed_count = sum(
                1 for states in self.hit_timing_stats.values()
                for timing in states.values()
                if timing["completed_at"] is not None
            )
            total_count = sum(
                len(states) for states in self.hit_timing_stats.values()
            )
            
            print(f"Total HITs: {total_count} ({completed_count} completed, {total_count - completed_count} pending)")
    
    def shutdown(self):
        """Shutdown MTurk manager and stop approval worker."""
        print("Shutting down MTurk manager...")
        
        # Save final timing stats and print summary
        self._save_timing_stats()
        self.print_timing_summary()
        
        self.running = False
        if self.approval_thread.is_alive():
            self.approval_thread.join(timeout=5)
        print("MTurk manager shutdown")
    
    def list_recent_hits(self, max_results: int = 10):
        """List recent HITs for debugging.
        
        Args:
            max_results: Maximum number of HITs to list
            
        """
        try:
            print(f"\n🔍 Listing recent HITs (max {max_results}):")
            response = self.mturk_client.list_hits(MaxResults=max_results)
            
            if not response.get("HITs"):
                print("   No HITs found")
                return
                
            for hit in response["HITs"]:
                hit_id = hit["HITId"]
                status = hit["HITStatus"]
                title = hit["Title"][:50]
                created = hit["CreationTime"]
                assignments = f"{hit['NumberOfAssignmentsPending']}/{hit['NumberOfAssignmentsAvailable']}/{hit['MaxAssignments']}"
                
                print(f"   • {hit_id}")
                print(f"     Status: {status} | Assignments (pending/avail/max): {assignments}")
                print(f"     Title: {title}...")
                print(f"     Created: {created}")
                
        except ClientError as e:
            print(f"   ❌ Failed to list HITs: {e}")


