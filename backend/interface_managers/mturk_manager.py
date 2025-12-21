"""MTurk Manager Module.

Manages Amazon Mechanical Turk HIT creation and lifecycle for crowdsourced robot data collection.
Creates HITs when critical states need labeling, tracks HIT status, and auto-approves submissions.
"""

import json
import time
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

        # Start background thread for auto-approval
        self.running = True
        self.approval_thread = Thread(target=self._approval_worker, daemon=True)
        self.approval_thread.start()

    def create_hit_for_state(
        self,
        episode_id: int,
        state_id: int,
        max_assignments: int,
        state_data: dict,
    ) -> Optional[str]:
        """Create an MTurk HIT for a critical state.

        Args:
            episode_id: Episode ID
            state_id: State ID
            max_assignments: Number of responses needed (required_responses_per_critical_state)
            state_data: State information to embed in HIT (images, task, etc.)

        Returns:
            HIT ID if successful, None otherwise

        """
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
            )

            hit_id = response["HIT"]["HITId"]
            hit_type_id = response["HIT"]["HITTypeId"]

            # Track HIT
            with self.hit_lock:
                self.hit_tracking[(episode_id, state_id)] = {
                    "hit_id": hit_id,
                    "hit_type_id": hit_type_id,
                    "max_assignments": max_assignments,
                    "assignments_submitted": 0,
                    "assignments_approved": 0,
                    "created_at": time.time(),
                    "status": "active",
                }

            env = "sandbox" if self.sandbox else "production"
            base_url = "workersandbox.mturk.com" if self.sandbox else "worker.mturk.com"
            requester_url = f"https://requester{'sandbox' if self.sandbox else ''}.mturk.com/manage"
            worker_preview_url = f"https://{base_url}/mturk/preview?groupId={hit_type_id}"
            
            print(f"✅ Created MTurk HIT ({env}): {hit_id} for episode={episode_id}, state={state_id}")
            print(f"   👤 Worker preview: {worker_preview_url}")
            print(f"   🔧 Requester manage: {requester_url}")

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

        Called by Flask endpoint after worker submits response.
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
                    f"📊 HIT assignment submitted: episode={episode_id}, state={state_id} "
                    f"({self.hit_tracking[key]['assignments_submitted']}/{self.hit_tracking[key]['max_assignments']})"
                )

                # Mark as complete and expire HIT if all assignments submitted
                if self.hit_tracking[key]["assignments_submitted"] >= self.hit_tracking[key]["max_assignments"]:
                    self.hit_tracking[key]["status"] = "complete"
                    hit_id = self.hit_tracking[key]["hit_id"]
                    
                    # Expire the HIT immediately to prevent more workers from accepting
                    try:
                        self.mturk_client.update_expiration_for_hit(
                            HITId=hit_id,
                            ExpireAt=0,
                        )
                        print(f"⏱️  Expired HIT {hit_id} (all assignments received)")
                    except ClientError as e:
                        print(f"⚠️  Failed to expire HIT {hit_id}: {e}")

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

    def shutdown(self):
        """Shutdown MTurk manager and stop approval worker."""
        self.running = False
        if self.approval_thread.is_alive():
            self.approval_thread.join(timeout=5)
        print("🛑 MTurk manager shutdown")
