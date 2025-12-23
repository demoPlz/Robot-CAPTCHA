"""Configuration for CrowdInterface crowd-sourced data collection."""

import argparse
import sys
from pathlib import Path


class CrowdInterfaceConfig:
    """Configuration for crowd-sourced robot data collection interface.

    This config manages settings for:
    - Task definition and labeling requirements
    - Critical state annotation (user responses per state)
    - Demo video recording and display
    - Simulation integration
    - Object tracking and pose estimation

    """

    def __init__(self):
        # ========== Task Settings ==========
        self.task_name: str = "drawer"  # Single-word identifier for the task
        self.task_text: str = "Open the drawer, put the cube on the desk into the middle drawer, and close the drawer"

        # ========== Labeling Requirements ==========
        self.required_responses_per_state: int = 1  # Non-critical states
        self.required_responses_per_critical_state: int = 2  # Critical states requiring multiple labels

        self.required_approvals_per_critical_state: int = 2
        
        # ========== Expert Worker Integration ==========
        # Number of expert workers who will label via localhost
        # MTurk max_assignments will be: required_responses_per_critical_state - num_expert_workers
        self.num_expert_workers: int = 1  # Set to 0 for no experts, or >= 1 to reserve slots

        # ========== Jitter Detection ==========
        # Automatic jitter detection: if new critical state is too similar to unlabeled previous critical,
        # automatically discard it. Threshold is L2 distance in joint positions (radians).
        self.jitter_threshold: float = 0.05  # radians - tune based on robot sensitivity

        # ========== Critical State Autofill ==========
        # When enabled, critical states receive num_autofill_actions + 1 responses (cloned) per response
        self.autofill_critical_states: bool = False
        self.num_autofill_actions: int | None = None

        # ========== UI Prompting ==========
        self.use_manual_prompt: bool = False  # Manual text/video prompt selection per state
        self.show_demo_videos: bool = False  # Display reference videos to users

        # ========== Demo Video Recording ==========
        # Records user interaction videos for training/demonstration purposes
        self.record_ui_demo_videos: bool = False
        self.ui_demo_videos_dir: str | None = None  # Defaults to data/prompts/{task_name}/videos
        self.clear_ui_demo_videos_dir: bool = False  # Clear directory on startup

        # ========== Simulation ==========
        self.use_sim: bool = True  # Use Isaac Sim for state simulaion
        self.use_gpu_physics: bool = False  # Use GPU physics (faster but uses more VRAM) vs CPU physics
        self.max_animation_users: int = 1  # Maximum simultaneous users viewing animations
        
        # USD file path for Isaac Sim (relative to repo root)
        self.usd_path: str = f"public/assets/usd/{self.task_name}.usd"

        # ========== Object Tracking ==========
        # Object names and their language descriptions for pose estimation
        # Note: Keys must match USD prim names in Isaac Sim
        # self.objects: dict[str, str] = {"Cube_Blue": "Blue cube", "Cube_Red": "Red cube", "Tennis": "Tennis ball"}

        self.objects: dict[str, str] = {"Cube_Blue": "Blue cube"}

        # Resolve mesh paths for pose estimation (relative to repo root)
        repo_root = Path(__file__).resolve().parent.parent
        self.object_mesh_paths: dict[str, str] = {
            "Cube_Blue": str((repo_root / "public" / "assets" / "cube.obj").resolve()),
            "Cube_Red": str((repo_root / "public" / "assets" / "cube.obj").resolve()),
            "Tennis": str((repo_root / "public" / "assets" / "sphere.obj").resolve()),
        }

        # ========== Joint Tracking ==========
        # Track prismatic joint positions of drawer for drawer task
        self.joint_tracking: list = ["Drawer_Joint"]

        # ========== Action Selection ==========
        # Strategy for selecting which crowdsourced action to execute
        self.action_selector_mode: str = "random"  # "random", "epsilon_greedy", or "learned"
        self.action_selector_epsilon: float = 0.1  # Exploration rate for epsilon_greedy mode
        self.action_selector_model_path: str | None = None  # Path to learned selector model

        # ========== MTurk Integration ==========
        self.use_mturk: bool = False  # Enable MTurk HIT creation for critical states
        self.mturk_sandbox: bool = False  # Use MTurk sandbox (False for production)
        self.mturk_reward: float = 0.25  # Payment per assignment in USD
        self.mturk_assignment_duration_seconds: int = 180  # Time allowed per assignment (3 minutes)
        self.mturk_lifetime_seconds: int = 3600  # How long HIT remains available (1 hour)
        self.mturk_auto_approval_delay_seconds: int = 60  # Auto-approve after 1 minute
        self.mturk_title: str = "Control a robot arm"
        self.mturk_description: str = "View a task environment and specify the next position for the robot to move to"
        self.mturk_keywords: str = "robot, manipulation, annotation"
        # External URL where your frontend is hosted (for MTurk ExternalQuestion)
        # If not set, will auto-detect from /tmp/cloudflared.log (created by start_tunnel.sh)
        # Manual override: set to your public URL (ngrok/cloudflare tunnel)
        self.mturk_external_url: str | None = None  # Auto-detected if using start_tunnel.sh

    @classmethod
    def from_cli_args(cls, argv=None):
        """Create a CrowdInterfaceConfig instance with CLI overrides.

        Parses crowd-specific CLI arguments and removes them from sys.argv,
        allowing LeRobot's argument parser to process remaining args without conflicts.

        Args:
            argv: Command-line arguments to parse (defaults to sys.argv[1:])

        Returns:
            CrowdInterfaceConfig: Configuration instance with CLI overrides applied

        """
        parser = argparse.ArgumentParser(add_help=False)

        # Task settings
        parser.add_argument(
            "--task-name",
            type=str,
            help="Single-word identifier for the task (default: drawer)",
        )
        
        # Labeling settings
        parser.add_argument(
            "--required-responses-per-critical-state",
            type=int,
            help="Number of user responses required for critical states",
        )
        parser.add_argument(
            "--required-approvals-per-critical-state",
            type=int,
            help="Number of pre-approved actions required per critical state",
        )
        parser.add_argument(
            "--num-expert-workers",
            type=int,
            help="Number of expert workers labeling via localhost (reserves slots from MTurk)",
        )
        parser.add_argument(
            "--jitter-threshold",
            type=float,
            help="L2 distance threshold for automatic jitter detection in radians (default: 0.01)",
        )
        parser.add_argument(
            "--autofill-critical-states",
            action="store_true",
            help="Auto-complete critical states after partial responses",
        )
        parser.add_argument(
            "--num-autofill-actions", type=int, help="Number of responses before auto-fill (default: all required)"
        )

        # UI settings
        parser.add_argument(
            "--use-manual-prompt", action="store_true", help="Require manual text/video prompt selection per state"
        )
        parser.add_argument("--show-demo-videos", action="store_true", help="Display reference videos to labelers")

        # Demo video recording
        parser.add_argument("--record-ui-demo-videos", action="store_true", help="Record user interaction videos")
        parser.add_argument(
            "--ui-demo-videos-dir", type=str, help="Directory for demo videos (default: data/prompts/{task}/videos)"
        )
        parser.add_argument(
            "--clear-ui-demo-videos-dir", action="store_true", help="Clear demo videos directory on startup"
        )

        # Simulation
        parser.add_argument("--use-sim", action="store_true", help="Enable Isaac Sim integration")
        parser.add_argument("--use-gpu-physics", action="store_true", help="Use GPU physics in Isaac Sim (faster but uses more VRAM)")
        parser.add_argument(
            "--max-animation-users",
            type=int,
            help="Maximum number of simultaneous users viewing animations (default: 2)",
        )
        parser.add_argument(
            "--usd-path",
            type=str,
            help="Path to USD file for Isaac Sim (relative to repo root)",
        )

        # Action selection
        parser.add_argument(
            "--action-selector-mode",
            type=str,
            choices=["random", "epsilon_greedy", "learned"],
            help="Action selection strategy: random, epsilon_greedy, or learned (default: random)",
        )
        parser.add_argument(
            "--action-selector-epsilon",
            type=float,
            help="Exploration rate for epsilon-greedy mode (default: 0.1)",
        )
        parser.add_argument(
            "--action-selector-model-path",
            type=str,
            help="Path to learned selector model weights",
        )

        # MTurk integration
        parser.add_argument(
            "--use-mturk",
            action="store_true",
            help="Enable MTurk HIT creation for critical states",
        )
        parser.add_argument(
            "--mturk-sandbox",
            action="store_true",
            help="Use MTurk sandbox environment (default: True)",
        )
        parser.add_argument(
            "--mturk-production",
            action="store_true",
            help="Use MTurk production environment (overrides --mturk-sandbox)",
        )
        parser.add_argument(
            "--mturk-reward",
            type=float,
            help="Payment per MTurk assignment in USD (default: 0.50)",
        )
        parser.add_argument(
            "--mturk-assignment-duration",
            type=int,
            help="Time allowed per MTurk assignment in seconds (default: 600)",
        )
        parser.add_argument(
            "--mturk-lifetime",
            type=int,
            help="How long HIT remains available in seconds (default: 3600)",
        )
        parser.add_argument(
            "--mturk-auto-approval-delay",
            type=int,
            help="Auto-approve delay in seconds (default: 60, minimum: 60)",
        )
        parser.add_argument(
            "--mturk-title",
            type=str,
            help="MTurk HIT title",
        )
        parser.add_argument(
            "--mturk-description",
            type=str,
            help="MTurk HIT description",
        )
        parser.add_argument(
            "--mturk-keywords",
            type=str,
            help="MTurk HIT keywords (comma-separated)",
        )
        parser.add_argument(
            "--mturk-external-url",
            type=str,
            help="Public URL for MTurk workers (e.g., https://abc123.trycloudflare.com)",
        )

        args, remaining = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

        # Remove crowd-specific args from sys.argv for downstream parsers
        sys.argv = [sys.argv[0]] + remaining

        # Create config and apply overrides
        config = cls()

        if args.task_name is not None:
            config.task_name = args.task_name
        if args.required_responses_per_critical_state is not None:
            config.required_responses_per_critical_state = args.required_responses_per_critical_state
        if args.required_approvals_per_critical_state is not None:
            config.required_approvals_per_critical_state = args.required_approvals_per_critical_state
        if args.num_expert_workers is not None:
            config.num_expert_workers = args.num_expert_workers
        if args.jitter_threshold is not None:
            config.jitter_threshold = args.jitter_threshold
        if args.autofill_critical_states:
            config.autofill_critical_states = True
        if args.num_autofill_actions is not None:
            config.num_autofill_actions = args.num_autofill_actions
        if args.use_manual_prompt:
            config.use_manual_prompt = True
        if args.show_demo_videos:
            config.show_demo_videos = True
        if args.record_ui_demo_videos:
            config.record_ui_demo_videos = True
        if args.ui_demo_videos_dir is not None:
            config.ui_demo_videos_dir = args.ui_demo_videos_dir
        if args.clear_ui_demo_videos_dir:
            config.clear_ui_demo_videos_dir = True
        if args.use_sim:
            config.use_sim = True
        if args.use_gpu_physics:
            config.use_gpu_physics = True
        if args.max_animation_users is not None:
            config.max_animation_users = args.max_animation_users
        if args.usd_path is not None:
            config.usd_path = args.usd_path
        if args.action_selector_mode is not None:
            config.action_selector_mode = args.action_selector_mode
        if args.action_selector_epsilon is not None:
            config.action_selector_epsilon = args.action_selector_epsilon
        if args.action_selector_model_path is not None:
            config.action_selector_model_path = args.action_selector_model_path
        if args.use_mturk:
            config.use_mturk = True
        if args.mturk_production:
            config.mturk_sandbox = False
        elif args.mturk_sandbox:
            config.mturk_sandbox = True
        if args.mturk_reward is not None:
            config.mturk_reward = args.mturk_reward
        if args.mturk_assignment_duration is not None:
            config.mturk_assignment_duration_seconds = args.mturk_assignment_duration
        if args.mturk_lifetime is not None:
            config.mturk_lifetime_seconds = args.mturk_lifetime
        if args.mturk_auto_approval_delay is not None:
            config.mturk_auto_approval_delay_seconds = args.mturk_auto_approval_delay
        if args.mturk_title is not None:
            config.mturk_title = args.mturk_title
        if args.mturk_description is not None:
            config.mturk_description = args.mturk_description
        if args.mturk_keywords is not None:
            config.mturk_keywords = args.mturk_keywords
        if args.mturk_external_url is not None:
            config.mturk_external_url = args.mturk_external_url

        return config

    def to_crowd_interface_kwargs(self) -> dict:
        """Convert configuration to CrowdInterface constructor kwargs.

        Returns:
            dict: Keyword arguments for CrowdInterface.__init__()

        """
        kwargs = {
            # Core settings
            "task_name": self.task_name,
            "task_text": self.task_text,
            "required_responses_per_state": self.required_responses_per_state,
            "required_responses_per_critical_state": self.required_responses_per_critical_state,
            "required_approvals_per_critical_state": self.required_approvals_per_critical_state,
            "num_expert_workers": self.num_expert_workers,
            "jitter_threshold": self.jitter_threshold,
            # Autofill
            "autofill_critical_states": self.autofill_critical_states,
            "num_autofill_actions": self.num_autofill_actions,
            # UI
            "use_manual_prompt": self.use_manual_prompt,
            "show_demo_videos": self.show_demo_videos,
            # Simulation
            "use_sim": self.use_sim,
            "max_animation_users": self.max_animation_users,
            "usd_path": self.usd_path,
            # Object tracking
            "objects": self.objects,
            "object_mesh_paths": self.object_mesh_paths,
            # Action selection
            "action_selector_mode": self.action_selector_mode,
            "action_selector_epsilon": self.action_selector_epsilon,
            "action_selector_model_path": self.action_selector_model_path,
            # MTurk integration
            "use_mturk": self.use_mturk,
            "mturk_sandbox": self.mturk_sandbox,
            "mturk_reward": self.mturk_reward,
            "mturk_assignment_duration_seconds": self.mturk_assignment_duration_seconds,
            "mturk_lifetime_seconds": self.mturk_lifetime_seconds,
            "mturk_auto_approval_delay_seconds": self.mturk_auto_approval_delay_seconds,
            "mturk_title": self.mturk_title,
            "mturk_description": self.mturk_description,
            "mturk_keywords": self.mturk_keywords,
            "mturk_external_url": self.mturk_external_url,
        }

        # Optional: demo video recording (only include if enabled)
        if self.record_ui_demo_videos:
            kwargs["record_demo_videos"] = True
            if self.ui_demo_videos_dir is not None:
                kwargs["demo_videos_dir"] = self.ui_demo_videos_dir
            if self.clear_ui_demo_videos_dir:
                kwargs["demo_videos_clear"] = True

        return kwargs
