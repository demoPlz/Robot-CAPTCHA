from __future__ import annotations


CAM_IDS = {
    "front":       14,   # change indices / paths as needed
    "left":        16,
    "right":       18,
    "perspective": 12,
}

JOINT_NAMES = [
    "joint_0", "joint_1", "joint_2",
    "joint_3", "joint_4", "joint_5",
    "left_carriage_joint"
]

import cv2
import time
import trossen_arm
import numpy as np

from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
from threading import Thread, Lock
from collections import deque
from math import cos, sin

import argparse

_REALSENSE_BLOCKLIST = (
    "realsense", "real sense", "d4", "depth", "infrared", "stereo module", "motion module"
)

### HELPERS

def _v4l2_node_name(idx: int) -> str:
    """Fast sysfs read of V4L2 device name; '' if unknown."""
    try:
        with open(f"/sys/class/video4linux/video{idx}/name", "r", encoding="utf-8") as f:
            return f.read().strip().lower()
    except Exception:
        return ""

def _is_webcam_idx(idx: int) -> bool:
    """True if /dev/video{idx} looks like a regular webcam, not a RealSense node."""
    name = _v4l2_node_name(idx)
    if not name:
        # If we can't read the name, allow it and let open/read decide.
        return True
    return not any(term in name for term in _REALSENSE_BLOCKLIST)

def _prep_capture(cap: cv2.VideoCapture, width=640, height=480, fps=None, mjpg=True):
    """Apply low-latency, webcam-friendly settings once at open."""
    # Keep the buffer tiny to minimize latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Many webcams unlock higher modes with MJPG
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(width))
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps:    cap.set(cv2.CAP_PROP_FPS,          float(fps))

class CrowdInterface():
    '''
    Sits between the frontend and the backend
    '''
    def __init__(self):

        self.states = deque()
        self.cams = {}
        self.latest_goal = None
        self.goal_lock = Lock()
        self._gripper_motion = 1  # Initialize gripper motion

        self._camera_poses = self._make_camera_poses()


        # Precompute immutable views and camera poses to avoid per-tick allocations
        H, W = 64, 64
        pink = np.full((H, W, 3), 255, dtype=np.uint8)  # one-time NumPy buffer
        blank = pink.tolist()                           # one-time JSON-serializable view
        # Reuse the same object for all cameras (read-only downstream)
        self._blank_views = {
            "left":        blank,
            "right":       blank,
            "front":       blank,
            "perspective": blank,
        }


    ### ---Camera Management---
    def init_cameras(self):
        """Open only *webcams* (skip RealSense nodes) once; skip any that fail."""
        self.cams = getattr(self, "cams", {})
        for name, idx in CAM_IDS.items():
            # Only attempt indices that look like webcams
            if not _is_webcam_idx(idx):
                print(f"⏭️  skipping '{name}' (/dev/video{idx}) — not a webcam")
                continue

            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                print(f"⚠️  camera “{name}” (id {idx}) could not be opened")
                continue

            # One-time efficiency settings
            _prep_capture(cap, width=640, height=480, fps=None, mjpg=True)

            # Verify we can actually read one frame
            ok, _ = cap.read()
            if not ok:
                cap.release()
                print(f"⚠️  camera “{name}” (id {idx}) opens but won't deliver frames")
                continue

            self.cams[name] = cap
            print(f"✓ Camera '{name}' opened successfully (/dev/video{idx})")


    def cleanup_cameras(self):
        """Close all cameras"""
        for cap in getattr(self, "cams", {}).values():
            try:
                cap.release()
            except Exception:
                pass
        self.cams = {}


    def get_views(self) -> dict[str, list]:
        """Return an RGB image dict from available webcams.
        Uses grab() → retrieve() pattern for near-simultaneous multi-cam capture."""
        if not hasattr(self, "cams"):
            self.cams = {}

        order = ("left", "right", "front", "perspective")
        # 1) grab from all first (non-blocking dequeue)
        for name in order:
            if name in self.cams:
                self.cams[name].grab()

        # 2) retrieve, convert, downscale
        views: dict[str, list] = {}
        for name in order:
            if name not in self.cams:
                continue
            frame = self._grab_frame(self.cams[name], size=(640, 480))
            if frame is not None:
                views[name] = frame.tolist()
        return views


    def _grab_frame(self, cap, size=(64, 64)) -> np.ndarray | None:
        # retrieve() after grab(); falls back to read() if needed
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            ok, frame = cap.read()
            if not ok or frame is None:
                return None
        # Resize then convert to RGB; INTER_AREA is efficient for downscale
        if size is not None:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    # --- State Management ---
    def add_state(self, joint_positions: dict, gripper_motion: int = None):
        if gripper_motion is not None:
            self._gripper_motion = int(gripper_motion)

        # Cheap, explicit cast of the 7 scalars to built-in floats
        jp = {k: float(v) for k, v in joint_positions.items()}

        self.states.append({
            "joint_positions": jp,
            "views": self.get_views(),       # reuse precomputed JSON-serializable views
            "camera_poses": self._camera_poses,  # reuse precomputed poses
            "gripper": self._gripper_motion,
            "controls": ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'],
        })
        # print(f"🟢 State added at {current_time}, total states: {len(self.states)}")
        # print(f"🟢 Joint positions: {joint_positions}")
        # print(f"🟢 Gripper: {self._gripper_motion}")
    
    def get_latest_state(self) -> dict:
        """Get the latest state (pops from queue)"""
        # print(f"🔍 get_latest_state called - states length: {len(self.states)}")
        if not self.states:
            # print("🔍 No states available, returning empty dict")
            return {}
        latest = self.states[-1]
        # print(f"🔍 Returning latest state with keys: {list(latest.keys())}")
        return latest
    
    # --- Goal Management ---
    def submit_goal(self, goal_data: dict):
        """Submit a new goal from the frontend"""
        self.latest_goal = goal_data
        # print(f"🔔 Goal received: {goal_data}")
    
    def get_latest_goal(self) -> dict | None:
        """Get and clear the latest goal (for robot loop to consume)"""
        goal = self.latest_goal
        self.latest_goal = None
        return goal
    
    def has_pending_goal(self) -> bool:
        """Check if there's a pending goal"""
        return self.latest_goal is not None
    
    # --- Helper Methods ---

    def _make_camera_poses(self) -> dict[str, list]: #TODO Placeholders for now
        def euler_pose(x: float, y: float, z: float,
               roll: float, pitch: float, yaw: float) -> list[list[float]]:
            """
            Build a 4x4 **world** matrix (row-major list-of-lists) from
            T = Trans(x,y,z) · Rz(yaw) · Ry(pitch) · Rx(roll)
            """
            cr, sr = cos(roll),  sin(roll)
            cp, sp = cos(pitch), sin(pitch)
            cy, sy = cos(yaw),   sin(yaw)

            # column-major rotation
            Rrow = np.array([
                [ cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [ sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [   -sp,              cp*sr,              cp*cr]
            ])

            T = np.eye(4)
            T[:3, :3] = Rrow.T
            T[:3,  3] = [x, y, z]
            return T.tolist()
        
        return {
            #           x     y     z     roll   pitch   yaw
            "front_pose":       euler_pose(1.0, 0.0, 0.15, 0.0, -np.pi/2 - 0.1, -np.pi/2),
            "left_pose":        euler_pose(0.2, -1.0, 0.15, -np.pi/2, 0.0, 0.0),
            "right_pose":       euler_pose(0.2,  1.0, 0.15, np.pi/2, 0.0, np.pi),
            "perspective_pose": euler_pose(1.3,  1.0, 1.0, np.pi/4, -np.pi/4, -3*np.pi/4),
        }
    

    
def create_flask_app(crowd_interface: CrowdInterface) -> Flask:
    """Create and configure Flask app with the crowd interface"""
    app = Flask(__name__)
    CORS(app)
    
    @app.route("/api/get-state")
    def get_state():
        import time
        current_time = time.time()
        state = crowd_interface.get_latest_state()
        # print(f"🔍 Flask route /api/get-state called at {current_time}")
        # print(f"🔍 crowd_interface.states length: {len(crowd_interface.states)}")
        # if len(crowd_interface.states) > 0:
        #     print(f"🔍 Latest state joint_positions: {crowd_interface.states[-1].get('joint_positions', 'NO_JOINTS')}")
        #     print(f"🔍 Latest state gripper_action: {crowd_interface.states[-1]['gripper']}")
        response = jsonify(state)
        # Prevent caching
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    @app.route("/api/test")
    def test():
        return jsonify({"message": "Flask server is working", "states_count": len(crowd_interface.states)})
    
    @app.route("/api/submit-goal", methods=["POST"])
    def submit_goal():
        data = request.get_json(force=True, silent=True) or {}
        crowd_interface.submit_goal(data)
        return jsonify({"status": "ok"})
    
    return app