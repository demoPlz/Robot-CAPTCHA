# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from isaacsim import SimulationApp

# Start the simulation
simulation_app = SimulationApp({"headless": False})

import requests
import numpy as np
import carb
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.articulations import Articulation
from pxr import Gf

# --- Configuration ---
USD_PATH = "/home/yilong/drawer_flattened.usd"
BACKEND_URL = "http://127.0.0.1:9000/initial-state"

ROBOT_PATH = "/World/wxai"
OBJ_CUBE_01_PATH = "/World/Cube_01"
OBJ_CUBE_02_PATH = "/World/Cube_02"
OBJ_TENNIS_PATH = "/World/Tennis"

# --- Helper Functions (omitted for brevity, no changes needed) ---
def get_initial_state_from_backend():
    """Fetches the initial state from your backend service."""
    try:
        print(f"Fetching initial state from {BACKEND_URL}")
        response = requests.get(BACKEND_URL, timeout=5)
        response.raise_for_status()
        state = response.json()
        print("Successfully fetched initial state from backend")
        return state
    except requests.exceptions.RequestException as e:
        carb.log_error(f"Failed to fetch initial state from backend: {str(e)}. Using default state.")
        return {
            "q": [0.0] * 7,  # Default 7-DOF joint positions
            "objects": {
                "Cube_01": {"pos": [0.5, 0.0, 0.1], "rot": [0, 0, 0, 1]},
                "Cube_02": {"pos": [0.5, 0.2, 0.1], "rot": [0, 0, 0, 1]},
                "Tennis": {"pos": [0.5, -0.2, 0.1], "rot": [0, 0, 0, 1]}
            }
        }

# --- Main Simulation Logic ---

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>
# CRITICAL CHANGE: Load the USD stage FIRST
print(f"Loading environment from {USD_PATH}")
omni.usd.get_context().open_stage(USD_PATH)

# Wait for the stage to load by stepping the simulation a few times
print("Waiting for stage to load...")
for i in range(20):
    simulation_app.update()

# CRITICAL CHANGE: Create the World object AFTER the stage is loaded
my_world = World(stage_units_in_meters=1.0)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<

# Now that the world and stage are loaded, add a ground plane
my_world.scene.add_default_ground_plane()
print("Stage loaded and World object created.")

# Fetch the initial state from your backend
initial_state = get_initial_state_from_backend()
initial_q = np.array(initial_state["q"])
initial_q = np.append(initial_q, initial_q[-1])
object_states = initial_state["objects"]

# Get handles to the prims on the stage by adding them to the scene
# This registers them with the physics engine
robot = my_world.scene.add(
    Articulation(prim_path=ROBOT_PATH, name="widowx_robot")
)
cube_01 = my_world.scene.add(
    RigidPrim(prim_path=OBJ_CUBE_01_PATH, name="cube_01")
)
cube_02 = my_world.scene.add(
    RigidPrim(prim_path=OBJ_CUBE_02_PATH, name="cube_02")
)
tennis_ball = my_world.scene.add(
    RigidPrim(prim_path=OBJ_TENNIS_PATH, name="tennis_ball")
)

my_world.reset()


robot.set_joint_positions(initial_q)

# Set object poses using the high-level API
if cube_01.is_valid():
    state = object_states.get("Cube_01")
    if state:
        pos = np.array(state["pos"])
        rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]]) # WXYZ
        cube_01.set_world_pose(position=pos, orientation=rot)
        print(f"Set pose for {cube_01.name}: pos={pos}, rot={state['rot']}")

if cube_02.is_valid():
    state = object_states.get("Cube_02")
    if state:
        pos = np.array(state["pos"])
        rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]]) # WXYZ
        cube_02.set_world_pose(position=pos, orientation=rot)
        print(f"Set pose for {cube_02.name}: pos={pos}, rot={state['rot']}")
        
if tennis_ball.is_valid():
    state = object_states.get("Tennis")
    if state:
        pos = np.array(state["pos"])
        rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]]) # WXYZ
        tennis_ball.set_world_pose(position=pos, orientation=rot)
        print(f"Set pose for {tennis_ball.name}: pos={pos}, rot={state['rot']}")

print("\nScene setup complete. Starting simulation loop...")
# --- Simulation Loop ---
while simulation_app.is_running():
    my_world.step(render=True)

simulation_app.close()