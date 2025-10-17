#!/usr/bin/env python3
"""
Isaac Sim worker script that does exactly what test.py does.
This script runs in Isaac Sim's Python environment.
"""

import sys
import json
import argparse
from PIL import Image
from isaacsim import SimulationApp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='JSON config file path')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for images')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start the simulation - exactly like test.py
    simulation_app = SimulationApp({"headless": True})
    
    import numpy as np
    import carb
    import omni.usd
    from omni.isaac.core import World
    from omni.isaac.core.prims import RigidPrim
    from omni.isaac.core.articulations import Articulation
    from pxr import Gf, UsdGeom, Usd
    from isaacsim.sensors.camera import Camera, get_all_camera_objects
    
    # Configuration - exactly like test.py
    USD_PATH = config['usd_path']
    ROBOT_PATH = "/World/wxai"
    OBJ_CUBE_01_PATH = "/World/Cube_01"
    OBJ_CUBE_02_PATH = "/World/Cube_02"
    OBJ_TENNIS_PATH = "/World/Tennis"
    
    def hide_robot():
        """Alternative: Move robot far away (preserves everything)"""
        hide_robot.original_pos, hide_robot.original_rot = robot.get_world_pose()
        hide_robot.current_joint_positions = robot.get_joint_positions()
        # Move robot below ground
        robot.set_world_pose(position=np.array([0, 0, -100]), orientation=hide_robot.original_rot)

    def show_robot():
        """Restore robot to original position"""
        robot.set_world_pose(position=hide_robot.original_pos, 
                            orientation=hide_robot.original_rot)
        robot.set_joint_positions(hide_robot.current_joint_positions)
    
    try:
        # Load the USD stage FIRST - exactly like test.py
        print(f"Loading environment from {USD_PATH}")
        omni.usd.get_context().open_stage(USD_PATH)

        # Wait for the stage to load - exactly like test.py
        for i in range(20):
            simulation_app.update()

        # Create the World object AFTER the stage is loaded - exactly like test.py
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        print("Stage loaded and World object created.")

        # Get initial state from config instead of backend
        initial_q = np.array(config.get('robot_joints', [0.0] * 7))
        initial_q = np.append(initial_q, initial_q[-1])  # Exactly like test.py
        object_states = config.get('object_poses', {
            "Cube_01": {"pos": [0.5, 0.0, 0.1], "rot": [0, 0, 0, 1]},
            "Cube_02": {"pos": [0.5, 0.2, 0.1], "rot": [0, 0, 0, 1]},
            "Tennis": {"pos": [0.5, -0.2, 0.1], "rot": [0, 0, 0, 1]}
        })

        # Get handles to the prims - exactly like test.py
        robot = world.scene.add(Articulation(prim_path=ROBOT_PATH, name="widowx_robot"))
        cube_01 = world.scene.add(RigidPrim(prim_path=OBJ_CUBE_01_PATH, name="cube_01"))
        cube_02 = world.scene.add(RigidPrim(prim_path=OBJ_CUBE_02_PATH, name="cube_02"))
        tennis_ball = world.scene.add(RigidPrim(prim_path=OBJ_TENNIS_PATH, name="tennis_ball"))

        cameras = {}
        stage = omni.usd.get_context().get_stage()
        all_cameras = get_all_camera_objects(root_prim='/')
        cameras = {all_cameras[i].name: all_cameras[i] for i in range(len(all_cameras))}

        # Reset world and set initial poses - exactly like test.py
        world.reset()

        # Initialize cameras AFTER world.reset() - exactly like test.py
        for camera in all_cameras:
            camera.initialize()
            camera.set_resolution((640,480))
            
            camera.add_rgb_to_frame()

        robot.set_joint_positions(initial_q)

        # Set object poses - exactly like test.py
        if cube_01.is_valid():
            state = object_states.get("Cube_01")
            if state:
                pos = np.array(state["pos"])
                rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]]) # WXYZ
                cube_01.set_world_pose(position=pos, orientation=rot)

        if cube_02.is_valid():
            state = object_states.get("Cube_02")
            if state:
                pos = np.array(state["pos"])
                rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]]) # WXYZ
                cube_02.set_world_pose(position=pos, orientation=rot)
                
        if tennis_ball.is_valid():
            state = object_states.get("Tennis")
            if state:
                pos = np.array(state["pos"])
                rot = np.array([state["rot"][3], state["rot"][0], state["rot"][1], state["rot"][2]]) # WXYZ
                tennis_ball.set_world_pose(position=pos, orientation=rot)

        print("\nScene setup complete. Starting simulation loop...")
        # --- Simulation Loop - exactly like test.py ---

        hide_robot()
        for step in range(10):
            world.step(render=True)

        # Capture initial images (robot hidden) - exactly like test.py
        initial_front_rgb = cameras['Camera_Front'].get_rgb()
        initial_left_rgb = cameras['Camera_Left'].get_rgb()
        initial_right_rgb = cameras['Camera_Right'].get_rgb()
        initial_top_rgb = cameras['Camera_Top'].get_rgb()

        # Save initial images to disk as JPEG
        Image.fromarray(initial_front_rgb).save(f'{args.output_dir}/initial_front_image.jpg', 'JPEG', quality=90)
        Image.fromarray(initial_left_rgb).save(f'{args.output_dir}/initial_left_image.jpg', 'JPEG', quality=90)
        Image.fromarray(initial_right_rgb).save(f'{args.output_dir}/initial_right_image.jpg', 'JPEG', quality=90)
        Image.fromarray(initial_top_rgb).save(f'{args.output_dir}/initial_top_image.jpg', 'JPEG', quality=90)

        # Prepare results as file paths - exactly like test.py would return
        results = {
            "front_rgb": f'{args.output_dir}/initial_front_image.jpg',
            "left_rgb": f'{args.output_dir}/initial_left_image.jpg',
            "right_rgb": f'{args.output_dir}/initial_right_image.jpg',
            "top_rgb": f'{args.output_dir}/initial_top_image.jpg',
            "status": "success"
        }
        
        # Output results as JSON to stdout for backend to capture
        print(json.dumps(results))
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()