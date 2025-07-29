# In server.py

from collections import deque

from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np

states = deque() # queue for storing states for data collection 
# {images, joint_positions, gripper_position} -> goal_poses

def get_current_robot_joint_positions():
    """
    SIMULATED: Reads the current joint angles from the physical robot hardware.
    The keys in this dictionary MUST match the <joint name="..."> in the URDF file.
    """
    print("Reading 'live' joint positions from the (simulated) robot...")
    
    simulated_positions_numpy = {
        "joint_0": 0.0 * np.pi / 180.0,
        "joint_1": 61.0 * np.pi / 180.0,
        "joint_2": 73.0 * np.pi / 180.0,
        "joint_3": 61.0 * np.pi / 180.0,
        "joint_4": 0.0 * np.pi / 180.0,
        "joint_5": 0.0 * np.pi / 180.0
    }

    # --- THIS IS THE FIX ---
    # Create a new dictionary, converting every NumPy value to a standard Python float.
    simulated_positions_python = {
        key: float(value) for key, value in simulated_positions_numpy.items()
    }
    # -----------------------

    return simulated_positions_python

app = Flask(__name__)
CORS(app)

# @app.route('/')
# def index():
#     return "Flask Server is running!"

@app.route('/api/get-current-pose')
def get_current_pose():
    """The API endpoint that the frontend will call."""
    current_positions = get_current_robot_joint_positions()
    return jsonify(current_positions)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)