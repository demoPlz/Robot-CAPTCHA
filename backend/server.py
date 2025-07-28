import time
import json
from flask import Flask, jsonify
from flask_cors import CORS

# --- This function is our "Robot Hardware Driver" ---
# In the REAL system, this would use the trossen_arm library to get live data.
# For this example, we simulate it to return dynamic, non-hardcoded values.
def get_current_robot_joint_positions():
    """
    SIMULATED: Reads the current joint angles from the physical robot hardware.
    
    IN A REAL SCENARIO, this would contain:
    # positions = driver.get_all_positions() 
    # return { f"joint_{i}": positions[i] for i in range(len(positions)) }
    """
    print("Reading 'live' joint positions from the (simulated) robot...")
    
    # We use the current time to generate slightly different values on each run
    # to prove the data is being fetched dynamically.
    base_angle = time.time() % 3.0  # A value that changes over time
    
    # Names must EXACTLY match the joint names in your URDF file
    simulated_positions = {
        "joint_0": 0.0,
        "joint_1": base_angle / 2.0,  # e.g., 0.7 rad
        "joint_2": base_angle,        # e.g., 1.4 rad
        "joint_3": -base_angle / 2.0, # e.g., -0.7 rad
        "joint_4": 0.0,
        "joint_5": 0.0,
        "joint_6": 0.0
    }
    return simulated_positions

# --- Standard Flask Server Setup ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin requests

@app.route('/api/get-current-pose')
def get_current_pose():
    """The API endpoint that the frontend will call."""
    current_positions = get_current_robot_joint_positions()
    return jsonify(current_positions)

if __name__ == '__main__':
    # Runs the backend server
    app.run(debug=True, host='0.0.0.0', port=5000)