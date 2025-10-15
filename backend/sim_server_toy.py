# pip install flask
from flask import Flask, jsonify
import random, math

app = Flask(__name__)

# Example joints converted from [0, 60, 75, -60, 0, 0, 2] degrees → radians
Q_EXAMPLE_RAD = [
    math.radians(0.0),
    math.radians(60.0),
    math.radians(75.0),
    math.radians(-60.0),
    math.radians(0.0),
    math.radians(0.0),
    math.radians(2.0),
]

def rand_pose(z):
    return {
        "pos": [random.uniform(0.0, 0.75), random.uniform(0.0, 0.5), z],
        "rot": [1.0, 0.0, 0.0, 0.0],  # w, x, y, z (identity)
    }

@app.get("/initial-state")
def initial_state():
    objects = {
        "Cube_01": rand_pose(1.025),
        "Cube_02": rand_pose(1.025),
        "Tennis":  rand_pose(1.0335),
    }
    return jsonify({"q": Q_EXAMPLE_RAD, "objects": objects})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, debug=False)