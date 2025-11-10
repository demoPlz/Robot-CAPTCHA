# ArUco Marker Tools

Three scripts for tracking and calibrating ArUco markers on the drawer using RealSense D455.

---

## 1. `track_aruco_realsense.py`

**Purpose:** Live visualization of ArUco marker detection and pose estimation.

**Use case:** Debugging marker detection, verifying camera setup, checking marker visibility.

**Usage:**
```bash
# Auto-detect dictionary (tries all)
python utility/track_aruco_realsense.py --serial 151422252817

# Specify dictionary
python utility/track_aruco_realsense.py --serial 151422252817 --dict 5X5_50 --marker-size 0.04
```

**Output:** Live video feed with detected markers, 3D axes, and position coordinates.

---

## 2. `calibrate_drawer_markers.py`

**Purpose:** Calibrate drawer marker positions at closed position (when only marker ID=1 is visible).

**Use case:** One-time setup to establish reference positions for drawer tracking.

**Workflow:**
1. **Step 1:** Close drawer → capture marker ID=1
2. **Step 2:** Open drawer → capture markers ID=0, 1, 2
3. **Result:** Computes world-frame positions of all markers at closed position using rigid body math

**Usage:**
```bash
python utility/calibrate_drawer_markers.py --serial 151422252817 --marker-size 0.04
```

**Output:** `data/calib/drawer_markers_closed.json` (reference positions for tracking)

---

## 3. `track_drawer_position.py`

**Purpose:** Real-time tracking of drawer position (distance from closed in meters/cm).

**Use case:** Monitor drawer state during teleoperation or data collection.

**Usage:**
```bash
python utility/track_drawer_position.py --serial 151422252817
```

**Output:** Live video with drawer distance displayed (e.g., "Drawer: 15.3 cm")

**Note:** Requires `drawer_markers_closed.json` from calibration script.

---

## Typical Workflow

1. **Setup (once):** Run `calibrate_drawer_markers.py` to generate reference positions
2. **Debug (as needed):** Use `track_aruco_realsense.py` to verify marker detection
3. **Track (runtime):** Use `track_drawer_position.py` during operation

---

## Common Parameters

- `--serial 151422252817` - RealSense D455 camera serial number
- `--dict 5X5_50` - ArUco dictionary (default for drawer markers)
- `--marker-size 0.04` - Physical marker size in meters (4cm for drawer)
- `--width 640 --height 480` - Camera resolution (optional)

---

## Dependencies

- Camera calibration files:
  - `data/calib/intrinsics_realsense_d455.npz`
  - `data/calib/extrinsics_realsense_d455.npz`
- Drawer calibration (for tracking only):
  - `data/calib/drawer_markers_closed.json`
