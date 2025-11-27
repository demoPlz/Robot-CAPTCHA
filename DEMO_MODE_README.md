# Demo Mode Instructions

## Overview

`demo.html` is a standalone version of `sim.html` that works **without a backend server**. It uses embedded state data instead of fetching from the Flask API.

## Key Differences from sim.html

- **No backend required**: All state data is embedded in the HTML file
- **"DEMO MODE" badge**: Visible in the top-right corner
- **Same functionality**: Full 3D robot visualization, IK controls, gripper controls, camera views
- **Disabled features** in demo mode:
  - Animation/replay (Isaac Sim integration)
  - Recording demos
  - Polling for new states
  - Backend API calls for submission

## How to Use Demo Mode

### Step 1: Collect State Data

You need to save actual state data from your running backend. There are two methods:

#### Method A: From Browser Console (Recommended)

1. Run your backend server normally
2. Open `sim.html` in your browser
3. Open browser DevTools (F12) and go to the Console tab
4. Look for the log that says "Successfully fetched initial state"
5. Expand the logged state object
6. Right-click on the object and select "Copy object" or "Store as global variable"
7. Save this JSON data

#### Method B: Direct API Call

```bash
curl http://127.0.0.1:9000/api/get-state > state_data.json
```

### Step 2: Add States to demo.html

1. Open `demo.html` in a text editor
2. Find the `DEMO_STATES` array (around line 1520)
3. Paste your saved state data into the array:

```javascript
const DEMO_STATES = [
  {
    "state_id": 0,
    "episode_id": "episode_001",
    "prompt": "Pick up the red block",
    "views": {
      "front": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
      "left": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
      "right": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    },
    "joint_positions": {
      "joint_0": 0.0,
      "joint_1": -0.5,
      "joint_2": 0.3,
      "joint_3": 0.0,
      "joint_4": 0.2,
      "joint_5": 0.0,
      "left_carriage_joint": 0.015
    },
    "gripper": 0,
    "camera_models": { /* ... */ },
    "camera_poses": { /* ... */ },
    "gripper_tip_calib": { /* ... */ },
    "controls": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
  },
  // Add more states here...
  {
    "state_id": 1,
    "episode_id": "episode_001",
    // ... next state
  }
];
```

### Step 3: Open demo.html

Simply open the file in any modern web browser:
- Double-click `demo.html`, or
- Drag and drop into browser, or
- Use a local file server: `python -m http.server 8000`

No backend needed!

## Required State Fields

Each state object must include:

### Essential Fields
- `state_id`: Unique identifier (number)
- `episode_id`: Episode identifier (string)
- `prompt`: Task description shown to user (string)
- `views`: Dictionary of camera views with base64 JPEG data
- `joint_positions`: Dictionary of joint names to positions (floats)
- `gripper`: Gripper state (-1, 0, or 1)
- `camera_models`: Camera intrinsics for each view
- `camera_poses`: Camera extrinsics (4x4 matrices)
- `gripper_tip_calib`: Gripper fingertip calibration
- `controls`: Array of enabled controls

### Optional Fields
- `text_prompt`: VLM-generated prompt (overrides `prompt`)
- `video_prompt`: Example video ID
- `example_video_url`: URL to example video
- `left_carriage_external_force`: Gripper force reading

## Behavior

1. **On Load**: Displays first state from `DEMO_STATES`
2. **On Submit**: Logs the goal, then reloads page with next state
3. **State Cycling**: Cycles through states sequentially
4. **No Data Loss**: Goals are logged to console but not saved (demo mode)

## Deployment

Demo mode can be deployed anywhere that serves static HTML:
- GitHub Pages
- Netlify
- Vercel
- Any CDN or web server
- Even locally (file://)

## Toggling Demo Mode

To switch back to backend mode, edit `demo.html`:

```javascript
const DEMO_MODE = false;  // Change true to false
```

Or just use `sim.html` instead.

## Tips

- **Multiple States**: Add 5-10 states for a realistic demo experience
- **Image Size**: Base64 JPEGs can be large; consider lower quality for smaller file size
- **State Variety**: Include different tasks, poses, and scenarios
- **Console Logs**: Check browser console for demo mode messages

## Troubleshooting

**"No demo states available" error**:
- The `DEMO_STATES` array is empty
- Add at least one complete state object

**Robot doesn't appear**:
- Check that `joint_positions` are valid
- Verify `camera_models` and `camera_poses` are present

**Views don't load**:
- Ensure base64 JPEG strings are complete
- Check that view keys match camera model keys

**Submit doesn't work**:
- This is normal - demo mode just reloads the page
- Check console for logged payloads
