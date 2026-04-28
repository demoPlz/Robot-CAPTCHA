// ============================================================================
// PER-TASK CONFIGURATION
// Limits, spans, and home positions for each task.
// Add new tasks here; unknown tasks fall back to '_default'.
// ============================================================================
const TASK_CONFIGS = {
  _default: {
    absLimits: {
      x:     { min: 0.10,  max: 0.63  },
      y:     { min: -0.30, max: 0.30  },
      z:     { min: 0.025, max: 0.2798  },
      roll:  { min: -1.9,  max: 1.9   },
      pitch: { min: -1,    max: 0.79  },
      yaw:   { min: -0.81, max: 0.81  },
    },
    sliderSpans: {
      x: 0.20, y: 0.40, z: 0.20,           // meters
      roll: 1.0, pitch: 0.81, yaw: 0.81,   // radians
    },
    homePositionDeg: [0, 60, 75, -60, 0, 0, 2],
    homeGripper: 1,    // 1 = open, -1 = closed
  },
  drawer: {
    absLimits: {
      x:     { min: 0.10,  max: 0.63  },
      y:     { min: -0.30, max: 0.30  },
      z:     { min: 0.025, max: 0.40  },
      roll:  { min: -1.9,  max: 1.9   },
      pitch: { min: -1,    max: 1     },
      yaw:   { min: -0.81, max: 0.81  },
    },
    sliderSpans: {
      x: 0.20, y: 0.40, z: 0.20,
      roll: 1.0, pitch: 0.81, yaw: 0.81,
    },
    homePositionDeg: [0, 60, 75, -60, 0, 0, 2],
    homeGripper: 1,
  },
  pour: {
    absLimits: {
      x:     { min: 0.10,  max: 0.63  },
      y:     { min: -0.30, max: 0.30  },
      z:     { min: 0.025, max: 0.40  },
      roll:  { min: -1.9,  max: 1.9   },
      pitch: { min: -1,    max: 1     },
      yaw:   { min: -0.81, max: 0.81  },
    },
    sliderSpans: {
      x: 0.20, y: 0.40, z: 0.20,
      roll: 1.0, pitch: 0.81, yaw: 0.81,
    },
    homePositionDeg: [0, 60, 75, -60, 0, 0, 2],
    homeGripper: 1,
  },
  insertion: {
    absLimits: {
      x:     { min: 0.10,  max: 0.63  },
      y:     { min: -0.30, max: 0.30  },
      z:     { min: 0.025, max: 0.40  },
      roll:  { min: -1.9,  max: 1.9   },
      pitch: { min: -1,    max: 1     },
      yaw:   { min: -0.81, max: 0.81  },
    },
    sliderSpans: {
      x: 0.20, y: 0.40, z: 0.20,
      roll: 1.0, pitch: 0.81, yaw: 0.81,
    },
    homePositionDeg: [0, 100, 80, -67, 0, 0, -1],
    homeGripper: -1,
  },
  switches: {
    absLimits: {
      x:     { min: 0.10,  max: 0.63  },
      y:     { min: -0.30, max: 0.30  },
      z:     { min: 0.025, max: 0.40  },
      roll:  { min: -1.9,  max: 1.9   },
      pitch: { min: -1,    max: 1     },
      yaw:   { min: -0.81, max: 0.81  },
    },
    sliderSpans: {
      x: 0.20, y: 0.40, z: 0.20,
      roll: 1.0, pitch: 0.81, yaw: 0.81,
    },
    homePositionDeg: [0, 100, 90, -78, 0, 0, -1],
    homeGripper: -1,
  },
  sorting: {
    absLimits: {
      x:     { min: 0.10,  max: 0.63  },
      y:     { min: -0.30, max: 0.30  },
      z:     { min: 0.025, max: 0.2798 },
      roll:  { min: -1.9,  max: 1.9   },
      pitch: { min: -1,    max: 0.79  },
      yaw:   { min: -0.81, max: 0.81  },
    },
    sliderSpans: {
      x: 0.20, y: 0.40, z: 0.20,
      roll: 1.0, pitch: 0.81, yaw: 0.81,
    },
    homePositionDeg: [0, 60, 75, -60, 0, 0, 2],
    homeGripper: 1,
    // Preset joint angles (degrees) for positioning above each container
    containerPresets: [
      { key: 'red',    color: '#ef4444', label: '', deg: [-62.6, 69.7, 83.3, -79.0, -38.9, -53.9, -1] },
      { key: 'yellow', color: '#eab308', label: '', deg: [-41.5, 79.3, 88.1, -62.1, -27.9, -32.1, -1] },
      { key: 'green',  color: '#22c55e', label: '', deg: [-30, 95.6, 102.9, -56.5, -20.7, -22.3, -1] },
    ],
  },
};

/** Get config for the current task (falls back to _default). */
function getTaskConfig() {
  const name = window.__INIT_STATE?.task_name;
  const cfg = TASK_CONFIGS[name] || TASK_CONFIGS._default;
  
  const pushDownPreset = { 
    key: 'down', 
    color: '#475569', 
    label: '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="4" x2="12" y2="16"></line><polyline points="7 11 12 16 17 11"></polyline><line x1="5" y1="20" x2="19" y2="20"></line></svg>', 
    action: 'push_down', 
    title: 'Push straight down to the bottom' 
  };
  
  // Shallow clone and inject the push_down preset for all tasks
  const clone = Object.assign({}, cfg);
  clone.containerPresets = clone.containerPresets ? [...clone.containerPresets] : [];
  
  if (!clone.containerPresets.find(p => p.action === 'push_down')) {
    clone.containerPresets.push(pushDownPreset);
  }
  
  return clone;
}
