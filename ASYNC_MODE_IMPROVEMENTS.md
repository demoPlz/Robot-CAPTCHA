# Asynchronous Mode Improvements

## Overview
Three key improvements have been implemented to make async mode more robust and user-friendly:

## 1. Automatic Finalization After Admin Collection

**Location**: [collect_data.py](backend/collect_data.py#L221-L234)

**Behavior**: After the episode collection loop completes (line 211), the system automatically finalizes the async pool, moving all completed critical states to the pool for random serving.

```python
# Auto-finalize async pool if in async mode
if _CROWD_CONFIG.asynchronous_mode:
    result = crowd_interface.state_manager.finalize_admin_phase()
    # Logs states ready for user labeling
```

**Why**: This eliminates the need for manual finalization - the system knows admin phase is complete when the loop exits.

## 2. Smart Pre-Approval Behavior in Async Mode

**Location**: [state_manager.py](backend/interface_managers/state_manager.py#L870-L897)

### Admin Submissions (Localhost)
- **Auto-approved**: Admin submissions skip the pre-approval modal entirely
- **Direct to post-approval**: Goes straight to critical state approval (accept, undo, jitter, mark as end, set prompt)
- **Execution happens immediately**: Robot executes after 1 admin submission (or auto-fills for gripper-only)

```python
if self.asynchronous_mode and is_admin_submission:
    # Auto-approve admin submission - skip pre-approval
    state_info["execution_history"] = [{
        "action": state_info["actions"][0],
        "approval": 1,  # Auto-approved
        ...
    }]
    state_info["pre_approval_loop_complete"] = True
```

### User Submissions (Via Netlify)
- **Queued for review**: Each submission goes to pre-approval modal one-by-one
- **No overlapping modals**: Submissions are queued and reviewed sequentially
- **Order preserved**: Review in the order submissions arrive

```python
elif self.asynchronous_mode and self.async_pool_finalized and not is_admin_submission:
    # Queue for one-by-one pre-approval review
    should_run_pre_approval = True
    state_info_copy = state_info.copy()
```

## 3. Gripper-Only Action Auto-Fill

**Location**: [state_manager.py](backend/interface_managers/state_manager.py#L710-L730)

**Detection**: Compares submitted joint positions with current state - if only gripper changed (< 0.001 radian threshold for all other joints), it's gripper-only.

**Behavior**: 
1. Instantly fills all `required_responses_per_critical_state` slots with the same action
2. Marks state as `gripper_only_autofilled = True`
3. State **NOT** added to async pool (skipped during finalization)
4. Users never see these states

```python
# Check if all joint positions are the same (only gripper changed)
position_changed = False
for joint_name in JOINT_NAMES[:-1]:  # Exclude gripper (last joint)
    if abs(submitted_val - current_val) > 0.001:
        position_changed = True
        break

is_gripper_only = not position_changed

if is_gripper_only:
    # Fill all remaining slots
    required_responses = self.required_responses_per_critical_state
    # ... auto-fill logic ...
    state_info["gripper_only_autofilled"] = True
```

**Why**: Gripper open/close actions are unambiguous and don't benefit from crowd labeling - no need to waste user time on these.

## Finalization Logic Enhancement

**Location**: [state_manager.py](backend/interface_managers/state_manager.py#L2331-L2360)

When finalizing the admin phase, gripper-only states are automatically skipped:

```python
for episode_id, states in self.completed_states_by_episode.items():
    for state_id, state_info in states.items():
        if state_info.get("critical", False):
            is_gripper_only = state_info.get("gripper_only_autofilled", False)
            
            if is_gripper_only:
                states_skipped += 1
                # Don't add to async pool
            else:
                self.async_state_pool[pool_key] = state_info
                states_moved += 1
```

## Example Workflow

### Admin Phase (10 episodes, 10 states each):
1. Admin collects 100 critical states from localhost
2. 20 are gripper-only → instantly auto-filled to 10 responses each
3. 80 require position changes → complete after 1 admin response each
4. Robot executes immediately after each admin submission
5. Loop ends → **automatic finalization**

### Finalization:
- Skips 20 gripper-only states (already complete)
- Moves 80 position-change states to async pool
- Logs: "80 states ready for labeling, skipped 20 gripper-only"

### User Phase:
1. Users access via Netlify
2. Get random states from pool of 80
3. Each submission goes to pre-approval modal **one at a time**
4. Admin reviews: approve/reject each submission in order received
5. Collect 9 more responses per state = 720 total user responses

## Benefits

1. **Streamlined Admin Experience**: Auto-approval means no modal interruptions during collection
2. **Quality Control**: User submissions still reviewed one-by-one for quality
3. **Efficiency**: Gripper-only actions don't waste user time
4. **Automatic Workflow**: No manual finalization needed
5. **Backward Compatible**: All changes only active when `asynchronous_mode = True`

## Configuration

Enable in [crowd_interface_config.py](backend/crowd_interface_config.py):

```python
self.asynchronous_mode = True
self.async_admin_responses_per_state = 1  # Admin completes states with 1 response
self.required_responses_per_critical_state = 10  # Total responses needed
```

With these settings:
- Admin provides 1 response per state (immediate execution)
- System auto-fills gripper-only states completely
- Remaining states need 9 more user responses each
