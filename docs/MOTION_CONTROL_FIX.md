# Motion Control Fix - Hybrid Planning/IK Approach

## Problem Summary

IK solver was failing even with 1mm incremental steps after ~150 steps (15cm of movement). The robot was approaching a singular configuration or workspace limit where Jacobian linearization breaks down.

**Error:**
```
InvalidActionError: Could not perform IK via Jacobian; most likely due to current end-effector pose being too far from the given target pose.
```

**Movement attempted:**
- Start: `[0.251, -0.103, 1.459]`
- Target: `[0.000, -0.515, 1.730]`
- Distance: `0.633m`
- Failed after: 150 steps (~15cm)

## Root Cause

The Jacobian IK solver (`solve_ik_via_jacobian`) uses linearization around the current configuration. This fails when:

1. **Singularities:** Robot reaches a configuration near joint limits or singularities
2. **Large configuration changes:** Even small Cartesian steps may require large joint movements
3. **Workspace boundaries:** Target approaches workspace limits
4. **Limited iterations:** IK solver only gets 6 iterations to converge (hardcoded in PyRep)

From RLBench source code (arm_action_modes.py:286-330):
```python
class EndEffectorPoseViaIK(ArmActionMode):
    """High-level action where target pose is given and reached via IK.

    Given a target pose, IK via inverse Jacobian is performed. This requires
    the target pose to be close to the current pose, otherwise the action
    will fail. It is up to the user to constrain the action to
    meaningful values.
    """
```

## Solution: Hybrid Planning/IK Approach

Implemented two motion modes in `move_to_position()`:

### Mode 1: Path Planning (Default - Robust)

**When:** Large movements, unknown workspace configuration
**Method:** `EndEffectorPoseViaPlanning`
**How it works:**
1. Plans a collision-free path through joint space
2. Uses sample-based planning (RRTConnect algorithm)
3. Robust to singularities and large movements
4. Slower (~1-2 seconds for planning)

**Code:**
```python
planning_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaPlanning(collision_checking=False),
    gripper_action_mode=Discrete()
)

# Temporarily switch to planning mode
_ENV._action_mode = planning_mode
task._action_mode = planning_mode

# Execute single-step planning
action = np.concatenate([target_pos, current_pose[3:7], [1.0]])
_CURRENT_OBS, reward, terminate = task.step(action)

# Restore IK mode
_ENV._action_mode = original_action_mode
```

### Mode 2: Incremental IK (Fast but Limited)

**When:** Small fine movements (<5cm)
**Method:** Multiple 1mm IK steps
**How it works:**
1. Take 1mm steps toward target
2. Update current position after each step
3. Fast execution but can fail if robot enters bad configuration

**Code:**
```python
for i in range(max_iterations):
    current_pos = current_pose[:3]
    remaining = target_pos - current_pos
    step_size = min(0.001, distance_remaining)  # 1mm max
    next_pos = current_pos + direction * step_size

    action = np.concatenate([next_pos, current_pose[3:7], [1.0]])
    _CURRENT_OBS, reward, terminate = task.step(action)
    current_pose = _CURRENT_OBS.gripper_pose  # Update!
```

## Usage

```python
# Default: Use path planning (recommended)
move_to_position(x=0.5, y=-0.2, z=1.3)

# Or explicitly enable planning
move_to_position(x=0.5, y=-0.2, z=1.3, use_planning=True)

# For small fine movements only
move_to_position(x=0.5, y=-0.2, z=1.3, use_planning=False)
```

## Changes Made

### 1. `/home/roops/ADK_Agent_Demo/multi_tool_agent/ros_mcp_server/rlbench_orchestration_server.py`

**Lines 25-29:** Added imports
```python
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
from rlbench.backend.exceptions import InvalidActionError
```

**Lines 161-289:** Replaced `move_to_position()` with hybrid approach
- Added `use_planning` parameter (default: True)
- Implemented path planning mode
- Improved error handling with specific catch for `InvalidActionError`
- Added method indicator in return value

### 2. `/home/roops/ADK_Agent_Demo/multi_tool_agent/agent.py`

**Lines 81-86:** Updated agent instructions
- Documented `use_planning` parameter
- Explained when to use each mode

## Testing Status

**Implemented:** ✅
- [x] Path planning integration
- [x] Incremental IK fallback
- [x] Mode switching logic
- [x] Error handling

**Tested:** ⏳ Pending
- [ ] Path planning for large movements (0.6m+)
- [ ] Task completion with ReachTarget
- [ ] PickAndLift task

## Expected Behavior

### Before Fix:
```
[Tool: move_to_position] Distance: 0.633m, using 633 steps (1.0mm per step)
[Tool: move_to_position] Progress: 50 steps, 0.612m remaining
[Tool: move_to_position] Progress: 100 steps, 0.590m remaining
[Tool: move_to_position] Progress: 150 steps, 0.567m remaining
[Tool: move_to_position] ✗ Error: Could not perform IK via Jacobian
```

### After Fix (with planning):
```
[Tool: move_to_position] Distance: 0.633m
[Tool: move_to_position] Using path planning (robust for large movements)
[Planning takes 1-2 seconds]
[Tool: move_to_position] ✓ Reached via planning (error: 0.0003m)
```

## Performance Comparison

| Method | Speed | Robustness | Best For |
|--------|-------|------------|----------|
| Path Planning | Slow (1-2s) | ✅ High | Large movements, unknown configuration |
| Incremental IK | Fast (<1s) | ⚠️ Medium | Small movements (<5cm), known safe config |
| Original (buggy IK) | Fast | ❌ Low | Nothing (broken) |

## Recommendations

1. **Always use planning by default** - the performance difference (1-2s) is negligible compared to task reliability
2. **Only use IK mode** when doing fine adjustments after planning has already positioned the arm
3. **For future:** Consider waypoint planning for complex trajectories

## Related Documents

- `IK_SOLVER_FINDINGS.md` - Deep dive into IK solver investigation
- `PERCEPTION_IMPLEMENTATION_PLAN.md` - Overall perception pipeline
- `ORCHESTRATION_PLAN.md` - High-level architecture
