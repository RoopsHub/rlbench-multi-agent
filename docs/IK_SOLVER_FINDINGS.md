# RLBench IK Solver Investigation - Findings

## Problem
Persistent `InvalidActionError: "Could not perform IK via Jacobian"` failures even with very small 3mm step sizes (211 steps for 0.633m movement).

## Root Causes Identified

### 1. **Interpolation Bug (FIXED)**
**Location:** `rlbench_orchestration_server.py:195` (old code)

**Problem:**
```python
current_pos = current_pose[:3]  # Captured ONCE at start
for i in range(1, num_steps + 1):
    alpha = i / num_steps
    interp_pos = current_pos + alpha * (target_pos - current_pos)  # USES ORIGINAL position!
```

Even with 211 steps:
- Step 1: Move from `current_pos` by `(1/211) * total_distance` = 3mm
- Step 2: Move from `current_pos` by `(2/211) * total_distance` = 6mm from ORIGINAL
- Step 3: Move from `current_pos` by `(3/211) * total_distance` = 9mm from ORIGINAL

The IK solver sees each command as "move 3mm/6mm/9mm from the ORIGINAL position", not incremental 3mm steps.

**Fix Applied:**
```python
for i in range(max_iterations):
    current_pos = current_pose[:3]  # GET UPDATED POSITION EACH ITERATION
    remaining = target_pos - current_pos
    distance_remaining = np.linalg.norm(remaining)

    step_size = min(max_step_size, distance_remaining)
    direction = remaining / distance_remaining
    next_pos = current_pos + direction * step_size  # TRUE 1mm step from current

    current_pose = _CURRENT_OBS.gripper_pose  # Update for next iteration
```

Now each step is truly 1mm from the CURRENT position.

### 2. **IK Solver Limitations (Understanding from Code)**
**Location:** `/home/roops/RLBench/venv/lib/python3.13/site-packages/pyrep/robots/arms/arm.py:200-245`

**Key Documentation:**
```python
def solve_ik_via_jacobian(...):
    """
    This IK method performs a linearisation around the current robot
    configuration via the Jacobian. The linearisation is valid when the
    start and goal pose are not too far away, but after a certain point,
    linearisation will no longer be valid.
    """
```

**Implementation:**
```python
ik_result, joint_values = sim.simCheckIkGroup(
    self._ik_group, [j.get_handle() for j in joints])

if ik_result == sim.sim_ikresult_fail:
    raise IKError('IK failed. Perhaps the distance was between the tip '
                  ' and target was too large.')
```

**What "too far" means:**
- NOT just Cartesian distance (3mm)
- Also considers:
  - **Orientation changes** (quaternion difference)
  - **Joint space distance** (Jacobian linearization breaks down)
  - **Workspace singularities** (near joint limits or singular configurations)
  - **Current robot configuration** (may be in awkward pose)

### 3. **IK Group Configuration**
**Location:** `arm.py:62-75`

```python
def set_ik_group_properties(self, resolution_method='pseudo_inverse',
                             max_iterations=6, dls_damping=0.1):
    res_method = {
        'pseudo_inverse': sim.sim_ik_pseudo_inverse_method,
        'damped_least_squares': sim.sim_ik_damped_least_squares_method,
        'jacobian_transpose': sim.sim_ik_jacobian_transpose_method
    }[resolution_method]
```

**Default:**
- Method: `pseudo_inverse` (Jacobian pseudo-inverse)
- Max iterations: **6** (very low!)
- The solver only gets 6 iterations to converge

## Why Previous Approach Failed

### Original Code (BROKEN):
1. Calculate total distance: 0.633m
2. Use 211 steps (3mm each)
3. **Loop 211 times:**
   - Each iteration: "Move to position that's `alpha * 0.633m` from ORIGINAL start"
   - IK solver sees: "Move 3mm/6mm/9mm... from original" (not incremental)
   - **First step already fails** because position hasn't been updated

### Fixed Code:
1. Calculate total distance: 0.633m
2. Max step: 1mm
3. **Loop until within 1mm:**
   - Get CURRENT position
   - Calculate remaining distance
   - Move 1mm toward target FROM CURRENT position
   - Update current position
   - **IK solver only sees 1mm incremental moves**

## Additional Considerations

### When IK Can Still Fail:
1. **Orientation mismatch:** Even if position is 1mm away, if orientation changes significantly, IK may fail
2. **Singularities:** Near joint limits or singularities, even tiny movements fail
3. **Collision:** If collision checking enabled and path blocked

### Alternative: `solve_ik_via_sampling`
**Location:** `arm.py:77-177`

For large movements (>5cm), use sampling-based IK:
```python
def solve_ik_via_sampling(..., trials=300, max_configs=1):
    """
    This IK method performs a random searches for manipulator configurations
    that matches the given end-effector pose in space.

    This is the method that should be used when the start pose is far
    from the end pose.
    """
```

## Recommendations

### Current Implementation (✓ DONE):
- [x] Fixed interpolation to use updated position each step
- [x] Reduced step size to 1mm (from 3mm)
- [x] Early termination when within 1mm of target
- [x] Safety limit on max iterations (distance/0.001 + 100)

### Future Improvements:
1. **Hybrid approach:**
   - Use `solve_ik_via_sampling` for large movements (>5cm)
   - Use `solve_ik_via_jacobian` for fine movements (<5cm)

2. **Orientation handling:**
   - Keep orientation constant during movement (already doing this)
   - Or: Interpolate orientation separately using SLERP

3. **Error recovery:**
   - If IK fails, try smaller step size (0.5mm)
   - If still fails, switch to sampling-based IK

4. **Alternative action mode:**
   - Use `EndEffectorPoseViaPlanning` instead of `EndEffectorPoseViaIK`
   - Planning mode uses path planning (slower but more robust)

## Test Strategy

1. **Verify fix works:**
   - Run orchestrator with "Reach the red target"
   - Check that 1mm incremental steps succeed
   - Validate robot reaches detected target

2. **Stress test:**
   - Try movements from different starting configurations
   - Test near workspace boundaries
   - Verify works for PickAndLift task

## Sensor Data Folder Change

**Requested:** Store sensor data in `/home/roops/ADK_Agent_Demo/sensor_data/` instead of `/tmp`

**Status:** ✓ COMPLETED in `rlbench_orchestration_server.py:42`:
```python
OUTPUT_DIR = Path(__file__).parent.parent.parent / "sensor_data"
OUTPUT_DIR.mkdir(exist_ok=True)
```

This resolves to: `/home/roops/ADK_Agent_Demo/sensor_data/`
