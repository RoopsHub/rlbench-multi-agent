# RLBench Task Understanding

## Task Analysis

### 1. ReachTarget Task

**File:** `/home/roops/RLBench/rlbench/tasks/reach_target.py`

**Objective:** Move the robot gripper to touch a colored target sphere

**Scene Objects:**
- `target` - The colored sphere to reach (changes color per episode)
- `distractor0`, `distractor1` - Two distractor spheres with different colors
- `boundaries` - Spawn boundary defining where objects can appear
- `success` - ProximitySensor that detects when gripper touches target

**Success Condition:**
```python
DetectedCondition(self.robot.arm.get_tip(), success_sensor)
```
- Task succeeds when gripper tip is detected by the proximity sensor
- Must physically reach and touch the target sphere

**Episode Initialization:**
1. Target gets a color from `colors[index]` (red, blue, green, etc.)
2. Two distractors get different random colors
3. All three spheres are randomly spawned within boundaries
4. Minimum distance between objects: 0.2m
5. No rotation variation (always upright)

**Ground Truth Access:**
```python
def get_low_dim_state(self) -> np.ndarray:
    return np.array(self.target.get_position())
```
- Can access actual target position via `task.target.get_position()`
- Position is in robot base frame (world coordinates)

**Reward Function:**
```python
return -np.linalg.norm(self.target.get_position() -
                      self.robot.arm.get_tip().get_position())
```
- Negative distance between gripper and target
- Closer = higher reward

### 2. PickAndLift Task

**File:** `/home/roops/RLBench/rlbench/tasks/pick_and_lift.py`

**Objective:** Pick up a colored block and lift it to a target height

**Scene Objects:**
- `target_block` - The colored block to pick (changes color per episode)
- `distractor` blocks (2) - Distractor blocks with different colors
- `boundary` - Spawn boundary for block placement
- `success_detector` - ProximitySensor at target lift height

**Success Conditions:**
```python
ConditionSet([
    GraspedCondition(self.robot.gripper, self.target_block),
    DetectedCondition(self.target_block, self.success_detector)
])
```
- Must grasp the target block AND
- Lift it to the height where success detector is located
- Both conditions must be satisfied

**Episode Initialization:**
1. Target block gets a color from `colors[index]`
2. Two distractor blocks get different random colors
3. Success detector position is randomly sampled within boundary
4. All blocks randomly spawned with minimum distance 0.1m

**Ground Truth Access:**
```python
def get_low_dim_state(self) -> np.ndarray:
    return np.concatenate([
        self.target_block.get_position(),
        self.success_detector.get_position()
    ], 0)
```
- Can access block position: `task.target_block.get_position()`
- Can access lift target: `task.success_detector.get_position()`

**Graspable Objects:**
```python
self.register_graspable_objects([self.target_block])
```
- Only target block is registered as graspable
- Distractors cannot be picked up

## How Tasks Are Controlled (Without Demos)

### Action Space

Both tasks use the action mode we configured:
```python
MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaIK(),  # or EndEffectorPoseViaPlanning()
    gripper_action_mode=Discrete()
)
```

**Action format:** `[x, y, z, qx, qy, qz, qw, gripper]`
- `x, y, z`: Target end-effector position in base frame
- `qx, qy, qz, qw`: Target orientation (quaternion)
- `gripper`: 1.0 = open, 0.0 = closed

### ReachTarget Control Strategy

**Without perception:**
1. Get ground truth: `target_pos = task.target.get_position()`
2. Move to target: `action = [target_pos, current_orientation, 1.0]`
3. Success when proximity sensor detects gripper

**With perception (our approach):**
1. Capture camera observation
2. Detect colored target in image
3. Convert pixel + depth → 3D position in base frame
4. Move to detected position
5. Success when proximity sensor detects gripper

### PickAndLift Control Strategy

**Without perception:**
1. Get block position: `block_pos = task.target_block.get_position()`
2. Move above block: `[block_pos[0], block_pos[1], block_pos[2] + 0.1, ...]`
3. Move down to block: `[block_pos, ..., 1.0]` (gripper open)
4. Close gripper: `[block_pos, ..., 0.0]`
5. Get lift target: `lift_pos = task.success_detector.get_position()`
6. Lift to target: `[lift_pos[0], lift_pos[1], lift_pos[2], ..., 0.0]`

**With perception (our approach):**
1. Capture camera observation
2. Detect colored block in image
3. Convert to 3D position
4. Move to grasp position (approach from above)
5. Close gripper
6. Detect lift target (or use known height)
7. Lift to target height

## Workspace Boundaries

The spawn boundaries determine where objects can appear. This defines the **reachable workspace**.

**Key insight:** Objects spawned by RLBench are ALWAYS within the robot's reachable workspace. If our detected position is outside workspace, our perception/transformation is wrong.

## Current Issues

### Problem: Detected Positions Outside Workspace

**Latest detection:**
- Detected: `[0.034, -0.566, 1.778]`
- Error: "target is outside of workspace"

**This means:**
1. Color detection found the red object ✓
2. Depth reading seems reasonable (0.315m) ✓
3. **Coordinate transformation is incorrect** ✗

### Possible Causes

1. **Camera extrinsics interpretation:**
   - We're using `obs.misc['front_camera_extrinsics']` as base→camera
   - Maybe it's actually camera→base?
   - Or needs transpose?

2. **Coordinate frame conventions:**
   - RLBench/CoppeliaSim might use different conventions
   - Y-up vs Z-up?
   - Right-handed vs left-handed?

3. **Intrinsics extraction:**
   - We extract [fx, fy, cx, cy] from 3x3 matrix
   - Maybe wrong indices or negative values are significant?

## Next Steps

### 1. Add Ground Truth Comparison Tool ✓

Created `get_target_position()` tool that returns:
- Actual target position from simulation
- Can compare with detected position
- Identify transformation error pattern

### 2. Test Pattern Recognition

Run test with ground truth comparison:
```python
# Get both positions
ground_truth = get_target_position()  # e.g., [0.5, -0.2, 1.0]
detected = detect_object_3d(...)       # e.g., [0.034, -0.566, 1.778]

# Analyze difference
diff = detected - ground_truth
# Look for patterns:
# - Axis swap? (X→Y, Y→Z, etc.)
# - Sign flip? (-X, -Y, -Z)
# - Offset? (constant translation)
# - Scale? (wrong units?)
```

### 3. Fix Transformation

Based on pattern, fix one of:
- Invert extrinsics matrix
- Transpose extrinsics
- Change coordinate frame interpretation
- Adjust intrinsics signs

### 4. Validate With Multiple Positions

Test with multiple object positions to ensure fix is general, not just for one case.

## Academic Requirements

For your framework to be academically valid:
1. ✓ **Task-agnostic:** Works for multiple tasks (ReachTarget, PickAndLift)
2. ✓ **Perception-based:** Uses camera observations, not ground truth
3. ✓ **Multi-agent:** Orchestrator coordinates sensing/perception/motion agents
4. ⏳ **Benchmarked:** Evaluated on RLBench standardized tasks
5. ⏳ **Success rate:** Must actually complete tasks

Using demos would violate requirements 2 and 3. You're absolutely right to insist on fixing perception.
