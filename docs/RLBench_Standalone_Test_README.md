# RLBench Standalone Simulation Test

## Purpose
This test verifies that RLBench works correctly with:
- ✅ Franka Panda robot simulation
- ✅ CoppeliaSim GUI visualization
- ✅ RGB camera image capture
- ✅ Depth image capture
- ✅ Point cloud generation
- ✅ Robot motion control

**No agents, no MCP, no framework - just pure RLBench!**

---

## How to Run

### Step 1: Activate your Python environment
```bash
cd /home/roops/ADK_Agent_Demo
source .venv/bin/activate  # Or your RLBench virtual environment
```

### Step 2: Run the test
```bash
python test_rlbench_standalone.py
```

### Step 3: Watch CoppeliaSim
A CoppeliaSim GUI window should open showing the Franka Panda robot performing a pick-and-place task.

---

## What Will Happen

1. **CoppeliaSim launches** - GUI window appears with 3D robot
2. **Task loads** - PickAndLift task with objects on table
3. **Initial capture** - RGB, depth, point cloud saved
4. **Robot moves** - 10 random actions executed (you see robot moving)
5. **Final capture** - Final state images saved
6. **Cleanup** - Environment shuts down

---

## Expected Output

```
================================================================================
RLBench Standalone Simulation Test
================================================================================
Testing: Franka Panda robot with RGB/Depth/PointCloud capture
================================================================================

✓ Output directory: /tmp/rlbench_test_output
✓ Task: PickAndLift
✓ Headless mode: False
✓ Number of steps: 10

[Step 1] Configuring observation sensors...
✓ Observation config:
  - Front camera: [256, 256] RGB + Depth + PointCloud
  - Additional cameras: Left/Right shoulder, Wrist
  - Robot state: Joints, Gripper

[Step 2] Configuring action mode...
✓ Action mode: JointVelocity + Discrete Gripper

[Step 3] Creating RLBench environment...
✓ Environment created

[Step 4] Launching CoppeliaSim...
>>> CoppeliaSim GUI window should appear now!
✓ CoppeliaSim launched successfully!

[Step 5] Loading task: PickAndLift...
✓ Task loaded: PickAndLift

[Step 6] Resetting task and capturing initial observation...
✓ Task reset
  Task description: pick_up_the_block

[Step 7] Inspecting observation data...
✓ Observation type: <class 'rlbench.backend.observation.Observation'>
  Available attributes:
    - front_rgb: (256, 256, 3) (dtype: float32)
    - front_depth: (256, 256) (dtype: float32)
    - front_point_cloud: (65536, 3) (dtype: float32)
    - joint_positions: (7,) (dtype: float32)
    - gripper_open: float64
    - gripper_pose: (7,) (dtype: float32)
    ...

[Step 8] Capturing RGB image from front camera...
✓ RGB image captured: (256, 256, 3) (range: [0.000, 1.000])
  Saved to: /tmp/rlbench_test_output/front_rgb.png

[Step 9] Capturing depth image from front camera...
✓ Depth image captured: (256, 256) (range: [0.500m, 2.500m])
  Saved raw depth to: /tmp/rlbench_test_output/front_depth_raw.png
  Saved colorized depth to: /tmp/rlbench_test_output/front_depth_colorized.png

[Step 10] Capturing point cloud from front camera...
✓ Point cloud captured: (65536, 3)
  Saved to: /tmp/rlbench_test_output/front_pointcloud.ply

[Step 11] Robot state information...
✓ Joint positions: [ 0.0  -0.5   0.0   -1.5   0.0   1.0   0.0]
✓ Gripper state: OPEN (1.000)
✓ Gripper pose: [ 0.3   0.0   0.8   0.0   0.0   0.0   1.0]

[Step 12] Executing 10 random actions (watch CoppeliaSim!)...
  Step 1/10: reward=0.000, terminate=False
  Step 2/10: reward=0.000, terminate=False
  ...
  Step 10/10: reward=0.000, terminate=False

[Step 13] Capturing final state...
✓ Final RGB saved to: /tmp/rlbench_test_output/front_rgb_final.png

[Step 14] Shutting down environment...
✓ Environment shutdown complete

================================================================================
TEST COMPLETE!
================================================================================

Summary:
  ✓ CoppeliaSim launched with Franka Panda robot
  ✓ Task: PickAndLift
  ✓ RGB images captured and saved
  ✓ Depth images captured and saved
  ✓ Point clouds captured and saved
  ✓ Robot executed 10 actions

Output files saved to: /tmp/rlbench_test_output

Files created:
  - front_depth_colorized.png (67.8 KB)
  - front_depth_raw.png (131.1 KB)
  - front_pointcloud.ply (2456.3 KB)
  - front_rgb.png (45.2 KB)
  - front_rgb_final.png (46.1 KB)

================================================================================
Next steps:
  1. Check the output files to verify data quality
  2. View point cloud: open3d /tmp/rlbench_test_output/front_pointcloud.ply
  3. If everything looks good, proceed to agent integration
================================================================================
```

---

## Output Files Location

All captured data is saved to: **`/tmp/rlbench_test_output/`**

### Files Created:
1. **`front_rgb.png`** - Initial RGB image (256x256)
2. **`front_depth_raw.png`** - Initial depth image (16-bit, millimeters)
3. **`front_depth_colorized.png`** - Depth visualization (color-coded)
4. **`front_pointcloud.ply`** - 3D point cloud (can open in MeshLab, CloudCompare, or Open3D)
5. **`front_rgb_final.png`** - Final RGB image after robot motion

---

## Viewing the Point Cloud

### Option 1: Open3D Viewer
```bash
python -c "import open3d as o3d; pcd = o3d.io.read_point_cloud('/tmp/rlbench_test_output/front_pointcloud.ply'); o3d.visualization.draw_geometries([pcd])"
```

### Option 2: MeshLab (if installed)
```bash
meshlab /tmp/rlbench_test_output/front_pointcloud.ply
```

### Option 3: CloudCompare (if installed)
```bash
cloudcompare /tmp/rlbench_test_output/front_pointcloud.ply
```

---

## Troubleshooting

### Issue 1: "Could not find COPPELIASIM_ROOT"
**Solution:** Set environment variable
```bash
export COPPELIASIM_ROOT=/home/roops/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

### Issue 2: CoppeliaSim window doesn't appear
**Solution:** Check if it's already running
```bash
pkill coppeliaSim  # Kill any existing instances
python test_rlbench_standalone.py  # Try again
```

### Issue 3: "No module named 'rlbench'"
**Solution:** Install RLBench in your environment
```bash
pip install git+https://github.com/stepjam/RLBench.git
```

### Issue 4: Point cloud is all zeros
**Solution:** This is normal for some tasks. Try different task:
Edit line 35 in `test_rlbench_standalone.py`:
```python
TASK_TO_USE = ReachTarget  # Change from PickAndLift
```

---

## Configuration Options

You can modify these variables at the top of `test_rlbench_standalone.py`:

```python
OUTPUT_DIR = Path("/tmp/rlbench_test_output")  # Where to save files
TASK_TO_USE = PickAndLift  # Task to run (PickAndLift, ReachTarget, etc.)
HEADLESS = False  # Set True to run without GUI
NUM_STEPS = 10  # Number of robot actions to execute
```

---

## Next Steps After Success

Once this test passes:
1. ✅ You've verified RLBench works
2. ✅ CoppeliaSim visualization works
3. ✅ Camera data (RGB, depth, point cloud) is correct
4. ✅ Robot can execute actions

**Then proceed to:**
- Integrate with your multi-agent framework
- Test the RLBench MCP server
- Run full pipeline with `adk web`

---

## Quick Commands Reference

```bash
# Run test
python test_rlbench_standalone.py

# Check output
ls -lh /tmp/rlbench_test_output/

# View RGB
eog /tmp/rlbench_test_output/front_rgb.png

# View depth
eog /tmp/rlbench_test_output/front_depth_colorized.png

# Clean output
rm -rf /tmp/rlbench_test_output/*
```

---

**Good luck! The CoppeliaSim window should show your Franka Panda robot in action! 🤖**
