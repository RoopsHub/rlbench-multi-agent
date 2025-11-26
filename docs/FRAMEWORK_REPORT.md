# Multi-Agent Robot Manipulation Framework - Comprehensive Report

## Executive Summary

This document provides a complete technical overview of the **Multi-Agent Robot Manipulation Framework** built using Google's ADK (Agent Development Kit). The framework implements a modular, human-in-the-loop approach to robotic manipulation tasks, leveraging Large Language Models (LLMs) to orchestrate complex sensor-perception-action pipelines.

**Primary Goal:** Evaluate and benchmark multi-agent LLM collaboration for robotic manipulation tasks, NOT to build a production robot system from scratch.

---

## Table of Contents

1. [Project Context: From Gazebo to RLBench](#1-project-context-from-gazebo-to-rlbench)
2. [Framework Architecture](#2-framework-architecture)
3. [Agent Hierarchy and Roles](#3-agent-hierarchy-and-roles)
4. [MCP Tool Integration](#4-mcp-tool-integration)
5. [Workflow and Execution Pipeline](#5-workflow-and-execution-pipeline)
6. [Technical Implementation Details](#6-technical-implementation-details)
7. [Key Design Decisions](#7-key-design-decisions)
8. [Next Steps: RLBench Migration](#8-next-steps-rlbench-migration)

---

## 1. Project Context: From Gazebo to RLBench

### 1.1 The Original Problem

The project initially used a **custom Gazebo/ROS2 simulation environment**, but encountered critical blocking issues:

**Perception Transform Error:**
- Camera reported objects at Z = 0.95m instead of actual Z = 0.01m
- This 94cm offset made all manipulation tasks impossible
- Attempted fixes: TF tree transforms, Kabsch calibration, empirical corrections
- **Result:** 2 weeks of debugging with no reliable solution

**Inverse Kinematics (IK) Failures:**
- Even with correct URDF, robot positions were systematically wrong
- IK solver produced unreachable joint configurations
- Manual calibration attempts (`calibrate_transform_kabsch.py`, `fixed_ik_solver.py`, `quick_calibrate.py`) failed to resolve core issues

**Key Insight:** The problem wasn't the multi-agent framework - it was the simulation infrastructure built from scratch.

### 1.2 The Paradigm Shift

**Critical Decision:** Don't build a robot simulator - use a proven testbed.

**Why This Matters:**
- Main goal: **Evaluate multi-agent framework** (agent collaboration, task planning, tool orchestration)
- Secondary goal: NOT to debug low-level robotics (transforms, IK, physics)
- Solution: Leverage existing, validated simulation platforms

### 1.3 RLBench: The Discovery

**What is RLBench?**
- Benchmark suite with 100+ manipulation tasks
- Built on CoppeliaSim (formerly V-REP) physics engine
- Provides ground truth for vision, poses, and motion
- Used by state-of-the-art research (e.g., MALMM paper - arXiv:2411.17636)

**Why RLBench is Perfect for This Project:**

| Requirement | RLBench Solution |
|-------------|------------------|
| RGB + Depth images | ✅ Native support, exportable as .png |
| Point clouds | ✅ Native support, exportable as .ply |
| Proven IK/motion | ✅ Works out-of-box, no calibration needed |
| Benchmark tasks | ✅ 100 tasks for agent evaluation |
| Multi-agent research | ✅ MALMM paper already validated this |
| Framework compatibility | ✅ Python API wrappable in MCP tools |

**Migration Path:**
1. Install CoppeliaSim v4.1.0 (RLBench's physics engine) - ~25 minutes
2. Create RLBench MCP adapter (~200 lines) to bridge RLBench → existing framework
3. **Zero changes to agent code** (framework is platform-agnostic via MCP abstraction)

**Bottom Line:** After 2 weeks fighting Gazebo, RLBench provides a 1-day migration path to a proven, publication-ready testbed.

---

## 2. Framework Architecture

### 2.1 High-Level Design

The framework implements a **hierarchical multi-agent system** with three layers:

```
┌─────────────────────────────────────────────────────────────┐
│              Root Orchestrator Agent                        │
│  (Planning, Approval Loop, Execution Coordination)          │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─────► Planning Phase (Generate plan, wait for approval)
             │
             └─────► Execution Phase (Sequential agent pipeline)
                     │
                     ├─► Sensing Agent (Capture RGB + Point Cloud)
                     │
                     ├─► Perception Agent (Detect objects, plan grasps)
                     │
                     └─► Motion Agent (Execute IK + trajectory + gripper)
```

### 2.2 Design Philosophy

**Key Principles:**
1. **Modularity:** Each agent has a single, well-defined responsibility
2. **Tool Abstraction:** Agents interact with hardware/simulation via MCP tools (not direct API calls)
3. **Platform Independence:** MCP abstraction layer enables switching backends (ROS → RLBench) without agent code changes
4. **Human Oversight:** Explicit approval loop before execution for safety and control
5. **Reasoning-Driven:** Uses DeepSeek-R1 (reasoning model) for better multi-turn instruction following

---

## 3. Agent Hierarchy and Roles

### 3.1 Root Orchestrator Agent

**File:** `agent.py:408-501`

**Role:** Top-level coordinator that manages the full task lifecycle.

**Responsibilities:**
1. **Phase 1 - Planning:**
   - Receive user task (e.g., "pick the red cube and place it in the blue tray")
   - Generate detailed execution plan in structured format
   - Display plan with: steps, tools, rationale, success criteria, risks
   - Wait for human approval

2. **Phase 2 - Execution:**
   - Upon approval, orchestrate sequential tool calls
   - Execute Sensing → Perception → Motion pipeline
   - Handle errors and report status

**Model:** DeepSeek-R1 Reasoner (selected for explicit reasoning capabilities)

**Critical Design Feature:**
- **No tools during planning** - prevents premature execution
- **Sequential execution** - strict ordering to prevent data dependencies breaking
- **Stateful conversation** - maintains context across planning/execution phases

**Example Interaction:**
```
User: "Pick the green cube and place it on the tray"

Orchestrator (Planning Phase):
┌──────────────────────────────────────────────────────┐
│ 📋 EXECUTION PLAN                                    │
├──────────────────────────────────────────────────────┤
│ Task: Pick green cube and place on tray              │
│ Type: sequential                                     │
│ Estimated Duration: 45 seconds                       │
│ Confidence: 0.85                                     │
│                                                      │
│ Execution Steps:                                     │
│ 1. [SensingAgent] Capture scene data                 │
│    → Tools: subscribe_and_download_image,            │
│              subscribe_and_download_pointcloud       │
│    → Rationale: Need RGB for color detection         │
│    → Success: Both files captured                    │
│                                                      │
│ 2. [PerceptionAgent] Detect and plan grasp           │
│    → Tools: plan_pick_and_place_tool                 │
│    → Rationale: Detect green cube, compute 6-DOF pose│
│    → Success: Object detected with confidence > 0.6  │
│                                                      │
│ 3. [MotionAgent] Execute pick and place              │
│    → Tools: get_joint_states, compute_ik_solution,   │
│              send_arm_trajectory, send_gripper_cmd   │
│    → Rationale: Execute motion using inverse kinem.  │
│    → Success: All motion commands succeed            │
└──────────────────────────────────────────────────────┘

Orchestrator: Please respond with 'approved' to execute

User: approved

Orchestrator (Execution Phase):
✓ Plan approved. Starting execution...
[Calls sensing tools → waits for result]
[Calls perception tools → waits for result]
[Calls motion tools → waits for result]
✓ Task completed successfully!
```

---

### 3.2 Sensing Agent

**File:** `agent.py:91-116`

**Role:** Robot sensor data acquisition layer.

**Input:** User task description (implicit)

**Responsibilities:**
1. Call `subscribe_and_download_image` to capture RGB image from ROS topic
2. Call `subscribe_and_download_pointcloud` to capture 3D point cloud
3. Return file paths as JSON

**Output Format:**
```json
{
  "user_task": "pick the red cube",
  "rgb_path": "/path/to/image_1234567890.png",
  "ply_path": "/path/to/pointcloud_1234567890.ply",
  "frame_preference": "base"
}
```

**Tools:**
- `subscribe_and_download_image` (from `ros_mcp_server/server.py`)
- `subscribe_and_download_pointcloud` (from `ros_mcp_server/server.py`)

**Model:** DeepSeek-R1 Reasoner

**Design Notes:**
- Outputs structured JSON for reliable parsing by downstream agents
- Does NOT process sensor data - just captures and returns paths
- Frame preference indicates coordinate system (base_link vs camera_head_link)

---

### 3.3 Perception Agent

**File:** `agent.py:118-144`

**Role:** Object detection, segmentation, and grasp planning.

**Input:** Sensor data JSON from Sensing Agent

**Responsibilities:**
1. Parse input JSON to extract `user_task`, `rgb_path`, `ply_path`, `frame_preference`
2. Call `plan_pick_and_place_tool` with these parameters
3. Return tool output exactly as received (no modification)

**Output Format (from perception tool):**
```json
{
  "grasp_candidates": [
    {
      "object_class": "red cube",
      "confidence": 0.92,
      "position": [0.45, 0.12, 0.03],
      "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
      "grasp_quality": 0.85,
      "approach_vector": [0.0, 0.0, -1.0]
    }
  ],
  "place_pose": {
    "position": [0.60, -0.15, 0.05],
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]
  }
}
```

**Tools:**
- `plan_pick_and_place_tool` (from `ros_mcp_server/perception.py`)

**Model:** DeepSeek-R1 Reasoner (with `format="json"` for structured output)

**Key Technology:**
- **GroundingDINO:** Open-vocabulary object detector (text-prompted)
  - Can detect "red cube", "blue cube", "tray" without retraining
  - Uses vision-language model for flexible object recognition
- **Point Cloud Segmentation:** Extracts 3D geometry using voxel downsampling + RANSAC plane removal
- **6-DOF Pose Estimation:** Computes orientation from principal component analysis (PCA)

**Critical Rules:**
- Output ONLY raw JSON from tool (no explanations, no markdown)
- Do NOT modify perception results
- Passes results unchanged to Motion Agent

---

### 3.4 Motion Agent

**File:** `agent.py:146-174`

**Role:** Robot motion execution and control.

**Input:** Perception results JSON (grasp candidates + place pose)

**Responsibilities:**
1. Parse perception output
2. Execute sequential motion pipeline:
   - a. `get_joint_states` → Check current robot configuration
   - b. `compute_ik_solution(grasp_pose)` → Convert Cartesian pose to joint angles
   - c. `send_arm_trajectory(joint_angles)` → Move to pick position
   - d. `send_gripper_command(0.0)` → Close gripper (grasp)
   - e. `compute_ik_solution(place_pose)` → Convert place pose to joint angles
   - f. `send_arm_trajectory(joint_angles)` → Move to place position
   - g. `send_gripper_command(0.055)` → Open gripper (release)
3. Report execution status

**Tools:**
- `get_joint_states` (from `ros_mcp_server/server.py`)
- `compute_ik_solution` (from `ros_mcp_server/server.py`)
- `send_arm_trajectory` (from `ros_mcp_server/server.py`)
- `send_gripper_command` (from `ros_mcp_server/server.py`)

**Model:** DeepSeek-R1 Reasoner

**Safety Guidelines:**
- Stop immediately if any command fails
- Log each action and result
- Report clear error messages

**Technical Details:**
- **IK Solver:** IKPy library with custom URDF-based kinematic chain
- **Trajectory Format:** Joint positions for 6-DOF arm
- **Gripper Control:** Parallel-jaw gripper (0.0m = closed, 0.055m = open)

---

## 4. MCP Tool Integration

### 4.1 What is MCP?

**Model Context Protocol (MCP):** A standardized protocol for connecting LLMs to external tools and data sources.

**Benefits:**
- **Abstraction:** Agents don't need to know if they're talking to ROS, RLBench, or a real robot
- **Modularity:** Swap backends by changing MCP server, not agent code
- **Standardization:** Tools exposed via JSON-RPC interface

### 4.2 MCP Architecture in This Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    ADK Agents                               │
│         (Sensing, Perception, Motion)                       │
└────────────┬────────────────────────────────────────────────┘
             │
             │ (Tool calls via MCP protocol)
             │
┌────────────▼────────────────────────────────────────────────┐
│              MCP Toolsets                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ ros_mcp_tools│  │perception_   │  │motion_tools  │     │
│  │              │  │tools         │  │              │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────┐
│           MCP Servers (Python processes)                    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  server.py (ROS Bridge + IK)                       │    │
│  │  - WebSocket to rosbridge (ws://localhost:9090)    │    │
│  │  - Image/PointCloud capture                        │    │
│  │  - IKPy solver for inverse kinematics              │    │
│  │  - Trajectory execution via ROS topics             │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │  perception.py (GroundingDINO + Segmentation)      │    │
│  │  - Object detection (text-prompted)                │    │
│  │  - 3D point cloud segmentation                     │    │
│  │  - 6-DOF pose estimation                           │    │
│  │  - Grasp planning                                  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────┬──────────────────┬──────────────────┬─────────────┘
          │                  │                  │
          │                  │                  │
┌─────────▼──────────────────▼──────────────────▼─────────────┐
│              Backend (Currently ROS2/Gazebo)                │
│              (Future: RLBench/CoppeliaSim)                  │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 MCP Toolset Configurations

**File:** `agent.py:49-85`

#### ROS MCP Toolset (Sensing)
```python
ros_mcp_tools = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PATH_TO_ROS_SERVER)],  # → server.py
        ),
        timeout=120,
    ),
    tool_filter=[
        "get_topics",
        "subscribe_and_download_image",
        "subscribe_and_download_pointcloud",
        "subscribe_and_download_depth_image"
    ]
)
```

#### Perception Toolset
```python
perception_tools = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PATH_TO_PERCEPTION_SERVER)]  # → perception.py
        ),
        timeout=120,
    ),
    # No filter - expose all perception tools
)
```

#### Motion Toolset
```python
motion_tools = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PATH_TO_ROS_SERVER)]  # → server.py (shared with sensing)
        ),
        timeout=120,
    ),
    tool_filter=[
        "get_joint_states",
        "send_arm_trajectory",
        "send_gripper_command",
        "compute_ik_solution"
    ]
)
```

### 4.4 Key MCP Tools

#### Sensing Tools (`server.py`)

**`subscribe_and_download_image`**
- Subscribes to ROS topic (e.g., `/camera_head/color/image_raw`)
- Receives compressed image via ROSBridge WebSocket
- Decodes base64 → PNG
- Saves to disk with timestamp
- Returns file path

**`subscribe_and_download_pointcloud`**
- Subscribes to ROS PointCloud2 topic (e.g., `/camera_head/depth/points`)
- Receives binary point cloud data
- Converts to Open3D format
- Saves as .ply file
- Returns file path

#### Perception Tools (`perception.py`)

**`plan_pick_and_place_tool`**
```python
plan_pick_and_place_tool(
    task_text: str,           # e.g., "pick the red cube"
    rgb_path: str,            # Path to RGB image
    ply_path: str,            # Path to point cloud
    frame_preference: str     # "base" or "camera"
) -> dict
```

**Workflow:**
1. Load RGB image and point cloud
2. Extract text prompt from task (e.g., "red cube")
3. Run GroundingDINO object detection on RGB
4. For each detected bounding box:
   - Extract 3D points from point cloud
   - Remove ground plane (RANSAC)
   - Compute centroid (position)
   - Compute PCA orientation (quaternion)
5. Transform poses from camera frame → base frame (using calibrated transform matrix)
6. Detect tray (using "tray" text prompt)
7. Return grasp candidates + place pose

**Key Configuration:**
- Camera intrinsics: fx=223.41, fy=223.41, cx=212.0, cy=120.0
- Detection thresholds: box=0.35, text=0.25
- Transform matrix: 4x4 homogeneous (calibrated via Kabsch algorithm)

#### Motion Tools (`server.py`)

**`get_joint_states`**
- Subscribes to `/joint_states` topic
- Returns current joint positions (6-DOF arm)

**`compute_ik_solution`**
```python
compute_ik_solution(
    position: list[float],          # [x, y, z]
    orientation_xyzw: list[float],  # Quaternion [x, y, z, w]
    frame: str = "base_link"
) -> list[float]                    # Joint angles
```
- Uses IKPy library with 6-DOF arm chain
- Converts Cartesian pose → joint angles
- Returns None if unreachable

**`send_arm_trajectory`**
```python
send_arm_trajectory(
    joint_positions: list[float]  # 6 joint angles
) -> bool
```
- Publishes to `/arm_controller/joint_trajectory` topic
- Uses quintic polynomial interpolation (5s duration)
- Returns success status

**`send_gripper_command`**
```python
send_gripper_command(
    position: float  # 0.0 = closed, 0.055 = open
) -> bool
```
- Publishes to `/gripper_controller/gripper_cmd` topic
- Controls parallel-jaw gripper

---

## 5. Workflow and Execution Pipeline

### 5.1 Complete Task Flow

```
User Input: "Pick the red cube and place it in the blue tray"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: PLANNING                                           │
├─────────────────────────────────────────────────────────────┤
│ Root Orchestrator Agent                                     │
│  1. Analyze task requirements                               │
│  2. Generate structured plan:                               │
│     - Sensing: Capture RGB + PointCloud                     │
│     - Perception: Detect "red cube" + "blue tray"           │
│     - Motion: Execute pick-and-place                        │
│  3. Display plan to user                                    │
│  4. WAIT for approval                                       │
└─────────────────────────────────────────────────────────────┘
    │
    │ User: "approved"
    ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: EXECUTION                                          │
├─────────────────────────────────────────────────────────────┤
│ Step 1: SENSING                                             │
│  Tool Call: subscribe_and_download_image                    │
│    → Result: /tmp/image_1699876543.png                      │
│  Tool Call: subscribe_and_download_pointcloud               │
│    → Result: /tmp/pointcloud_1699876543.ply                 │
│  Output: sensor_data_json                                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: PERCEPTION                                          │
│  Tool Call: plan_pick_and_place_tool(                       │
│    task_text="pick the red cube and place in blue tray",   │
│    rgb_path="/tmp/image_1699876543.png",                    │
│    ply_path="/tmp/pointcloud_1699876543.ply",               │
│    frame_preference="base"                                  │
│  )                                                          │
│  Internal Processing:                                       │
│    - GroundingDINO detects "red cube" @ bbox [120,80,240,200]│
│    - Extract 3D points from point cloud                     │
│    - Compute 6-DOF pose: pos=[0.45, 0.12, 0.03]             │
│    - Detect "blue tray" → place_pose=[0.60, -0.15, 0.05]    │
│  Output: perception_output_json                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 3: MOTION                                              │
│  Tool Call: get_joint_states                                │
│    → Current joints: [0.0, -0.5, 1.2, 0.0, 1.0, 0.0]        │
│  Tool Call: compute_ik_solution(                            │
│    position=[0.45, 0.12, 0.03],                             │
│    orientation_xyzw=[0, 0, 0, 1]                            │
│  )                                                          │
│    → IK joints: [0.2, -0.8, 1.5, 0.1, 0.9, 0.0]             │
│  Tool Call: send_arm_trajectory([0.2, -0.8, 1.5, ...])      │
│    → SUCCESS: Moved to pick position                        │
│  Tool Call: send_gripper_command(position=0.0)              │
│    → SUCCESS: Gripper closed                                │
│  Tool Call: compute_ik_solution(                            │
│    position=[0.60, -0.15, 0.05],                            │
│    orientation_xyzw=[0, 0, 0, 1]                            │
│  )                                                          │
│    → IK joints: [0.5, -0.6, 1.3, 0.2, 1.1, 0.0]             │
│  Tool Call: send_arm_trajectory([0.5, -0.6, 1.3, ...])      │
│    → SUCCESS: Moved to place position                       │
│  Tool Call: send_gripper_command(position=0.055)            │
│    → SUCCESS: Gripper opened                                │
│  Output: "Task completed successfully"                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
User sees: "✓ Task completed successfully!"
```

### 5.2 Key Workflow Characteristics

**Sequential Execution:**
- Agents execute in strict order: Sensing → Perception → Motion
- Each agent WAITS for previous agent to complete
- No parallel execution (prevents data dependency issues)

**Data Flow:**
- Sensing → Perception: File paths via JSON
- Perception → Motion: Grasp poses via JSON
- All data passed as structured JSON (no free text)

**Error Handling:**
- Tool failures propagate back to orchestrator
- Motion agent stops immediately on failure
- User sees clear error messages

**Human-in-the-Loop:**
- Explicit approval required before execution
- User can modify plan (triggers re-planning)
- Provides safety and control for research experiments

---

## 6. Technical Implementation Details

### 6.1 Model Selection

**Original Models (Commented Out):**
```python
# Ollama models (llama3.2:latest)
# Issues: Premature tool execution, hallucination
```

**Current Model: DeepSeek-R1 Reasoner**
```python
planning_model = LiteLlm(model="deepseek/deepseek-reasoner")
sensing_model = LiteLlm(model="deepseek/deepseek-reasoner")
perception_model = LiteLlm(model="deepseek/deepseek-reasoner", format="json")
action_model = LiteLlm(model="deepseek/deepseek-reasoner")
```

**Why DeepSeek-R1?**
- **Explicit reasoning capabilities:** Better for multi-turn workflows
- **Instruction following:** Doesn't execute tools prematurely
- **JSON format support:** Structured output for perception agent
- **Complex planning:** Handles conditional logic better than Llama 3.2

### 6.2 Coordinate Frame Management

**The Transform Problem:**
- Camera sees world in camera_head_link frame
- Robot operates in base_link frame
- Need transformation: T_base_from_camera (4x4 matrix)

**Calibration Method:**
- Used Kabsch algorithm (SVD-based rigid transform estimation)
- Ground truth correspondences from manual measurements
- Transform matrix stored in `perception.py:79-84`

**Current Transform:**
```python
T_base_from_camera = [
    0.78620,  0.00662,  0.61794, -0.64671,
    0.00017,  0.99994, -0.01093, -0.00082,
   -0.61798,  0.00870,  0.78615,  1.28648,
    0.00000,  0.00000,  0.00000,  1.00000,
]
```

**Status:** Still has 94cm Z-offset error (reason for RLBench migration)

### 6.3 Inverse Kinematics

**Implementation:** IKPy library with custom kinematic chain

**File:** `server.py:compute_ik_solution()`

**Joint Chain:**
```python
EXPECTED_JOINT_NAMES = [
    "link1_to_link2",
    "link2_to_link3",
    "link3_to_link4",
    "link4_to_link5",
    "link5_to_link6",
    "link6_to_link6_flange"
]
```

**Algorithm:**
1. Convert quaternion → rotation matrix
2. Build 4x4 homogeneous transform
3. Call IKPy solver with target pose
4. Return joint angles (or None if unreachable)

**Limitations (in Gazebo setup):**
- IK solutions sometimes unreachable due to transform errors
- No collision checking
- No joint limit validation

### 6.4 GroundingDINO Integration

**What is GroundingDINO?**
- Open-vocabulary object detector
- Uses CLIP-like vision-language model
- Can detect objects from text descriptions (no retraining needed)

**Configuration:**
```python
model_config = "/home/6GSoft/models/GroundingDINO_SwinT_OGC.py"
model_checkpoint = "/home/6GSoft/models/groundingdino_swint_ogc.pth"
box_threshold = 0.35  # Confidence threshold
text_threshold = 0.25  # Text similarity threshold
device = "cpu"        # Runs on CPU (no GPU required)
```

**Detection Pipeline:**
1. Parse task text to extract object names (e.g., "red cube")
2. Run GroundingDINO with text prompt
3. Get bounding boxes with confidence scores
4. Map 2D boxes → 3D point cloud regions
5. Extract object geometry

**Advantages:**
- No training data required
- Flexible task descriptions
- Works with novel objects

---

## 7. Key Design Decisions

### 7.1 Why Multi-Agent (Not Monolithic)?

**Modularity:**
- Each agent can be tested independently
- Easy to swap models (e.g., different LLM for perception)
- Clear separation of concerns

**Debugging:**
- Can inspect intermediate outputs (sensor_data_json, perception_output_json)
- Easier to isolate failures

**Benchmarking:**
- Can compare different LLMs per agent role
- Example: GPT-4 for planning vs Claude for perception

**Scalability:**
- Can add agents (e.g., SafetyAgent for collision checking)
- Can parallelize independent agents in future

### 7.2 Why MCP Abstraction?

**Platform Independence:**
- Current: ROS2/Gazebo
- Future: RLBench/CoppeliaSim
- Possible: Real robot (UR5, Franka Panda)
- **Agent code stays the same**

**Standardization:**
- Tools have consistent interface (JSON input/output)
- LLMs learn tool schemas, not platform-specific APIs

**Tool Composition:**
- Can combine tools from different sources
- Example: ROS for motion + GroundingDINO for vision

### 7.3 Why Human-in-the-Loop?

**Safety:**
- Research setup - don't want autonomous execution during debugging
- User can catch errors before robot moves

**Evaluation:**
- User can assess plan quality before execution
- Useful for measuring planning capabilities

**Flexibility:**
- User can modify plans (e.g., "change destination to green tray")
- Enables interactive experimentation

### 7.4 Why DeepSeek-R1 Over Llama 3.2?

**Original Issue:**
```python
# Ollama models had:
# - Premature tool execution (called tools during planning)
# - Hallucination (invented tool parameters)
```

**DeepSeek-R1 Benefits:**
- **Chain-of-thought reasoning:** Explicit reasoning tokens improve multi-step planning
- **Instruction adherence:** Better at following "DO NOT call tools during planning"
- **JSON formatting:** Native support for structured output
- **Context retention:** Better at maintaining state across multi-turn conversations

---

## 8. Next Steps: RLBench Migration

### 8.1 Migration Plan

**Goal:** Replace Gazebo/ROS2 backend with RLBench, keeping agent code unchanged.

**Steps:**

**1. Install RLBench Stack (~25 minutes)**
```bash
# Install CoppeliaSim v4.1.0
wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
export COPPELIASIM_ROOT=/path/to/CoppeliaSim

# Install PyRep
pip install git+https://github.com/stepjam/PyRep.git

# Install RLBench
pip install git+https://github.com/stepjam/RLBench.git
```

**2. Create RLBench MCP Server (~200 lines)**

Create `ros_mcp_server/rlbench_server.py`:
```python
from mcp.server.fastmcp import FastMCP
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
import numpy as np
from pathlib import Path

mcp = FastMCP("rlbench-mcp-server")

# Initialize RLBench
obs_config = ObservationConfig()
obs_config.set_all(True)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete()
    ),
    obs_config=obs_config,
    headless=False
)

@mcp.tool()
def subscribe_and_download_image(topic: str) -> dict:
    """Capture RGB image from RLBench camera"""
    obs = env.get_observation()
    # RLBench provides: obs.front_rgb (numpy array)

    # Save as PNG
    timestamp = int(time.time() * 1000)
    path = f"/tmp/rlbench_rgb_{timestamp}.png"
    cv2.imwrite(path, cv2.cvtColor(obs.front_rgb, cv2.COLOR_RGB2BGR))

    return {"file_path": path}

@mcp.tool()
def subscribe_and_download_pointcloud(topic: str) -> dict:
    """Capture point cloud from RLBench depth"""
    obs = env.get_observation()
    # RLBench provides: obs.front_depth (numpy array)
    # + camera intrinsics

    # Convert depth → point cloud
    pcd = depth_to_pointcloud(obs.front_depth, obs.front_camera_intrinsics)

    # Save as PLY
    timestamp = int(time.time() * 1000)
    path = f"/tmp/rlbench_pcd_{timestamp}.ply"
    o3d.io.write_point_cloud(path, pcd)

    return {"file_path": path}

@mcp.tool()
def send_arm_trajectory(joint_positions: list[float]) -> dict:
    """Execute arm motion in RLBench"""
    action = np.zeros(8)  # 7 joints + gripper
    action[:6] = joint_positions[:6]
    action[-1] = 1.0  # Keep gripper state

    obs, reward, terminate = env.step(action)
    return {"success": True}

@mcp.tool()
def send_gripper_command(position: float) -> dict:
    """Control gripper (0.0 = closed, 0.055 = open)"""
    action = np.zeros(8)
    action[-1] = 1.0 if position > 0.025 else 0.0  # Binary gripper

    obs, reward, terminate = env.step(action)
    return {"success": True}

# ... (implement compute_ik_solution, get_joint_states)
```

**3. Update Agent Configuration (~5 minutes)**

In `agent.py`, change:
```python
PATH_TO_ROS_SERVER = current_file_dir / "ros_mcp_server" / "rlbench_server.py"
```

**4. Test with Existing Agents (~2-3 hours)**
```bash
adk web  # Launch ADK web interface
# User: "Pick the red cube"
# Verify: Sensing → Perception → Motion pipeline works
```

### 8.2 Expected Benefits

**Immediate:**
- No transform debugging (RLBench provides ground truth)
- Reliable IK (CoppeliaSim's built-in solver)
- Faster iteration (no ROS/Gazebo startup time)

**Long-term:**
- 100 benchmark tasks for evaluation
- Direct comparison to MALMM paper results
- Publication-ready experiments

### 8.3 Zero Agent Code Changes

**Why this works:**
```
Agents call:
  subscribe_and_download_image()  ← Interface stays the same
  plan_pick_and_place_tool()      ← Interface stays the same
  compute_ik_solution()           ← Interface stays the same

MCP servers implement these differently:
  ROS backend: WebSocket → rosbridge → Gazebo
  RLBench backend: Python API → CoppeliaSim
```

**The abstraction holds because:**
- Tools have standard schemas (input/output JSON)
- Agents only see tool results, not implementation
- MCP protocol is backend-agnostic

---

## 9. Conclusion

### 9.1 Framework Summary

This multi-agent framework demonstrates:

1. **Hierarchical Agent Design:**
   - Root orchestrator for planning + approval
   - Specialized agents for sensing, perception, motion

2. **Tool Abstraction via MCP:**
   - Platform-independent agent code
   - Easy backend swapping (ROS → RLBench)

3. **Human-in-the-Loop Control:**
   - Explicit approval loop for safety
   - Interactive plan refinement

4. **State-of-the-Art Components:**
   - DeepSeek-R1 for reasoning
   - GroundingDINO for open-vocabulary detection
   - IKPy for inverse kinematics

### 9.2 Current Status

**Working:**
- ✅ Multi-agent orchestration
- ✅ Planning phase with approval loop
- ✅ MCP tool integration
- ✅ GroundingDINO object detection
- ✅ Sequential execution pipeline

**Blocked (Gazebo-specific issues):**
- ❌ Coordinate transform (94cm Z-offset)
- ❌ IK solver reliability
- ❌ Perception accuracy

### 9.3 Path Forward

**Short-term (1-2 weeks):**
1. Migrate to RLBench/CoppeliaSim
2. Validate framework on benchmark tasks
3. Resolve transform/IK issues via proven platform

**Medium-term (1-2 months):**
1. Benchmark against MALMM paper
2. Evaluate different LLM combinations
3. Measure planning quality vs execution success rate

**Long-term (3-6 months):**
1. Publish framework evaluation results
2. Extend to more complex tasks (stacking, sorting)
3. Deploy to real robot (UR5/Franka Panda)

### 9.4 Key Takeaway

**The framework architecture is sound** - the issues are in the simulation backend, not the agent design. By migrating to RLBench, we can:
- **Unblock evaluation** in 1 day vs weeks of debugging
- **Leverage proven infrastructure** (100 benchmark tasks)
- **Compare to state-of-the-art** (MALMM paper)
- **Keep agent code unchanged** (MCP abstraction works!)

---

## Appendix A: File Structure

```
ADK_Agent_Demo/
├── multi_tool_agent/
│   ├── agent.py                          # Main framework implementation
│   ├── ros_mcp_server/
│   │   ├── server.py                     # ROS MCP server (sensing + motion)
│   │   ├── perception.py                 # GroundingDINO perception server
│   │   ├── fixed_ik_solver.py            # IK solver (IKPy-based)
│   │   ├── calibrate_transform_kabsch.py # Transform calibration
│   │   ├── quick_calibrate.py            # Quick calibration utility
│   │   └── test_*.py                     # Various test scripts
│   ├── agent_v1_backup.py                # Backup (old version)
│   └── agent_v2_orchestrator_backup.py   # Backup (orchestrator version)
├── .venv/                                # Python virtual environment
└── debug_out/                            # Debug output files
```

---

## Appendix B: Dependencies

**Python Packages:**
```
google-adk          # Agent Development Kit
google.generativeai # Gemini API (optional)
litellm             # Multi-LLM interface
mcp                 # Model Context Protocol
groundingdino-py    # Object detection
open3d              # Point cloud processing
opencv-python       # Image processing
scipy               # Scientific computing
ikpy                # Inverse kinematics
websockets          # ROS Bridge connection
numpy               # Numerical computing
torch               # Deep learning (for GroundingDINO)
```

**External Systems (Current):**
- ROS2 Humble
- Gazebo Classic
- rosbridge_server

**External Systems (Future):**
- CoppeliaSim v4.1.0
- PyRep
- RLBench

---

## Appendix C: References

**Papers:**
- MALMM: Multi-Agent Large Language Models for Zero-Shot Robotics Manipulation
  - arXiv:2411.17636
  - https://arxiv.org/abs/2411.17636
  - Uses 3-agent framework (Planner, Coder, Supervisor) with RLBench

**Tools:**
- RLBench: https://github.com/stepjam/RLBench
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- Google ADK: https://github.com/google/adk
- IKPy: https://github.com/Phylliade/ikpy

---

**Report Generated:** 2025-11-19
**Framework Version:** v2 (Orchestrator-based)
**Status:** Active development, migrating to RLBench
