# Multi-Agent Robot Manipulation Framework

## Overview

A multi-agent system using Google's Agent Development Kit (ADK) and RLBench simulation environment. The framework implements a a human-aligned multi-agent framework for language-guided robotic manipulation

---

## Architecture

### High-Level Design

The framework implements a **hierarchical multi-agent system** with human oversight:

```
┌─────────────────────────────────────────────────────────────┐
│              Root Orchestrator Agent                        │
│  (Planning, Approval Loop, Execution Coordination)          │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─────► Planning Phase (Generate plan, wait for approval)
             │
             └─────► Execution Phase (Sequential pipeline)
                     │
                     ├─► Sensing (Capture RGB + Depth + Point Cloud)
                     │
                     ├─► Perception (Detect objects via GroundingDINO)
                     │
                     └─► Motion (Execute Cartesian control + gripper)
```

### Key Principles

1. **Modularity:** Each component has a single, well-defined responsibility
2. **Tool Abstraction:** MCP (Model Context Protocol) provides platform-independent tool interface
3. **Human Oversight:** Explicit approval required before execution
4. **Open Vocabulary:** Text-based object detection (no retraining needed)

---

## Components

### 1. Root Orchestrator

**Role:** Top-level coordinator managing the full task lifecycle

**Workflow:**
- **Phase 1 - Planning:**
  - Parse user request (e.g., "pick up the red cube and lift it")
  - Generate detailed execution plan with motion sequences
  - Display plan with risk level and adjustable parameters
  - Wait for human approval

- **Phase 2 - Execution:**
  - Sensing: Load task and capture camera observations
  - Perception: Detect objects using open-vocabulary detection
  - Motion: Execute motion sequences with gripper control

**Model:** DeepSeek (via LiteLLM) AND OpenAI o5mini

### 2. Perception System

**Technology:** GroundingDINO (open-vocabulary object detector)

**Capabilities:**
- Text-prompted object detection (e.g., "red cube", "blue sphere")
- Multi-object detection (e.g., "red cube . red sphere")
- 3D position extraction from depth and point cloud data
- HSV color verification for improved accuracy

**Key Feature:** No training required - detects objects from text descriptions

### 3. Motion Control

**Action Mode:** MoveArmThenGripper with EndEffectorPoseViaIK

**Control Strategy:**
- Cartesian position control (X, Y, Z coordinates)
- Binary gripper states (open/closed)
- Task-specific motion sequences
- Collision checking disabled for performance

### 4. MCP Integration

**RLBench MCP Server:** Bridges RLBench API to standardized MCP tools

**Key Tools:**
- `load_task(task_name)` - Initialize RLBench task
- `get_camera_observation()` - Capture RGB, depth, and camera data
- `get_target_position()` - Get ground truth (reference only)
- `detect_object_3d(prompt, ...)` - Open-vocabulary 3D detection
- `move_to_position(x, y, z)` - Cartesian motion control
- `control_gripper(state)` - Gripper control (open/close)

---

## Completed Tasks

The framework successfully implements the following RLBench tasks:

### 1. ReachTarget
- **Description:** Move end-effector to target sphere
- **Detection:** Single object ("red sphere")
- **Gripper:** Remains open throughout
- **Risk Level:** LOW

### 2. PickAndLift
- **Description:** Pick up cube and lift to sphere position
- **Detection:** Multi-object ("red cube . red sphere")
- **Gripper:** Open → Close → Closed
- **Risk Level:** MEDIUM

### 3. PushButton
- **Description:** Push button with end-effector
- **Detection:** Single object (button)
- **Gripper:** Remains open throughout
- **Risk Level:** LOW

### 4. PutRubbishInBin
- **Description:** Pick up trash and place in bin
- **Detection:** Multi-object ("trash . bin")
- **Gripper:** Open → Close → Open
- **Risk Level:** MEDIUM

### 5. StackBlocks
- **Description:** Stack 2 red cubes on green cube (stacking zone)
- **Detection:** Multi-object ("red cube") - detects all cubes
- **Gripper:** Multiple pick-place cycles
- **Risk Level:** HIGH (8 objects, precision stacking)

---

## Workflow

### Complete Execution Flow

```
User Input: "Pick up the red cube and lift it"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: PLANNING                                           │
├─────────────────────────────────────────────────────────────┤
│ Task Analysis:                                              │
│  - Task Type: PickAndLift                                   │
│  - Target Objects: red cube, red sphere                     │
│  - Gripper Strategy: Open → Close → Closed                  │
│  - Risk Level: MEDIUM                                       │
│                                                             │
│ Motion Plan with Justifications:                            │
│  1. control_gripper("open")                                 │
│     WHY: Prepare gripper for grasping                       │
│  2. move_to_position(cube_x, cube_y, cube_z + 0.15)        │
│     WHY: Approach height above cube                         │
│  3. move_to_position(cube_x, cube_y, cube_z + 0.015)       │
│     WHY: Grasp height with clearance                        │
│  ... [full sequence shown]                                  │
│                                                             │
│ Adjustable Parameters:                                      │
│  - Approach Height: 0.15m [0.10-0.20m]                     │
│  - Grasp Offset: 0.015m [0.01-0.03m]                       │
│                                                             │
│ ⏳ AWAITING APPROVAL                                        │
└─────────────────────────────────────────────────────────────┘
    │
    │ User: "approved"
    ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: EXECUTION                                          │
├─────────────────────────────────────────────────────────────┤
│ Sensing:                                                    │
│  1. load_task("PickAndLift") → success                     │
│  2. get_camera_observation() → paths captured              │
│  3. get_target_position() → ground truth (reference)       │
│                                                             │
│ Perception:                                                 │
│  4. detect_object_3d("red cube . red sphere", ...)         │
│     → cube: [0.064, 0.227, 0.773] (conf: 0.85)            │
│     → sphere: [0.207, 0.210, 0.996] (conf: 0.91)          │
│                                                             │
│ Motion:                                                     │
│  5. control_gripper("open") → success                      │
│  6. move_to_position(0.064, 0.227, 0.923) → success       │
│  7. move_to_position(0.064, 0.227, 0.788) → success       │
│  8. control_gripper("close") → success                     │
│  9. move_to_position(0.207, 0.210, 0.996) → success       │
│                                                             │
│ ✅ Result: SUCCESS                                          │
└─────────────────────────────────────────────────────────────┘
```

### Key Workflow Characteristics

**Sequential Execution:**
- Strict ordering: Sensing → Perception → Motion
- Each phase waits for previous phase completion
- No parallel execution

**Data Flow:**
- All communication via structured tool calls
- Detection results include 3D positions and confidence scores
- Ground truth available for validation (not used in execution)

**Error Handling:**
- Tool failures propagate to orchestrator
- Execution stops on first failure
- Clear error messages provided to user

**Human-in-the-Loop:**
- Explicit approval required before execution
- User can adjust parameters before approval
- Plan modification triggers re-planning

---

## Technical Details

### Environment Configuration

**RLBench Setup:**
- Action Mode: MoveArmThenGripper with EndEffectorPoseViaIK
- Observation Config: RGB, depth, point cloud, low-dim state
- Camera Resolution: 512x512
- Collision Checking: Disabled

**Environment Variables:**
```python
COPPELIASIM_ROOT=/path/to/CoppeliaSim
LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
COPPELIASIM_DISABLE_FFMPEG=1
```

### Detection Strategy

**Per-Task Detection Prompts:**
- ReachTarget: Single object from task description
- PickAndLift: "red cube . red sphere"
- PushButton: Single object (button)
- PutRubbishInBin: "trash . bin"
- StackBlocks: "red cube" (all cubes including green stacking zone)

**Detection Pipeline:**
1. Parse task description for object keywords
2. Extract color and shape information
3. Run GroundingDINO with text prompt
4. Extract 3D positions from depth/point cloud
5. Return detected objects array with positions and confidence

### Adjustable Parameters by Task

**PutRubbishInBin:**
- Approach Height: 0.15m [0.10-0.20m]
- Grasp Offset: 0.015m [0.01-0.03m]
- Bin Drop Height: 0.10m [0.05-0.15m]

**PickAndLift:**
- Approach Height: 0.15m [0.10-0.20m]
- Grasp Offset: 0.015m [0.01-0.03m]

**PushButton:**
- Approach Height: 0.10m [0.05-0.15m]
- Push Depth: 0.002m [0.001-0.005m]

**StackBlocks:**
- Approach Height: 0.15m [0.10-0.20m]
- Grasp Offset: 0.015m [0.01-0.03m]
- Stack Offset: 0.055m [0.05-0.07m]
- Stack Zone XY: [0.0, 0.3] (green cube position)

---

## Key Features

### 1. Open-Vocabulary Detection
- No training required for new objects
- Flexible text-based prompts
- Color and shape extraction from natural language

### 2. Human-in-the-Loop Control
- Safety through explicit approval
- Interactive parameter adjustment
- Plan quality assessment before execution

### 3. Platform Abstraction
- MCP provides standardized tool interface
- Framework independent of backend implementation
- Easy to extend with new tasks

### 4. Enhanced UI
- Emoji-based formatting for clarity
- Risk level classification (LOW/MEDIUM/HIGH)
- Detailed motion justifications
- Parameter adjustment guidance

---

## Dependencies

**Core Framework:**
- google-adk==1.5.0
- litellm==1.74.0.post1
- mcp==1.21.2

**Vision & Perception:**
- groundingdino-py==0.4.0
- opencv-python-headless==4.12.0.88
- supervision==0.6.0

**RLBench & Robotics:**
- rlbench==1.2.0
- PyRep==4.1.0.3
- ikpy==3.4.2

**Deep Learning:**
- torch==2.9.1
- torchvision==0.24.1
- transformers==4.57.1

**External Requirements:**
- CoppeliaSim v4.1.0
- CUDA-capable GPU (recommended for GroundingDINO)

---

## Current Status

**Working:**
- ✅ Multi-agent orchestration with human approval
- ✅ Open-vocabulary object detection
- ✅ 5 completed RLBench tasks
- ✅ MCP tool integration
- ✅ Sequential execution pipeline
- ✅ Parameter adjustment workflow

**Architecture:**
- Platform: RLBench + CoppeliaSim
- Perception: GroundingDINO
- Control: Cartesian position control via IK
- LLM: DeepSeek (via LiteLLM)

---

## File Structure

```
rlbench-multi-agent/
├── multi_tool_agent/
│   ├── agent.py                               # Main orchestrator
│   └── ros_mcp_server/
│       ├── rlbench_orchestration_server.py    # RLBench MCP server
│       └── perception_orchestration_server.py # GroundingDINO server
├── docs/
│   ├── FRAMEWORK_REPORT.md                    # This file
│   └── RLBENCH_TASK_UNDERSTANDING.md          # Task specifications
├── requirements.txt                            # Python dependencies
└── README.md                                   # Setup and usage guide
```

---

## References


**Tools:**
- RLBench: https://github.com/stepjam/RLBench
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- Google ADK: https://github.com/google/adk
- IKPy: https://github.com/Phylliade/ikpy
