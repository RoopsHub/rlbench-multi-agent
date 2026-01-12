# RLBench Multi-Agent Framework

A human-aligned multi-agent framework for language-guided robotic manipulation using Google's Agent Development Kit and RLBench simulation.

## Overview

This framework implements human-in-the-loop multi-agent orchestration for manipulation tasks:
- **Root Orchestrator**: Generates plans with adjustable parameters, awaits approval
- **Sensing**: Captures RGB, depth, and point cloud observations
- **Perception**: Open-vocabulary object detection via GroundingDINO
- **Motion**: Cartesian position control with gripper sequences

## Features

- ✅ Multi-agent orchestration with human approval workflow
- ✅ Open-vocabulary object detection (no training required)
- ✅ Five completed RLBench tasks (ReachTarget, PickAndLift, PushButton, PutRubbishInBin, StackBlocks)
- ✅ Model Context Protocol (MCP) for platform abstraction
- ✅ Text-prompted 3D object localization
- ✅ Parameter adjustment during planning phase

## Project Structure

```
rlbench-multi-agent/
├── multi_tool_agent/
│   ├── agent.py                               # Multi-agent orchestrator
│   └── ros_mcp_server/
│       ├── rlbench_orchestration_server.py    # RLBench MCP tools
│       ├── perception_orchestration_server.py # GroundingDINO detection
│       └── sensor_data/                       # Runtime data (gitignored)
├── docs/                                      # Framework documentation
├── requirements.txt                           # Python dependencies
└── README.md                                  # This file
```

## Setup

### Prerequisites
- Python 3.8+
- CoppeliaSim v4.1.0 (for RLBench)
- RLBench: https://github.com/stepjam/RLBench
- CUDA-capable GPU (recommended for GroundingDINO)

### Installation

1. Install RLBench:
```bash
# Follow installation instructions from:
# https://github.com/stepjam/RLBench
```

2. Clone this repository:
```bash
git clone https://github.com/RoopsHub/rlbench-multi-agent.git
cd rlbench-multi-agent
```

3. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Download GroundingDINO weights:
```bash
mkdir -p multi_tool_agent/ros_mcp_server/model
# Place groundingdino_swint_ogc.pth in model/ directory
```

6. Configure environment variables:
```bash
export COPPELIASIM_ROOT=/path/to/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
```

## Usage

### Running the System

```bash
source .venv/bin/activate
python -m google.adk.ui.web --agent-module multi_tool_agent.agent --agent-name root_agent --port 8000
```

Navigate to http://localhost:8000 in browser.

### Example Tasks

**ReachTarget:**
```
User: "Reach the red target"
Agent: [Plan with parameters] → Approval → Execute
```

**PickAndLift:**
```
User: "Pick up the red cube and lift it"
Agent: [Plan with parameters] → Approval → Execute
```

## Key Components

### Agent Architecture
- **DeepSeek / OpenAI**: Planning and execution reasoning
- **GroundingDINO**: Text-prompted open-vocabulary detection
- **RLBench**: Benchmark manipulation environment (100+ tasks)
- **MCP**: Standardized tool protocol for platform abstraction

### Perception Pipeline
1. Capture RGB + Depth from RLBench cameras
2. Detect objects with text prompts (e.g., "red cube . red sphere")
3. Extract 3D positions from depth and point cloud
4. Transform coordinates to robot base frame

### Motion Control
- Cartesian position control via inverse kinematics
- Binary gripper states (open/close)
- Task-specific motion sequences with approach heights
- Sequential execution with error propagation

## Documentation

Detailed guides in `docs/` folder:
- `FRAMEWORK_REPORT.md` - System architecture and completed tasks
- `RLBENCH_TASK_UNDERSTANDING.md` - Task implementation guide for replication

## Troubleshooting

**Detection confidence low:**
- Adjust GroundingDINO thresholds in perception_orchestration_server.py
- Improve detection prompts with color and shape descriptors

**Position outside workspace:**
- Verify camera-to-base coordinate transformation
- Use ground truth comparison for debugging

**Model not loading:**
- Confirm GroundingDINO weights in `multi_tool_agent/ros_mcp_server/model/`
- Check CUDA availability for GPU inference

## License

[Add your license here]

## Acknowledgments

- [Google Agent Development Kit (ADK)](https://github.com/google/adk)
- [RLBench robotic manipulation benchmark](https://github.com/stepjam/RLBench)
- [GroundingDINO open-vocabulary object detection](https://github.com/IDEA-Research/GroundingDINO)
- DeepSeek and OpenAI language models
