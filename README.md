# RLBench Multi-Agent Framework

A multi-agent robotic manipulation system using Google's Agent Development Kit (ADK) and RLBench simulation environment.

## Overview

This framework implements a human-in-the-loop multi-agent system for robot manipulation tasks:
- **Planning Agent**: Generates execution plans for approval
- **Sensing Agent**: Captures camera observations and sensor data
- **Perception Agent**: Detects objects using GroundingDINO
- **Motion Agent**: Executes pick-and-place motions

## Features

- ✅ Multi-agent orchestration with human approval workflow
- ✅ Open-vocabulary object detection with GroundingDINO
- ✅ Depth-filtered 3D localization (1.7cm accuracy)
- ✅ RLBench integration (ReachTarget, PickAndLift tasks)
- ✅ Model Context Protocol (MCP) for tool abstraction
- ✅ Path planning with collision avoidance

## Project Structure

```
rlbench-multi-agent/
├── multi_tool_agent/
│   ├── agent.py                    # Multi-agent orchestrator
│   └── ros_mcp_server/
│       ├── rlbench_orchestration_server.py    # RLBench control tools
│       ├── perception_orchestration_server.py # Object detection tools
│       └── sensor_data/            # Runtime sensor data (gitignored)
├── docs/                           # Documentation
├── archive/                        # Old versions and experiments
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup

### Prerequisites
- Python 3.8+
- CoppeliaSim (for RLBench)
- CUDA-capable GPU (for GroundingDINO)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RoopsHub/rlbench-multi-agent.git
cd rlbench-multi-agent
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download GroundingDINO model weights:
```bash
# Create model directory
mkdir -p multi_tool_agent/ros_mcp_server/model

# Download weights (example - adjust URL as needed)
# Place groundingdino_swint_ogc.pth and GroundingDINO_SwinT_OGC.py in model/
```

5. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys (DeepSeek, etc.)
```

## Usage

### Running the Multi-Agent System

```bash
# Activate virtual environment
source .venv/bin/activate

# Start ADK web interface
python -m google.adk.ui.web --agent-module multi_tool_agent.agent --agent-name root_agent --port 8000
```

Then open http://localhost:8000 in your browser.

### Example Tasks

**ReachTarget:**
```
User: "Reach the red target"
Agent: [Generates plan] → Wait for approval → Execute
```

**PickAndLift:**
```
User: "Pick up the red cube"
Agent: [Generates plan] → Wait for approval → Execute
```

## Key Components

### Agent Architecture
- **DeepSeek Reasoner**: Planning and reasoning
- **DeepSeek Chat**: Fast execution agents
- **GroundingDINO**: Text-based object detection
- **RLBench**: Robot manipulation simulation

### Perception Pipeline
1. Capture RGB + Depth from RLBench camera
2. Detect object with GroundingDINO (text prompt)
3. Depth-filtered averaging for accurate 3D position
4. Transform to robot base frame

### Motion Control
- Path planning with collision avoidance
- Gripper state preservation during motion
- Task-specific sequences (reach, grasp, lift)

## Documentation

See `docs/` folder for detailed documentation:
- `FRAMEWORK_REPORT.md` - Complete system architecture
- `PERCEPTION_ACCURACY_IMPROVEMENTS.md` - Perception improvements (32% accuracy boost)
- `ORCHESTRATION_PLAN.md` - Multi-agent design
- `PICKANDLIFT_INTEGRATION_GUIDE.md` - Pick-and-lift implementation

## Troubleshooting

**Colors not detected correctly:**
- Ensure proper lighting in RLBench scene
- Adjust GroundingDINO thresholds in perception_orchestration_server.py

**Path planning failures:**
- Check target position is within robot workspace
- Enable collision_checking parameter

**Model not loading:**
- Verify GroundingDINO model files are in `multi_tool_agent/ros_mcp_server/model/`

## Contributing

Contributions welcome! Please read the documentation in `docs/` before submitting PRs.

## License

[Add your license here]

## Acknowledgments

- Google Agent Development Kit (ADK)
- RLBench robotic manipulation benchmark
- GroundingDINO for open-vocabulary detection
- DeepSeek for LLM models
