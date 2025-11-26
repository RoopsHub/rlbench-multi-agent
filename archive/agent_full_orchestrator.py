"""
Multi-Agent Robot Manipulation Framework with Dynamic Planning

This implementation features:
1. Planning Agent - Dynamically generates execution plans
2. Human-in-the-loop Approval - Via adk web chat interface
3. Dynamic SequentialAgent Construction - Built from approved plan
4. Execution Pipeline - Sensing → Perception → Motion

Workflow:
User Task → Planning Agent → Display Plan → User Approval → Execute → Results
"""

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_session_manager import StdioServerParameters
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# === MODEL CONFIGURATION ===
# ==============================================================================

# Original Ollama models (commented out - had issues with premature tool execution and hallucination)
# planning_model = LiteLlm(model="ollama_chat/llama3.2:latest")
# perception_model = LiteLlm(model="ollama_chat/llama3.2:latest", format="json")
# action_model = LiteLlm(model="ollama_chat/llama3.2:latest")

# DeepSeek R1 - Reasoning model for complex multi-turn workflows
# Uses explicit reasoning capabilities for better instruction following
planning_model = LiteLlm(model="deepseek/deepseek-reasoner")
sensing_model = LiteLlm(model="deepseek/deepseek-reasoner")
perception_model = LiteLlm(model="deepseek/deepseek-reasoner", format="json")
action_model = LiteLlm(model="deepseek/deepseek-reasoner")

# ==============================================================================
# === MCP TOOLSETS CONFIGURATION ===
# ==============================================================================

current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# RLBench Server Configuration
# Switch between ROS and RLBench by commenting/uncommenting:
# PATH_TO_ROS_SERVER = current_file_dir / "ros_mcp_server" / "server.py"  # Original ROS
PATH_TO_ROS_SERVER = current_file_dir / "ros_mcp_server" / "rlbench_server.py"  # RLBench
PATH_TO_PERCEPTION_SERVER = current_file_dir / "ros_mcp_server" / "perception.py"

# CoppeliaSim environment variables for RLBench
# These need to be passed to MCP server subprocess
COPPELIASIM_ROOT = os.environ.get('COPPELIASIM_ROOT', '/home/roops/CoppeliaSim')
RLBENCH_ENV = {
    'COPPELIASIM_ROOT': COPPELIASIM_ROOT,
    'LD_LIBRARY_PATH': COPPELIASIM_ROOT,
    'QT_QPA_PLATFORM': 'xcb',
}

ros_mcp_tools = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PATH_TO_ROS_SERVER)],
            env=RLBENCH_ENV,  # Pass CoppeliaSim environment
        ),
        timeout=120,
    ),
    tool_filter=["get_topics", "subscribe_and_download_image",
                 "subscribe_and_download_pointcloud", "subscribe_and_download_depth_image"]
)

perception_tools = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PATH_TO_PERCEPTION_SERVER)]
        ),
        timeout=120,
    ),
)

motion_tools = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PATH_TO_ROS_SERVER)],
            env=RLBENCH_ENV,  # Pass CoppeliaSim environment
        ),
        timeout=120,
    ),
    tool_filter=["get_joint_states", "send_arm_trajectory", "send_gripper_command", "compute_ik_solution"]
)

# ==============================================================================
# === EXECUTION AGENTS (Sensing, Perception, Motion) ===
# ==============================================================================

sensing_agent = Agent(
    name="SensingAgent",
    model=sensing_model,
    description="Captures sensor data (RGB images and point clouds) from the robot",
    instruction=(
        "You are a robot sensing agent. Your task is to capture the necessary sensor data.\n\n"

        "**Your Responsibilities:**\n"
        "1. Call `subscribe_and_download_image` to capture an RGB image\n"
        "2. Call `subscribe_and_download_pointcloud` to capture point cloud data\n"
        "3. Return the file paths in JSON format\n\n"

        "**Output Format:**\n"
        "You MUST output ONLY a valid JSON object with this exact structure:\n"
        "{\n"
        '  "user_task": "the original task description",\n'
        '  "rgb_path": "/path/to/image.png",\n'
        '  "ply_path": "/path/to/pointcloud.ply",\n'
        '  "frame_preference": "base"\n'
        "}\n\n"

        "Do not add any explanations or text outside the JSON object."
    ),
    tools=[ros_mcp_tools],
    output_key="sensor_data_json"
)

perception_agent = Agent(
    name="PerceptionAgent",
    model=perception_model,
    description="Analyzes sensor data to detect objects and plan grasps",
    instruction=(
        "You are a robot perception agent. Your task is to detect objects and plan grasps.\n\n"

        "**Input Data:**\n"
        "You will receive sensor data from the previous agent: {sensor_data_json}\n\n"

        "**Your Responsibilities:**\n"
        "1. Parse the input JSON to extract:\n"
        "   - user_task (map to 'task_text' parameter)\n"
        "   - rgb_path (map to 'rgb_path' parameter)\n"
        "   - ply_path (map to 'ply_path' parameter)\n"
        "   - frame_preference (map to 'frame_preference' parameter)\n"
        "2. Call `plan_pick_and_place_tool` with these mapped parameters\n"
        "3. Return the tool's output exactly as received\n\n"

        "**Critical Rules:**\n"
        "- Output ONLY the raw JSON returned by the tool\n"
        "- Do NOT add explanations, summaries, or markdown formatting\n"
        "- Do NOT modify the tool's output in any way\n"
    ),
    tools=[perception_tools],
    output_key="perception_output_json"
)

motion_agent = Agent(
    name="MotionAgent",
    model=action_model,
    description="Executes robot motion commands for pick-and-place tasks",
    instruction=(
        "You are a robot motion execution agent. Your task is to execute pick-and-place motions safely.\n\n"

        "**Input Data:**\n"
        "You will receive perception results: {perception_output_json}\n\n"

        "**Your Responsibilities:**\n"
        "1. Parse the perception output to extract grasp pose and place pose\n"
        "2. Execute the following sequence:\n"
        "   a. Call `get_joint_states` to check current robot state\n"
        "   b. Call `compute_ik_solution` to convert grasp pose (position + orientation_xyzw) to joint angles\n"
        "   c. Call `send_arm_trajectory` with computed joint angles to move to pick position\n"
        "   d. Call `send_gripper_command` with position=0.0 to close gripper and grasp\n"
        "   e. Call `compute_ik_solution` to convert place pose to joint angles\n"
        "   f. Call `send_arm_trajectory` with computed joint angles to move to place position\n"
        "   g. Call `send_gripper_command` with position=0.055 to open gripper and release\n"
        "3. Report the execution status and any issues encountered\n\n"

        "**Safety Guidelines:**\n"
        "- Stop immediately if any command fails\n"
        "- Log each action and its result\n"
        "- Report clear error messages if something goes wrong\n"
    ),
    tools=[motion_tools]
)
#
## ==============================================================================
## === PLANNING AGENT WITH HARDCODED CAPABILITIES ===
## ==============================================================================
#
#planning_agent = Agent(
#    name="PlanningAgent",
#    model=planning_model,
#    description="Generates dynamic execution plans for robot manipulation tasks",
#    instruction=(
#        "You are an autonomous robot task planning agent. Your role is to generate detailed, "
#        "executable plans for robotic manipulation tasks.\n\n"
#
#        "**AVAILABLE AGENTS AND THEIR CAPABILITIES:**\n\n"
#
#        "1. **SensingAgent**\n"
#        "   - Role: Captures sensor data from the robot\n"
#        "   - Capabilities: RGB imaging, Point cloud capture, Depth sensing\n"
#        "   - Available Tools:\n"
#        "     • subscribe_and_download_image (captures RGB image)\n"
#        "     • subscribe_and_download_pointcloud (captures 3D point cloud)\n"
#        "     • subscribe_and_download_depth_image (captures depth image)\n"
#        "   - Output: sensor_data_json containing {rgb_path, ply_path, frame_preference}\n\n"
#
#        "2. **PerceptionAgent**\n"
#        "   - Role: Analyzes sensor data to detect objects and plan grasps\n"
#        "   - Capabilities: Object detection, Segmentation, Grasp planning, Place planning\n"
#        "   - Available Tools:\n"
#        "     • plan_pick_and_place_tool (detects objects, generates grasp and place poses)\n"
#        "   - Input: sensor_data_json from SensingAgent\n"
#        "   - Output: perception_output_json containing {grasp_candidates, place_pose}\n\n"
#
#        "3. **MotionAgent**\n"
#        "   - Role: Executes robot motion commands\n"
#        "   - Capabilities: Inverse kinematics, Arm trajectory execution, Gripper control, Joint state monitoring\n"
#        "   - Available Tools:\n"
#        "     • get_joint_states (reads current robot joint positions)\n"
#        "     • compute_ik_solution (converts Cartesian pose to joint angles)\n"
#        "     • send_arm_trajectory (moves robot arm using joint angles)\n"
#        "     • send_gripper_command (opens/closes gripper)\n"
#        "   - Input: perception_output_json from PerceptionAgent\n"
#        "   - Output: execution_status\n\n"
#
#        "**YOUR TASK:**\n"
#        "Analyze the user's request and generate a detailed execution plan.\n\n"
#
#        "**REASONING PROCESS:**\n"
#        "1. **Understand the Task:**\n"
#        "   - What is the goal? (e.g., pick and place, sorting, inspection)\n"
#        "   - What objects are involved?\n"
#        "   - What information is needed? (color → RGB, position → point cloud)\n\n"
#
#        "2. **Select Required Agents:**\n"
#        "   - Which agents are needed to accomplish this task?\n"
#        "   - In what order should they execute?\n"
#        "   - What data flows between them?\n\n"
#
#        "3. **Map Tools to Actions:**\n"
#        "   - Which specific MCP tools does each agent need?\n"
#        "   - Are all required tools available?\n\n"
#
#        "4. **Assess Risks:**\n"
#        "   - What could go wrong?\n"
#        "   - How can failures be mitigated?\n\n"
#
#        "**OUTPUT FORMAT:**\n"
#        "You MUST output a valid JSON object with this exact structure:\n\n"
#        "```json\n"
#        "{\n"
#        '  "task_description": "Brief summary of the task",\n'
#        '  "plan_type": "sequential",\n'
#        '  "estimated_duration_seconds": 45,\n'
#        '  "confidence": 0.85,\n'
#        '  "execution_steps": [\n'
#        "    {\n"
#        '      "step_id": 1,\n'
#        '      "agent": "SensingAgent",\n'
#        '      "action": "capture_scene_data",\n'
#        '      "required_tools": ["subscribe_and_download_image", "subscribe_and_download_pointcloud"],\n'
#        '      "inputs_from_step": [],\n'
#        '      "rationale": "Need RGB for color detection and point cloud for 3D position",\n'
#        '      "expected_output": "sensor_data_json",\n'
#        '      "success_criteria": "Both image and point cloud captured successfully"\n'
#        "    },\n"
#        "    {\n"
#        '      "step_id": 2,\n'
#        '      "agent": "PerceptionAgent",\n'
#        '      "action": "detect_and_plan_grasp",\n'
#        '      "required_tools": ["plan_pick_and_place_tool"],\n'
#        '      "inputs_from_step": [1],\n'
#        '      "rationale": "Detect target object and generate grasp poses",\n'
#        '      "expected_output": "perception_output_json",\n'
#        '      "success_criteria": "Object detected with confidence > 0.6"\n'
#        "    },\n"
#        "    {\n"
#        '      "step_id": 3,\n'
#        '      "agent": "MotionAgent",\n'
#        '      "action": "execute_pick_and_place",\n'
#        '      "required_tools": ["get_joint_states", "send_arm_trajectory", "send_gripper_command"],\n'
#        '      "inputs_from_step": [2],\n'
#        '      "rationale": "Execute motion to pick and place object",\n'
#        '      "expected_output": "execution_status",\n'
#        '      "success_criteria": "All motion commands executed without errors"\n'
#        "    }\n"
#        "  ],\n"
#        '  "risk_assessment": {\n'
#        '    "potential_failures": ["object_not_found", "poor_grasp", "collision"],\n'
#        '    "mitigation_strategies": ["multiple_viewpoints", "grasp_alternatives", "collision_checking"]\n'
#        "  }\n"§
#        "}\n"
#        "```\n\n"
#
#        "**IMPORTANT:**\n"
#        "- Output ONLY the JSON object, no additional text\n"
#        "- Ensure all JSON is valid and properly formatted\n"
#        "- Always include all three agents for pick-and-place tasks\n"
#        "- Be specific about which tools each agent needs\n"
#    ),
#    tools=[],
#    output_key="plan_json"
#)
#
# ==============================================================================
# === UTILITY FUNCTIONS ===
# ==============================================================================

def format_plan_for_display(plan_json: str) -> str:
    """
    Convert plan JSON to human-readable formatted text for user approval.

    Args:
        plan_json: JSON string containing the execution plan

    Returns:
        Formatted string for display
    """
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError:
        return "Error: Invalid plan JSON"

    output = []
    output.append("\n" + "="*70)
    output.append("EXECUTION PLAN")
    output.append("="*70)
    output.append(f"\n**Task:** {plan.get('task_description', 'N/A')}")
    output.append(f"**Type:** {plan.get('plan_type', 'N/A')}")
    output.append(f"**Estimated Duration:** {plan.get('estimated_duration_seconds', 'N/A')} seconds")
    output.append(f"**Confidence:** {plan.get('confidence', 'N/A')}")

    output.append("\n**Execution Steps:**")
    for step in plan.get('execution_steps', []):
        output.append(f"\n{step.get('step_id')}. [{step.get('agent')}] {step.get('action')}")
        output.append(f"   → Tools: {', '.join(step.get('required_tools', []))}")
        output.append(f"   → Rationale: {step.get('rationale', 'N/A')}")
        output.append(f"   → Success Criteria: {step.get('success_criteria', 'N/A')}")

    risks = plan.get('risk_assessment', {})
    if risks:
        output.append("\n**Risk Assessment:**")
        output.append(f"   Potential Failures: {', '.join(risks.get('potential_failures', []))}")
        output.append(f"   Mitigation Strategies: {', '.join(risks.get('mitigation_strategies', []))}")

    output.append("\n" + "="*70)
    output.append("\n**Please respond with:**")
    output.append("- 'approved' or 'looks good' to execute the plan")
    output.append("- Specific feedback to revise the plan (e.g., 'change destination to blue tray')")
    output.append("="*70 + "\n")

    return "\n".join(output)

def is_plan_approved(user_response: str) -> bool:
    """
    Check if user response indicates approval.

    Args:
        user_response: The user's text response

    Returns:
        True if approved, False otherwise
    """
    approval_keywords = [
        "approved", "approve", "looks good", "good to go",
        "proceed", "execute", "yes", "ok", "okay", "go ahead",
        "perfect"
    ]

    response_lower = user_response.lower().strip()
    return any(keyword in response_lower for keyword in approval_keywords)

def build_sequential_agent_from_plan(plan_json: str) -> SequentialAgent:
    """
    Dynamically construct a SequentialAgent from the approved plan.

    Args:
        plan_json: JSON string containing the execution plan

    Returns:
        SequentialAgent configured according to the plan
    """
    plan = json.loads(plan_json)

    # Map agent names to agent instances
    agent_map = {
        "SensingAgent": sensing_agent,
        "PerceptionAgent": perception_agent,
        "MotionAgent": motion_agent
    }

    # Extract agents in order from execution steps
    sub_agents = []
    for step in plan.get('execution_steps', []):
        agent_name = step.get('agent')
        if agent_name in agent_map:
            agent_instance = agent_map[agent_name]
            # Only add if not already in the list (avoid duplicates)
            if agent_instance not in sub_agents:
                sub_agents.append(agent_instance)

    # Create the SequentialAgent
    pipeline = SequentialAgent(
        name="DynamicExecutionPipeline",
        sub_agents=sub_agents,
        description=f"Executing plan for: {plan.get('task_description', 'robot manipulation task')}"
    )

    return pipeline

# ==============================================================================
# === ROOT AGENT FOR ADK WEB ===
# ==============================================================================

# Create a stateful orchestrator that handles the full workflow
root_agent = Agent(
    name="RobotManipulationOrchestrator",
    model=planning_model,
    description=(
        "Orchestrates robot manipulation tasks with planning, approval, and execution.\n"
        "Manages multi-turn conversations for human-in-the-loop control."
    ),
    instruction=(
        "You are an orchestrator for a robot manipulation system.\n\n"

        "**WORKFLOW:**\n\n"

        "**PHASE 1 - PLANNING (when user gives initial task):**\n"
        "1. Understand the user's task\n"
        "2. Generate a text-based execution plan (DO NOT CALL ANY TOOLS)\n"
        "3. Display the plan in this format:\n\n"
        "==================================\n\n"
        "📋 EXECUTION PLAN\n"
        "===============================================\n\n"
        "**Task:** [describe task]\n"
        "**Type:** sequential\n"
        "**Estimated Duration:** 45 seconds\n"
        "**Confidence:** 0.85\n\n"
        "**Execution Steps:**\n\n"
        "1. [SensingAgent] Capture scene data\n"
        "   → Tools: subscribe_and_download_image, subscribe_and_download_pointcloud\n"
        "   → Rationale: Need RGB for color detection and point cloud for 3D position\n"
        "   → Success Criteria: Both image and point cloud captured successfully\n\n"
        "2. [PerceptionAgent] Detect and plan grasp\n"
        "   → Tools: plan_pick_and_place_tool\n"
        "   → Rationale: Detect target object and generate grasp poses\n"
        "   → Success Criteria: Object detected with confidence > 0.6\n\n"
        "3. [MotionAgent] Execute pick and place\n"
        "   → Tools: get_joint_states, compute_ik_solution, send_arm_trajectory, send_gripper_command\n"
        "   → Rationale: Execute motion to pick and place object using inverse kinematics\n"
        "   → Success Criteria: All motion commands executed without errors\n\n"
        "**Risk Assessment:**\n"
        "    Potential Failures: object_not_found, poor_grasp, collision\n"
        "    Mitigation Strategies: multiple_viewpoints, grasp_alternatives, collision_checking\n\n"
        "==========================================\n\n"
        "**Please respond with 'approved' to execute or provide feedback for changes.**\n\n"

        "4. STOP and WAIT for user response\n\n"

        "**PHASE 2 - EXECUTION (after user approves):**\n"
        "When user says 'approved', 'yes', 'looks good', 'ok', 'proceed', or 'go ahead':\n\n"

        "1. Say: '✓ Plan approved. Starting execution...'\n\n"

        "2. **STEP 1 - Capture sensor data:**\n"
        "   - Call subscribe_and_download_image with topic='/camera_head/color/image_raw'\n"
        "   - WAIT for the response to get the image file path\n"
        "   - Call subscribe_and_download_pointcloud with topic='/camera_head/depth/points'\n"
        "   - WAIT for the response to get the point cloud file path\n"
        "   - Remember both file paths for the next step\n\n"

        "3. **STEP 2 - Plan grasp (ONLY after step 1 completes):**\n"
        "   - Extract the rgb_path from step 1's image result\n"
        "   - Extract the ply_path from step 1's point cloud result\n"
        "   - Call plan_pick_and_place_tool with:\n"
        "     * task_text: [the original user task]\n"
        "     * rgb_path: [actual path from step 1, NOT empty string]\n"
        "     * ply_path: [actual path from step 1, NOT empty string]\n"
        "     * frame_preference: 'base'\n"
        "   - WAIT for the response to get grasp and place poses\n\n"

        "4. **STEP 3 - Execute motion (ONLY after step 2 completes):**\n"
        "   - Parse the grasp_candidates and place_pose from step 2\n"
        "   - Call get_joint_states to check current robot state\n"
        "   - Call compute_ik_solution with the first grasp pose (position + orientation_xyzw) to get joint angles\n"
        "   - Call send_arm_trajectory with the computed joint angles to move to pick position\n"
        "   - Call send_gripper_command with position=0.0 to close gripper and grasp\n"
        "   - Call compute_ik_solution with the place pose to get joint angles\n"
        "   - Call send_arm_trajectory with the computed joint angles to move to place position\n"
        "   - Call send_gripper_command with position=0.055 to open gripper and release\n\n"

        "5. Report completion\n\n"

        "**IF USER PROVIDES FEEDBACK (not approval):**\n"
        "- Acknowledge the feedback\n"
        "- Generate a revised plan\n"
        "- Display it in the same format\n"
        "- Ask for approval again\n"
        "- DO NOT execute tools\n\n"

        "**CRITICAL RULES:**\n"
        "1. NEVER call tools during planning phase\n"
        "2. ONLY call tools after user approval\n"
        "3. Execute steps SEQUENTIALLY - wait for each step to complete before starting the next\n"
        "4. NEVER pass empty strings or None to plan_pick_and_place_tool - always use actual file paths from step 1\n"
        "5. DO NOT call multiple steps in parallel - they must run in sequence\n"
    ),
    tools=[ros_mcp_tools, perception_tools, motion_tools]
)

# ==============================================================================
# === ENTRY POINT FOR ADK WEB ===
# ==============================================================================

# The root_agent will be automatically detected and used by 'adk web'
# To run: adk web
