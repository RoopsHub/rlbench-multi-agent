"""
AgentV2.py - Simplified Agentic Architecture with Human-in-the-Loop

This version implements:
1. Planning Agent with autonomous reasoning
2. Human approval gate for plan verification (with revision loop)
3. Adaptive sensing agent
4. Perception agent with error handling
5. Motion agent with monitored execution
6. VLM-based Validation Agent for post-execution verification
7. Support for both Ollama (local) and API-based models (OpenAI, Anthropic, etc.)

Simplified Workflow:
User Task → Planning Agent → Human Approval → Sensing → Perception →
Motion Execution → VLM Validation (verifies task completion)
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
from typing import Dict, Optional, List
from datetime import datetime

# ==============================================================================
# === MODEL CONFIGURATION - Choose Your LLM ===
# ==============================================================================

"""
CONFIGURATION GUIDE:

For OLLAMA (Local Models):
---------------------------
model = LiteLlm(model="ollama_chat/llama3.2:latest")

Available Ollama models (install with `ollama pull <model>`):
- ollama_chat/llama3.2:latest       (Fast, good for sensing/motion)
- ollama_chat/llama3.1:70b          (Better reasoning)
- ollama_chat/qwen2.5:14b           (Good balance)
- ollama_chat/deepseek-r1:latest    (Reasoning model)
- ollama_chat/llava:latest          (Vision model for validation)


For API-BASED MODELS (OpenAI, Anthropic, etc.):
------------------------------------------------
Requires API key in environment variable or direct configuration

1. OpenAI (GPT-4, GPT-4o, o1, o3-mini):
   model = LiteLlm(
       model="openai/gpt-4o",  # or "openai/o1-preview" for reasoning
       api_key=os.getenv("OPENAI_API_KEY")  # Set via: export OPENAI_API_KEY="sk-..."
   )

2. Anthropic (Claude):
   model = LiteLlm(
       model="anthropic/claude-3-5-sonnet-20241022",  # Latest Claude with vision
       api_key=os.getenv("ANTHROPIC_API_KEY")  # Set via: export ANTHROPIC_API_KEY="sk-ant-..."
   )

3. Google (Gemini):
   model = LiteLlm(
       model="gemini/gemini-2.0-flash-thinking-exp",  # Reasoning model
       api_key=os.getenv("GEMINI_API_KEY")
   )

4. Groq (Fast API inference):
   model = LiteLlm(
       model="groq/llama-3.3-70b-versatile",
       api_key=os.getenv("GROQ_API_KEY")
   )


RECOMMENDED CONFIGURATION:
--------------------------
- Planning Agent: Use reasoning models (o1, Claude, Gemini Thinking, DeepSeek-R1)
- Perception Agent: Use structured output models (GPT-4o with JSON mode)
- Sensing/Motion: Use fast local models (Llama 3.2, Qwen 2.5)
- Validation Agent: Use vision models (GPT-4o-vision, Claude 3.5 Sonnet, LLaVA)
"""

# --- PLANNING AGENT MODEL (Use best reasoning model) ---
# Using DeepSeek Reasoner for planning
planning_model = LiteLlm(model="deepseek/deepseek-reasoner")

# Backup Option 1: Ollama local reasoning model
# planning_model = LiteLlm(model="ollama_chat/llama3.2:latest")

# Backup Option 2: OpenAI o1/o3 reasoning model (UNCOMMENT TO USE)
# planning_model = LiteLlm(
#     model="openai/o1-preview",  # or "openai/o3-mini"
#     api_key=os.getenv("OPENAI_API_KEY")
# )

# Backup Option 3: Anthropic Claude (UNCOMMENT TO USE)
# planning_model = LiteLlm(
#     model="anthropic/claude-3-5-sonnet-20241022",
#     api_key=os.getenv("ANTHROPIC_API_KEY")
# )

# Option 4: Google Gemini Thinking (UNCOMMENT TO USE)
# planning_model = LiteLlm(
#     model="gemini/gemini-2.0-flash-thinking-exp",
#     api_key=os.getenv("GEMINI_API_KEY")
# )

# --- PERCEPTION AGENT MODEL (Needs JSON output) ---
perception_model = LiteLlm(model="deepseek/deepseek-chat")

# Backup: Ollama local model
# perception_model = LiteLlm(
#     model="ollama_chat/llama3.2:latest",
#     format="json"  # Ensures structured JSON output
# )

# Alternative: GPT-4o with JSON mode
# perception_model = LiteLlm(
#     model="openai/gpt-4o",
#     api_key=os.getenv("OPENAI_API_KEY"),
#     format="json"
# )

# --- SENSING/MOTION AGENTS MODEL (Can use faster models) ---
action_model = LiteLlm(model="deepseek/deepseek-chat")

# Alternative: Fast API model 
# action_model = LiteLlm(
#     model="groq/llama-3.3-70b-versatile",  # Very fast
#     api_key=os.getenv("GROQ_API_KEY")
# )

# --- VALIDATION AGENT MODEL (Vision Language Model) ---
# Option 1: Ollama LLaVA (local vision model)
validation_model = LiteLlm(model="ollama_chat/llava:latest")

# Option 2: OpenAI GPT-4o with vision 
# validation_model = LiteLlm(
#     model="openai/gpt-4o",  # GPT-4o has vision capabilities
#     api_key=os.getenv("OPENAI_API_KEY")
# )

# Option 3: Anthropic Claude 3.5 with vision 
# validation_model = LiteLlm(
#     model="anthropic/claude-3-5-sonnet-20241022",  # Has vision
#     api_key=os.getenv("ANTHROPIC_API_KEY")
# )

# Option 4: Google Gemini with vision
# validation_model = LiteLlm(
#     model="gemini/gemini-2.0-flash-exp",
#     api_key=os.getenv("GEMINI_API_KEY")
# )


# ==============================================================================
# === MCP TOOLSETS CONFIGURATION ===
# ==============================================================================

current_file_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# CoppeliaSim environment for RLBench
COPPELIASIM_ROOT = os.environ.get('COPPELIASIM_ROOT', '/home/roops/CoppeliaSim')
RLBENCH_ENV = dict(os.environ)
RLBENCH_ENV.update({
    'COPPELIASIM_ROOT': COPPELIASIM_ROOT,
    'LD_LIBRARY_PATH': COPPELIASIM_ROOT,
    'DISPLAY': os.environ.get('DISPLAY', ':0'),
    'QT_QPA_PLATFORM': 'xcb',
    'QT_QPA_PLATFORM_PLUGIN_PATH': f'{COPPELIASIM_ROOT}',
})

# Paths to MCP servers
PATH_TO_RLBENCH_SERVER = current_file_dir / "ros_mcp_server" / "rlbench_orchestration_server.py"
PATH_TO_PERCEPTION_SERVER = current_file_dir / "ros_mcp_server" / "perception_orchestration_server.py"

# RLBench MCP Tools (for sensing and motion)
rlbench_tools = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PATH_TO_RLBENCH_SERVER)],
            env=RLBENCH_ENV,
        ),
        timeout=120,
    ),
    tool_filter=["get_camera_observation", "get_current_state", "get_target_position",
                 "reset_task"]  # Sensing + environment tools
)

# Perception Tools (for object detection and 3D localization)
perception_tools = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PATH_TO_PERCEPTION_SERVER)]
        ),
        timeout=120,
    ),
)

# Motion Tools (for robot arm and gripper control)
motion_tools = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(PATH_TO_RLBENCH_SERVER)],
            env=RLBENCH_ENV,
        ),
        timeout=120,
    ),
    tool_filter=["move_to_position", "control_gripper", "lift_gripper"]
)


# ==============================================================================
# === AGENT 1: PLANNING AGENT ===
# ==============================================================================

planning_agent = Agent(
    name="PlanningAgent",
    model=planning_model,  # Uses reasoning model
    description="Autonomous task planner with deep reasoning capabilities",
    instruction=(
        "You are an autonomous robot task planning agent for RLBench simulation tasks.\n\n"

        "**Your Role:**\n"
        "Analyze user requests and generate comprehensive, executable plans for RLBench manipulation tasks.\n\n"

        "**Supported Task Types:**\n"
        "1. **ReachTarget**: Move gripper to touch a colored target sphere\n"
        "2. **PickAndLift**: Pick up a colored block and lift it to a target height\n\n"

        "**Reasoning Process:**\n"
        "1. **IDENTIFY TASK TYPE:**\n"
        "   - Keywords: 'reach', 'touch' → ReachTarget\n"
        "   - Keywords: 'pick', 'lift', 'grasp' → PickAndLift\n"
        "   - Extract target color/description (e.g., 'red target', 'blue block')\n\n"

        "2. **ANALYZE REQUIREMENTS:**\n"
        "   - Sensors: RGB camera, depth map, point cloud (all from single observation)\n"
        "   - Perception: GroundingDINO for 2D detection + depth filtering for 3D (1.7cm accuracy)\n"
        "   - Motion: Direct Cartesian control with path planning\n\n"

        "3. **GENERATE A DETAILED PLAN:**\n"
        "   Your output MUST be a JSON object with this structure:\n"
        "   {\n"
        "     'task_type': 'ReachTarget' or 'PickAndLift',\n"
        "     'task_description': '<Brief summary>',\n"
        "     'target_object': '<Object description for perception (e.g., red target, blue block)>',\n"
        "     'steps': [\n"
        "       {\n"
        "         'step_id': 1,\n"
        "         'phase': 'reset',\n"
        "         'action': 'reset_environment',\n"
        "         'task_name': 'ReachTarget' or 'PickAndLift',\n"
        "         'rationale': 'Initialize RLBench environment',\n"
        "         'success_criteria': 'Environment loaded with random object positions'\n"
        "       },\n"
        "       {\n"
        "         'step_id': 2,\n"
        "         'phase': 'sensing',\n"
        "         'action': 'capture_observation',\n"
        "         'rationale': 'Get RGB, depth, and point cloud from front camera',\n"
        "         'success_criteria': 'All sensor data paths returned'\n"
        "       },\n"
        "       {\n"
        "         'step_id': 3,\n"
        "         'phase': 'perception',\n"
        "         'action': 'detect_object_3d',\n"
        "         'text_prompt': '<target_object from above>',\n"
        "         'rationale': 'Use GroundingDINO + depth filtering to get 3D position',\n"
        "         'success_criteria': 'position_3d returned with confidence > 0.4',\n"
        "         'expected_accuracy': '1.7cm error'\n"
        "       },\n"
        "       {\n"
        "         'step_id': 4,\n"
        "         'phase': 'motion',\n"
        "         'action': '<task-specific motion>',\n"
        "         'sub_actions': '<see task-specific examples below>',\n"
        "         'rationale': '<task-specific rationale>',\n"
        "         'success_criteria': 'task_completed flag is true'\n"
        "       }\n"
        "     ],\n"
        "     'risk_assessment': {\n"
        "       'potential_failures': ['detection_error', 'path_planning_failure', 'grasp_miss'],\n"
        "       'mitigation_strategies': ['compare_with_ground_truth', 'use_planning_mode', 'adjust_grasp_height']\n"
        "     },\n"
        "     'estimated_duration_seconds': <20 for ReachTarget, 60 for PickAndLift>,\n"
        "     'confidence': <0.8-0.95>\n"
        "   }\n\n"

        "**Task-Specific Motion Plans:**\n\n"

        "For ReachTarget (step 4):\n"
        "{\n"
        "  'action': 'move_to_target',\n"
        "  'sub_actions': ['move_to_position(detected_x, detected_y, detected_z)'],\n"
        "  'rationale': 'Simple reach - just move gripper to detected position'\n"
        "}\n\n"

        "For PickAndLift (step 4):\n"
        "{\n"
        "  'action': 'pick_and_lift_sequence',\n"
        "  'sub_actions': [\n"
        "    'ensure_gripper_open',\n"
        "    'move_above_object (z + 0.15m)',\n"
        "    'descend_to_grasp (z + 0.02m)',\n"
        "    'close_gripper',\n"
        "    'lift_to_target (0.25m up or to lift_target_position)'\n"
        "  ],\n"
        "  'rationale': 'Standard pick-and-lift: approach from above, grasp, lift',\n"
        "  'critical_notes': 'Keep gripper CLOSED during lift, use exact lift target from ground truth'\n"
        "}\n\n"

        "4. **IF ASKED TO REVISE:**\n"
        "   - Consider human feedback carefully\n"
        "   - Explain changes and reasoning\n"
        "   - Maintain JSON structure\n\n"

        "**Important Guidelines:**\n"
        "- Task type determines motion sequence\n"
        "- Text prompt for perception must be specific (e.g., 'red block', not just 'block')\n"
        "- ReachTarget is simpler (1 movement), PickAndLift is multi-step\n"
        "- All positions are in robot base frame\n"
        "- Ground truth available for debugging (get_target_position tool)\n"
        "- Output ONLY the JSON plan, no additional commentary\n"
    ),
    tools=[rlbench_tools],  # Can query robot capabilities if needed
    output_key="proposed_plan"
)


# ==============================================================================
# === HUMAN-IN-THE-LOOP: APPROVAL GATE ===
# ==============================================================================

class HumanApprovalGate:
    """
    Interactive approval system for human verification of agent plans.

    Features:
    - Displays generated plans in readable format
    - Allows approval, rejection, or modification
    - Tracks modification history
    """

    def __init__(self):
        self.modification_history = []
        self.approval_log = []

    def request_approval(self, plan: Dict, stage: str = "Planning") -> Dict:
        """
        Display plan and request human approval.

        Args:
            plan: The plan dictionary to review
            stage: Which stage this approval is for (for logging)

        Returns:
            Dictionary with status ('approved', 'rejected', 'modified') and updated plan
        """
        print("\n" + "="*70)
        print(f"🔍 {stage.upper()} - HUMAN REVIEW REQUIRED")
        print("="*70)

        # Pretty-print the plan
        self._display_plan(plan)

        print("\n" + "-"*70)
        print("OPTIONS:")
        print("  1. ✓ APPROVE   - Proceed with this plan")
        print("  2. ✗ REJECT    - Ask agent to regenerate")
        print("  3. ✎ MODIFY    - Edit the plan manually")
        print("  4. ? EXPLAIN   - Ask agent to explain reasoning")
        print("-"*70)

        while True:
            choice = input("\nYour choice (1/2/3/4): ").strip()

            if choice == "1":
                # APPROVED
                self.approval_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": stage,
                    "status": "approved",
                    "plan": plan
                })
                print("\n✓ Plan APPROVED. Proceeding to execution...")
                return {"status": "approved", "plan": plan}

            elif choice == "2":
                # REJECTED
                feedback = input("\n❓ Why are you rejecting this plan? (helps agent improve): ").strip()
                self.approval_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "stage": stage,
                    "status": "rejected",
                    "feedback": feedback
                })
                print("\n✗ Plan REJECTED. Agent will regenerate...")
                return {"status": "rejected", "feedback": feedback, "plan": None}

            elif choice == "3":
                # MODIFY
                print("\n✎ MODIFICATION MODE")
                print("Enter the complete modified plan as JSON.")
                print("(Tip: Copy the displayed plan, edit in a text editor, then paste here)")
                print("\nPaste modified JSON (end with empty line):")

                lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    lines.append(line)

                modified_json = "\n".join(lines)

                try:
                    modified_plan = json.loads(modified_json)
                    self.modification_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "original": plan,
                        "modified": modified_plan
                    })
                    self.approval_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "stage": stage,
                        "status": "modified",
                        "plan": modified_plan
                    })
                    print("\n✓ Plan MODIFIED and APPROVED.")
                    return {"status": "approved", "plan": modified_plan}
                except json.JSONDecodeError as e:
                    print(f"\n❌ Invalid JSON: {e}")
                    print("Please try again or choose a different option.")
                    continue

            elif choice == "4":
                # EXPLAIN
                print("\n💭 AGENT EXPLANATION REQUEST")
                print("(Feature: Would query planning agent to explain its reasoning)")
                print("For now, review the 'rationale' fields in each step.\n")
                continue

            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

    def _display_plan(self, plan: Dict):
        """Pretty-print the plan for human review"""
        print(f"\nTask: {plan.get('task_description', 'N/A')}")
        print(f"Target Object: {plan.get('target_object', 'N/A')}")
        print(f"Destination: {plan.get('destination', 'N/A')}")
        print(f"Estimated Duration: {plan.get('estimated_duration_seconds', 'N/A')} seconds")
        print(f"Confidence: {plan.get('confidence', 'N/A')}")

        print("\n📋 EXECUTION STEPS:")
        for step in plan.get('steps', []):
            print(f"\n  Step {step.get('step_id')}: {step.get('action')}")
            print(f"    Phase: {step.get('phase')}")
            print(f"    Rationale: {step.get('rationale', 'N/A')}")
            print(f"    Success Criteria: {step.get('success_criteria', 'N/A')}")

        risks = plan.get('risk_assessment', {})
        if risks:
            print("\n⚠️  RISK ASSESSMENT:")
            print(f"  Potential Failures: {', '.join(risks.get('potential_failures', []))}")
            print(f"  Mitigations: {', '.join(risks.get('mitigation_strategies', []))}")

        print("\n📄 FULL JSON:")
        print(json.dumps(plan, indent=2))


# ==============================================================================
# === AGENT 2: SENSING AGENT ===
# ==============================================================================

sensing_agent = Agent(
    name="AdaptiveSensingAgent",
    model=action_model,
    description="Intelligently captures sensor data based on plan requirements",
    instruction=(
        "You are an autonomous sensing agent responsible for gathering environmental data.\n\n"

        "**INPUT (via state):**\n"
        "You will receive an approved plan: {approved_plan}\n\n"

        "**Your Responsibilities:**\n\n"

        "1. **ANALYZE THE PLAN:**\n"
        "   - Read the approved plan carefully\n"
        "   - Identify which sensors are needed (check 'required_sensors' in plan steps)\n"
        "   - Understand what data quality is required\n\n"

        "2. **DECIDE ON DATA COLLECTION:**\n"
        "   Based on the plan, determine which tools to call:\n"
        "   - If RGB/color mentioned → call `subscribe_and_download_image`\n"
        "   - If 3D position/point cloud needed → call `subscribe_and_download_pointcloud`\n"
        "   - If depth information needed → call `subscribe_and_download_depth_image`\n\n"

        "3. **CAPTURE DATA:**\n"
        "   - Call the appropriate ROS topic subscription tools\n"
        "   - Store the returned file paths\n\n"

        "4. **VALIDATE DATA QUALITY:**\n"
        "   While you can't directly inspect images, consider:\n"
        "   - Did the tool call succeed?\n"
        "   - Are file paths returned?\n"
        "   - Based on tool response, assess if data capture was successful\n\n"

        "5. **HANDLE ISSUES:**\n"
        "   If data capture fails:\n"
        "   - Try recapturing (if transient error)\n"
        "   - Report the specific error\n"
        "   - Suggest whether replanning is needed\n\n"

        "6. **OUTPUT FORMAT:**\n"
        "   Your final response MUST be a JSON object:\n"
        "   {\n"
        "     'user_task': '<original task from plan>',\n"
        "     'rgb_path': '/path/to/image.png',\n"
        "     'ply_path': '/path/to/pointcloud.ply',\n"
        "     'depth_path': '/path/to/depth.png',  # If captured\n"
        "     'frame_preference': 'base',  # Reference frame\n"
        "     'capture_status': 'success',  # or 'partial' or 'failed'\n"
        "     'quality_assessment': 'good',  # 'good', 'acceptable', 'poor'\n"
        "     'notes': ['Any observations or issues encountered']\n"
        "   }\n\n"

        "**Key Principles:**\n"
        "- You have autonomy to decide which sensors to use\n"
        "- Don't blindly follow instructions - reason about what's needed\n"
        "- If something seems wrong, speak up\n"
        "- Your data quality affects downstream agents' success\n"
    ),
    tools=[ros_mcp_tools],
    output_key="sensor_data"
)


# ==============================================================================
# === AGENT 3: PERCEPTION AGENT ===
# ==============================================================================

perception_agent = Agent(
    name="ResilientPerceptionAgent",
    model=perception_model,  # Uses JSON-mode model
    description="Analyzes sensor data with autonomous error handling and decision-making",
    instruction=(
        "You are an autonomous perception agent with authority to make critical decisions.\n\n"

        "**INPUTS (via state):**\n"
        "- approved_plan: {approved_plan}\n"
        "- sensor_data: {sensor_data}\n\n"

        "**Your Mission:**\n\n"

        "1. **UNDERSTAND THE GOAL:**\n"
        "   - Parse the approved plan to know what object to find\n"
        "   - Understand the task requirements (e.g., 'red cube')\n"
        "   - Know the destination for place planning\n\n"

        "2. **ANALYZE SENSOR DATA:**\n"
        "   - Extract file paths from sensor_data JSON\n"
        "   - Map sensor data to the perception tool requirements:\n"
        "     * 'user_task' → 'task_text' (e.g., 'pick red cube')\n"
        "     * 'rgb_path' → 'rgb_path'\n"
        "     * 'ply_path' → 'ply_path'\n"
        "     * 'frame_preference' → 'frame_preference'\n\n"

        "3. **CALL PERCEPTION TOOL:**\n"
        "   - Use `plan_pick_and_place_tool` with the mapped arguments\n"
        "   - This tool performs:\n"
        "     * Object detection and segmentation\n"
        "     * Grasp pose generation\n"
        "     * Place pose planning\n\n"

        "4. **AUTONOMOUS ERROR HANDLING:**\n"
        "   Based on the tool's response, you have authority to decide:\n\n"

        "   **IF Object Not Detected:**\n"
        "   - OPTION A: Request additional sensor data (different viewpoint)\n"
        "   - OPTION B: Ask human to verify object description\n"
        "   - OPTION C: Report failure and recommend replanning\n"
        "   - Set status to 'needs_human_review'\n\n"

        "   **IF Grasp Confidence is Low (< 0.4):**\n"
        "   - Proceed but flag in output for validation agent to check\n"
        "   - Generate alternative approaches if possible\n"
        "   - Set status to 'needs_review'\n\n"

        "   **IF Unexpected Issues (obstacles, scene clutter):**\n"
        "   - Identify and report them clearly\n"
        "   - Assess if task is still feasible\n"
        "   - Recommend modifications to the plan\n\n"

        "5. **OUTPUT FORMAT:**\n"
        "   Your final response MUST be a JSON object:\n"
        "   {\n"
        "     'status': 'success',  # 'success', 'needs_review', 'failed'\n"
        "     'object_detected': true,\n"
        "     'target_object': 'red cube',\n"
        "     'frame_id': 'base_link',\n"
        "     'grasp_candidates': [\n"
        "       {\n"
        "         'pose': {\n"
        "           'position': [x, y, z],\n"
        "           'orientation_xyzw': [x, y, z, w]\n"
        "         },\n"
        "         'score': 0.85,\n"
        "         'width_required': 0.05\n"
        "       }\n"
        "       // ... more candidates\n"
        "     ],\n"
        "     'place_pose': {\n"
        "       'position': [x, y, z],\n"
        "       'orientation_xyzw': [x, y, z, w]\n"
        "     },\n"
        "     'confidence': 0.85,  # Overall confidence in success\n"
        "     'recommendations': ['proceed', 'use_top_grasp'],  # or ['review_grasp', 'replan']\n"
        "     'issues': [],  # List any problems encountered\n"
        "     'debug': {\n"
        "       'segmented_object_ply': './debug_out/red_cube.ply',\n"
        "       'overlay_png': './debug_out/overlay.png'\n"
        "     },\n"
        "     'notes': ['Additional observations']\n"
        "   }\n\n"

        "**Critical Guidelines:**\n"
        "- You are NOT a passive tool caller - you REASON and DECIDE\n"
        "- Flag anything uncertain for post-execution validation\n"
        "- If tool returns an error, parse it and explain clearly\n"
        "- Your output directly affects robot motion - be precise\n"
        "- Proceed even if confidence is moderate - validation will verify\n"
    ),
    tools=[perception_tools],
    output_key="perception_result"
)


# ==============================================================================
# === AGENT 4: MOTION AGENT ===
# ==============================================================================

motion_agent = Agent(
    name="MonitoredMotionAgent",
    model=action_model,
    description="Executes robot motion with real-time monitoring and safety checks",
    instruction=(
        "You are an autonomous motion execution agent with safety authority.\n\n"

        "**INPUTS (via state):**\n"
        "- approved_plan: {approved_plan}\n"
        "- perception_result: {perception_result}\n\n"

        "**Your Mission:**\n\n"

        "1. **EXTRACT EXECUTION PARAMETERS:**\n"
        "   - Get the top grasp pose from perception_result (use first grasp candidate)\n"
        "   - Get the place pose\n"
        "   - Review perception confidence and recommendations\n\n"

        "2. **PLAN THE MOTION SEQUENCE:**\n"
        "   Standard pick-and-place sequence:\n"
        "   a. Check current robot state (get_joint_states)\n"
        "   b. Plan and execute trajectory to pre-grasp pose\n"
        "   c. Move to grasp pose (send_arm_trajectory)\n"
        "   d. Close gripper to grasp object (send_gripper_command with close=true)\n"
        "   e. Lift object (small upward motion)\n"
        "   f. Move to place pose (send_arm_trajectory)\n"
        "   g. Open gripper to release (send_gripper_command with close=false)\n"
        "   h. Retreat to safe position\n\n"

        "3. **EXECUTE WITH MONITORING:**\n"
        "   For each motion step:\n"
        "   - Call the appropriate tool\n"
        "   - Check the response for success/failure\n"
        "   - Validate robot state after critical actions\n"
        "   - Log each action\n\n"

        "4. **SAFETY CHECKS:**\n"
        "   - Before each motion, verify joint states are valid\n"
        "   - After grasp, verify gripper closure (if feedback available)\n"
        "   - If ANY step fails, STOP immediately\n"
        "   - Don't proceed if something feels wrong\n\n"

        "5. **ERROR HANDLING:**\n"
        "   If a tool call fails:\n"
        "   - PAUSE execution\n"
        "   - Report the specific error clearly\n"
        "   - Assess if recovery is possible\n"
        "   - Don't attempt risky recovery moves\n\n"

        "6. **OUTPUT FORMAT:**\n"
        "   Your final response MUST be a JSON object:\n"
        "   {\n"
        "     'execution_status': 'success',  # 'success', 'partial', 'failed'\n"
        "     'completed_steps': [\n"
        "       {'step': 'get_joint_states', 'status': 'success', 'timestamp': '...'},\n"
        "       {'step': 'move_to_grasp', 'status': 'success', 'timestamp': '...'},\n"
        "       // ... all executed steps\n"
        "     ],\n"
        "     'final_state': {\n"
        "       'joint_positions': [...],\n"
        "       'gripper_state': 'open/closed',\n"
        "       'end_effector_pose': {...}\n"
        "     },\n"
        "     'issues_encountered': [],  # Any problems during execution\n"
        "     'recovery_actions': [],  # Any recovery attempts made\n"
        "     'execution_time_seconds': 42.5,\n"
        "     'notes': ['Additional observations']\n"
        "   }\n\n"

        "**Critical Principles:**\n"
        "- Safety is your HIGHEST priority\n"
        "- Never rush - validate each step\n"
        "- If uncertain, STOP and report\n"
        "- Log everything for validation\n"
        "- A safe partial execution is better than a risky complete one\n"
    ),
    tools=[motion_tools],
    output_key="execution_result"
)


# ==============================================================================
# === AGENT 5: VLM VALIDATION AGENT (Post-Execution) ===
# ==============================================================================

validation_agent = Agent(
    name="VLMValidationAgent",
    model=validation_model,  # Uses vision-language model
    description="Uses vision to verify task completion after motion execution",
    instruction=(
        "You are a vision-based validation agent. Your role is to verify whether a robotic task "
        "was completed successfully by analyzing the final scene.\n\n"

        "**INPUTS (via state):**\n"
        "- original_task: {original_task}\n"
        "- approved_plan: {approved_plan}\n"
        "- execution_result: {execution_result}\n"
        "- validation_image_path: {validation_image_path}  # Image captured after motion\n\n"

        "**Your Mission:**\n\n"

        "1. **UNDERSTAND THE GOAL:**\n"
        "   - Read the original task (e.g., 'Pick red cube and place in tray')\n"
        "   - Identify the target object and destination\n"
        "   - Understand what success looks like\n\n"

        "2. **ANALYZE THE VALIDATION IMAGE:**\n"
        "   - You will receive an image of the scene AFTER motion execution\n"
        "   - Use your vision capabilities to examine the scene\n"
        "   - Look for:\n"
        "     * Is the target object in the destination location?\n"
        "     * Is the object properly placed (not tipped over, aligned correctly)?\n"
        "     * Are there any errors (object dropped, wrong location, etc.)?\n\n"

        "3. **COMPARE AGAINST EXPECTATIONS:**\n"
        "   - Based on the task description, determine if the goal was achieved\n"
        "   - Consider partial success (e.g., object moved but not perfectly placed)\n"
        "   - Identify any discrepancies\n\n"

        "4. **ASSESS EXECUTION RESULT:**\n"
        "   - Review the execution_result from the motion agent\n"
        "   - Did the motion agent report success?\n"
        "   - Does the visual evidence match the reported status?\n\n"

        "5. **OUTPUT FORMAT:**\n"
        "   Your response MUST be a JSON object:\n"
        "   {\n"
        "     'task_completed': true,  # true/false - was the task successful?\n"
        "     'confidence': 0.95,  # 0.0-1.0 confidence in your assessment\n"
        "     'visual_observations': [\n"
        "       'Red cube is visible in the tray',\n"
        "       'Cube appears upright and stable',\n"
        "       'No other objects disturbed'\n"
        "     ],\n"
        "     'discrepancies': [],  # List any issues found (empty if task successful)\n"
        "     'success_criteria_met': {\n"
        "       'object_in_destination': true,\n"
        "       'object_stable': true,\n"
        "       'no_collisions': true,\n"
        "       'workspace_clear': true\n"
        "     },\n"
        "     'recommendation': 'approve',  # 'approve', 'retry', 'replan'\n"
        "     'explanation': 'The red cube has been successfully placed in the tray. The object is stable and properly positioned.'\n"
        "   }\n\n"

        "**Critical Guidelines:**\n"
        "- Be thorough and honest in your assessment\n"
        "- If you can't see clearly, report lower confidence\n"
        "- Look for both success AND potential issues\n"
        "- Your validation determines if the task is truly complete\n"
        "- Err on the side of caution - if unsure, flag for human review\n"
    ),
    tools=[ros_mcp_tools],  # Can capture validation image
    output_key="validation_result"
)


# ==============================================================================
# === ORCHESTRATOR: SIMPLIFIED AGENTIC WORKFLOW MANAGER ===
# ==============================================================================

class AgenticOrchestrator:
    """
    Orchestrates the simplified agentic workflow with human-in-the-loop.

    Workflow:
    1. Planning → Human Approval (with revision loop)
    2. Sensing → Adaptive data capture
    3. Perception → Analysis with error handling
    4. Motion → Monitored execution
    5. VLM Validation → Verify task completion with vision
    """

    def __init__(self):
        self.planner = planning_agent
        self.human_gate = HumanApprovalGate()
        self.sensor = sensing_agent
        self.perception = perception_agent
        self.motion = motion_agent
        self.validator = validation_agent

        # Memory system for learning (optional)
        self.task_memory = []
        self.session_start = datetime.now()

    def execute_task(self, user_task: str, max_planning_attempts: int = 3) -> Dict:
        """
        Execute a complete robotic task with simplified agentic workflow.

        Args:
            user_task: Natural language task description
            max_planning_attempts: Maximum number of planning revisions allowed

        Returns:
            Complete task state with all agent outputs and validation
        """
        print("\n" + "="*70)
        print("🤖 AGENTIC ROBOT MANIPULATION SYSTEM - V2 (SIMPLIFIED)")
        print("="*70)
        print(f"Task: {user_task}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        # Initialize task state
        state = {
            "original_task": user_task,
            "session_id": self.session_start.isoformat(),
            "human_interventions": []
        }

        # ======================================================================
        # STAGE 1: PLANNING (with human approval loop)
        # ======================================================================
        print("\n" + "▶"*35)
        print("📋 STAGE 1: PLANNING")
        print("▶"*35)

        plan_approved = False
        for attempt in range(1, max_planning_attempts + 1):
            print(f"\n🔄 Planning attempt {attempt}/{max_planning_attempts}...")

            # Generate plan
            planning_input = {
                "query": user_task,
                "previous_feedback": state.get("planning_feedback", "")
            }

            try:
                plan_response = self.planner.query(planning_input)
                proposed_plan_json = plan_response.get("proposed_plan", "{}")
                proposed_plan = json.loads(proposed_plan_json)
            except Exception as e:
                print(f"❌ Planning failed: {e}")
                continue

            # Human approval gate
            approval = self.human_gate.request_approval(proposed_plan, stage="Planning")
            state["human_interventions"].append({
                "stage": "planning",
                "attempt": attempt,
                "decision": approval["status"]
            })

            if approval["status"] == "approved":
                state["approved_plan"] = approval["plan"]
                plan_approved = True
                print(f"✅ Plan approved after {attempt} attempt(s)")
                break
            elif approval["status"] == "rejected":
                state["planning_feedback"] = approval.get("feedback", "Plan rejected by human")
                print(f"🔄 Plan rejected. Regenerating...")
                continue

        if not plan_approved:
            print(f"\n❌ TASK FAILED: Planning could not be approved after {max_planning_attempts} attempts")
            return state

        # ======================================================================
        # STAGE 2: SENSING
        # ======================================================================
        print("\n" + "▶"*35)
        print("👁️  STAGE 2: SENSING")
        print("▶"*35)

        try:
            sensor_input = {"approved_plan": json.dumps(state["approved_plan"])}
            sensor_response = self.sensor.query(sensor_input)
            sensor_data_json = sensor_response.get("sensor_data", "{}")
            state["sensor_data"] = json.loads(sensor_data_json)

            print(f"✅ Sensor data captured:")
            print(f"   RGB: {state['sensor_data'].get('rgb_path', 'N/A')}")
            print(f"   PLY: {state['sensor_data'].get('ply_path', 'N/A')}")
            print(f"   Status: {state['sensor_data'].get('capture_status', 'unknown')}")
        except Exception as e:
            print(f"❌ SENSING FAILED: {e}")
            return state

        # ======================================================================
        # STAGE 3: PERCEPTION
        # ======================================================================
        print("\n" + "▶"*35)
        print("🧠 STAGE 3: PERCEPTION")
        print("▶"*35)

        try:
            perception_input = {
                "approved_plan": json.dumps(state["approved_plan"]),
                "sensor_data": json.dumps(state["sensor_data"])
            }
            perception_response = self.perception.query(perception_input)
            perception_result_json = perception_response.get("perception_result", "{}")
            state["perception_result"] = json.loads(perception_result_json)

            print(f"✅ Perception complete:")
            print(f"   Status: {state['perception_result'].get('status', 'unknown')}")
            print(f"   Object Detected: {state['perception_result'].get('object_detected', False)}")
            print(f"   Confidence: {state['perception_result'].get('confidence', 0.0):.2f}")
            print(f"   Grasp Candidates: {len(state['perception_result'].get('grasp_candidates', []))}")

            # Check for critical failures
            if state['perception_result'].get('status') == 'failed':
                print(f"\n❌ PERCEPTION FAILED - Cannot proceed to motion")
                return state

        except Exception as e:
            print(f"❌ PERCEPTION FAILED: {e}")
            return state

        # ======================================================================
        # STAGE 4: MOTION EXECUTION (No pre-validation)
        # ======================================================================
        print("\n" + "▶"*35)
        print("🦾 STAGE 4: MOTION EXECUTION")
        print("▶"*35)

        try:
            motion_input = {
                "approved_plan": json.dumps(state["approved_plan"]),
                "perception_result": json.dumps(state["perception_result"])
            }
            motion_response = self.motion.query(motion_input)
            execution_result_json = motion_response.get("execution_result", "{}")
            state["execution_result"] = json.loads(execution_result_json)

            exec_status = state["execution_result"].get("execution_status", "unknown")
            print(f"✅ Motion execution complete:")
            print(f"   Status: {exec_status}")
            print(f"   Steps Completed: {len(state['execution_result'].get('completed_steps', []))}")
            print(f"   Issues: {len(state['execution_result'].get('issues_encountered', []))}")
        except Exception as e:
            print(f"❌ MOTION EXECUTION FAILED: {e}")
            state["execution_result"] = {
                "execution_status": "failed",
                "error": str(e)
            }

        # ======================================================================
        # STAGE 5: VLM VALIDATION (Post-Execution Verification)
        # ======================================================================
        print("\n" + "▶"*35)
        print("🔍 STAGE 5: VLM VALIDATION")
        print("▶"*35)

        try:
            # Capture validation image
            print("📸 Capturing validation image...")
            validation_image_response = self.sensor.query({"action": "capture_validation_image"})
            validation_image_path = validation_image_response.get("image_path", "")

            # Run VLM validation
            validation_input = {
                "original_task": state["original_task"],
                "approved_plan": json.dumps(state["approved_plan"]),
                "execution_result": json.dumps(state["execution_result"]),
                "validation_image_path": validation_image_path
            }
            validation_response = self.validator.query(validation_input)
            validation_result_json = validation_response.get("validation_result", "{}")
            state["validation_result"] = json.loads(validation_result_json)

            task_completed = state["validation_result"].get("task_completed", False)
            confidence = state["validation_result"].get("confidence", 0.0)
            recommendation = state["validation_result"].get("recommendation", "unknown")

            print(f"✅ Validation complete:")
            print(f"   Task Completed: {task_completed}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Recommendation: {recommendation}")
            print(f"\n📝 Explanation:")
            print(f"   {state['validation_result'].get('explanation', 'N/A')}")

        except Exception as e:
            print(f"⚠️  VALIDATION FAILED: {e}")
            print(f"   Cannot verify task completion visually")
            state["validation_result"] = {
                "task_completed": "unknown",
                "error": str(e)
            }

        # Store in memory
        self.task_memory.append(state)

        # Final summary
        self._print_final_summary(state)

        return state

    def _print_final_summary(self, state: Dict):
        """Print final task summary"""
        print("\n" + "="*70)
        print("📊 TASK SUMMARY")
        print("="*70)

        # Execution outcome
        exec_status = state.get("execution_result", {}).get("execution_status", "unknown")
        print(f"\nMotion Execution: {exec_status.upper()}")

        # Validation outcome
        validation_result = state.get("validation_result", {})
        task_completed = validation_result.get("task_completed", "unknown")
        confidence = validation_result.get("confidence", 0.0)

        if task_completed == True:
            print(f"✅ TASK VERIFIED AS COMPLETE (Confidence: {confidence:.2f})")
        elif task_completed == False:
            print(f"❌ TASK NOT COMPLETED (Confidence: {confidence:.2f})")
        else:
            print(f"⚠️  TASK COMPLETION UNKNOWN (Validation Error)")

        # Visual observations
        observations = validation_result.get("visual_observations", [])
        if observations:
            print(f"\n👁️  Visual Observations:")
            for obs in observations:
                print(f"  • {obs}")

        # Discrepancies
        discrepancies = validation_result.get("discrepancies", [])
        if discrepancies:
            print(f"\n⚠️  Issues Found:")
            for disc in discrepancies:
                print(f"  • {disc}")

        print("\n" + "="*70)
        print(f"Session completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

    def query(self, input_data: Dict) -> Dict:
        """
        ADK-compatible query interface.

        This method provides compatibility with the ADK web interface
        by wrapping the execute_task method.

        Args:
            input_data: Dictionary with 'query' key containing the task

        Returns:
            Complete task state with all agent outputs
        """
        # Extract the task from the input
        user_task = input_data.get("query", "")

        if not user_task:
            return {
                "error": "No query provided",
                "status": "failed"
            }

        # Execute the full agentic workflow
        result = self.execute_task(user_task)

        return result


# ==============================================================================
# === ROOT AGENT (for ADK compatibility) ===
# ==============================================================================

# For direct Python execution with full human-in-the-loop workflow
_orchestrator_instance = AgenticOrchestrator()

# For ADK web interface, create a simplified sequential pipeline
# (without human approval gates since web UI doesn't support interactive input)
root_agent = SequentialAgent(
    name="SimplifiedRobotManipulationPipeline",
    sub_agents=[
        sensing_agent,
        perception_agent,
        motion_agent
    ],
    description=(
        "Simplified robotic manipulation pipeline for web interface.\n\n"
        "Workflow:\n"
        "1. Sensing Agent - Captures RGB image and point cloud data\n"
        "2. Perception Agent - Detects objects and plans grasp/place poses\n"
        "3. Motion Agent - Executes the pick-and-place motion\n\n"
        "Note: For the full agentic workflow with human-in-the-loop approval, "
        "run this module directly with Python."
    )
)


# ==============================================================================
# === MAIN EXECUTION ===
# ==============================================================================

if __name__ == "__main__":
    # Example task
    task = "Pick the red cube and place it in the tray"

    # Execute with full agentic workflow (human-in-the-loop)
    result = _orchestrator_instance.execute_task(task)

    # Optional: Save session to file for later analysis
    session_file = f"task_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(session_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Session saved to: {session_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save session: {e}")
