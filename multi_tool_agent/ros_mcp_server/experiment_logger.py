"""
Experiment Logger for HAMMR Evaluation
=======================================
Shared by both MCP servers (rlbench_orchestration_server, perception_orchestration_server).
Writes JSONL records to disk — one JSON object per line, append-safe across processes.

Record types:
  trial_start   — emitted by orchestration server at load_task()
  perception    — emitted by perception server at detect_object_3d()
  motion_step   — emitted by orchestration server at each move_to_position()

Results directory: multi_tool_agent/evaluation/results/
"""

import json
import fcntl
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).parent.parent / "evaluation" / "results"
TRIALS_LOG = EVAL_DIR / "trials.jsonl"
_TRIAL_ID_FILE = EVAL_DIR / ".current_trial_id"   # shared temp file between processes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _append(record: dict):
    """Process-safe append of one JSON record to the JSONL log file."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    record["logged_at"] = datetime.now().isoformat()
    with open(TRIALS_LOG, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Trial lifecycle (called by orchestration server)
# ---------------------------------------------------------------------------

def start_trial(task_name: str) -> str:
    """
    Mark the start of a new trial. Call from load_task().
    Returns a trial_id string that identifies this trial in all subsequent records.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    trial_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    _TRIAL_ID_FILE.write_text(trial_id)
    _append({
        "event": "trial_start",
        "trial_id": trial_id,
        "task": task_name,
    })
    return trial_id


def get_current_trial_id() -> str:
    """
    Read the current trial_id written by the orchestration server.
    Called by the perception server (separate process, no shared memory).
    Returns 'unknown' if no trial has been started yet.
    """
    if _TRIAL_ID_FILE.exists():
        return _TRIAL_ID_FILE.read_text().strip()
    return "unknown"


# ---------------------------------------------------------------------------
# Perception metrics (called by perception server)
# ---------------------------------------------------------------------------

def log_perception(
    trial_id: str,
    task: str,
    success: bool,
    confidence: float,
    inference_time_ms: float,
    position_3d: list,
    gt_position: list = None,
    failure_reason: str = None,
):
    """
    Log the result of one detect_object_3d() call.

    Args:
        trial_id:           from get_current_trial_id()
        task:               task name string
        success:            whether detection succeeded
        confidence:         GroundingDINO logit score (0–1)
        inference_time_ms:  time spent inside GroundingDINO predict()
        position_3d:        detected [x, y, z] in robot base frame
        gt_position:        ground truth [x, y, z] (optional, used to compute error)
        failure_reason:     e.g. "no_detection", "model_not_loaded"
    """
    record = {
        "event": "perception",
        "trial_id": trial_id,
        "task": task,
        "success": success,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "inference_time_ms": round(inference_time_ms, 1),
        "position_3d": [round(v, 4) for v in position_3d] if position_3d else None,
    }

    if gt_position and position_3d:
        import numpy as np
        error_m = float(np.linalg.norm(np.array(position_3d) - np.array(gt_position)))
        record["gt_position"] = [round(v, 4) for v in gt_position]
        record["position_error_cm"] = round(error_m * 100, 2)

    if failure_reason:
        record["failure_reason"] = failure_reason

    _append(record)


# ---------------------------------------------------------------------------
# Motion metrics (called by orchestration server)
# ---------------------------------------------------------------------------

def log_motion_step(
    trial_id: str,
    task: str,
    step: int,
    target: list,
    final: list,
    error_distance_m: float,
    task_completed: bool,
    time_ms: float,
    method: str,
    success: bool,
    failure_reason: str = None,
):
    """
    Log the result of one move_to_position() call.

    Args:
        trial_id:           current trial id
        task:               task name string
        step:               sequential step number within this trial (1, 2, 3, ...)
        target:             requested [x, y, z]
        final:              actual reached [x, y, z]
        error_distance_m:   Euclidean distance between target and final (metres)
        task_completed:     RLBench terminate flag
        time_ms:            wall clock time for this move
        method:             "planning" or "ik"
        success:            whether move_to_position returned success=True
        failure_reason:     e.g. "ik_error", "planning_failed" (if success=False)
    """
    record = {
        "event": "motion_step",
        "trial_id": trial_id,
        "task": task,
        "step": step,
        "target_position": [round(v, 4) for v in target],
        "final_position": [round(v, 4) for v in final] if final else None,
        "error_distance_m": round(error_distance_m, 4) if error_distance_m is not None else None,
        "error_distance_cm": round(error_distance_m * 100, 2) if error_distance_m is not None else None,
        "task_completed": task_completed,
        "time_ms": round(time_ms, 1),
        "method": method,
        "success": success,
    }

    if failure_reason:
        record["failure_reason"] = failure_reason

    _append(record)
