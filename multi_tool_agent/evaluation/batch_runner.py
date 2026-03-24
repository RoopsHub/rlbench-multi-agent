"""
Batch Runner for HAMMR Evaluation
===================================
Runs the full root_agent system N times per task, auto-approving the plan.

Session strategy:
  - One session per task type  →  MCP server (CoppeliaSim) stays alive across trials
  - New session per task type  →  fresh LLM context when switching tasks
  - load_task() inside each trial resets object positions (fast, no CoppeliaSim restart)

Trial result source:
  - Success/failure determined from trials.jsonl (task_completed flag from RLBench),
    NOT from agent text output

Outputs:
  - multi_tool_agent/evaluation/results/per_trial.csv   (one row per trial)
  - multi_tool_agent/evaluation/results/summary.csv     (aggregated per task)

Usage:
  cd /home/roops/rlbench-multi-agent
  source .venv/bin/activate

  # Run all tasks, 10 trials each
  python multi_tool_agent/evaluation/batch_runner.py

  # Run specific tasks with custom trial count
  python multi_tool_agent/evaluation/batch_runner.py --tasks PickAndLift StackBlocks --n 5
"""

import asyncio
import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

# Add multi_tool_agent to path so we can import agent.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent import root_agent

# ==============================================================================
# Paths
# ==============================================================================

RESULTS_DIR = Path(__file__).parent / "results"
TRIAL_ID_FILE = RESULTS_DIR / ".current_trial_id"
TRIALS_LOG    = RESULTS_DIR / "trials.jsonl"

# ==============================================================================
# Configuration
# ==============================================================================

TASKS = {
    "ReachTarget":     "Reach the red target",
    "PickAndLift":     "Pick up the red block and lift it to the target",
    "PushButton":      "Push the button",
    "PutRubbishInBin": "Put the rubbish in the bin",
    "StackBlocks":     "Stack the blocks",
}

N_TRIALS          = 10
TRIAL_TIMEOUT_S   = 300   # 5 min max per trial before giving up

APPROVAL_KEYWORDS = ["AWAITING APPROVAL", "Awaiting your approval", "awaiting approval"]

# ==============================================================================
# Section 1: Trial result extraction from trials.jsonl
# ==============================================================================

def get_trial_result(trial_id: str, task_name: str, trial_num: int) -> dict:
    """
    Read trials.jsonl, filter by trial_id, compute all metrics.
    This is the ground truth — not the agent's text output.
    """
    base = {
        "task": task_name,
        "trial_num": trial_num,
        "trial_id": trial_id,
        "success": False,
        "failure_reason": None,
        "total_time_s": None,
        "perception_success": None,
        "perception_confidence": None,
        "perception_inference_ms": None,
        "perception_error_cm": None,
        "motion_steps": 0,
        "mean_motion_error_cm": None,
    }

    if not TRIALS_LOG.exists():
        base["failure_reason"] = "no_log_file"
        return base

    events = []
    with open(TRIALS_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("trial_id") == trial_id:
                    events.append(record)
            except json.JSONDecodeError:
                continue

    if not events:
        base["failure_reason"] = "no_events_logged"
        return base

    perception_events = [e for e in events if e["event"] == "perception"]
    motion_events     = [e for e in events if e["event"] == "motion_step"]
    trial_start       = next((e for e in events if e["event"] == "trial_start"), None)

    # --- Success: any motion step where RLBench signalled terminate ---
    success = any(e.get("task_completed", False) for e in motion_events)

    # --- Failure reason ---
    failure_reason = None
    if not success:
        perc_failed = any(not e.get("success", True) for e in perception_events)
        no_perception = len(perception_events) == 0
        if no_perception or perc_failed:
            failure_reason = "perception_failure"
        elif not motion_events:
            failure_reason = "no_motion_executed"
        elif any(not e.get("success", True) for e in motion_events):
            failure_reason = "motion_ik_error"
        else:
            failure_reason = "task_incomplete"

    # --- Total trial time ---
    total_time_s = None
    if trial_start and events:
        try:
            t0     = datetime.fromisoformat(trial_start["logged_at"])
            t_last = datetime.fromisoformat(events[-1]["logged_at"])
            total_time_s = round((t_last - t0).total_seconds(), 2)
        except Exception:
            pass

    # --- Perception metrics (first successful perception event) ---
    perc = next((e for e in perception_events if e.get("success")),
                perception_events[0] if perception_events else None)
    perception_success     = perc.get("success") if perc else None
    perception_confidence  = perc.get("confidence") if perc else None
    perception_infer_ms    = perc.get("inference_time_ms") if perc else None
    perception_error_cm    = perc.get("position_error_cm") if perc else None

    # --- Motion metrics ---
    motion_steps = len(motion_events)
    errors = [e["error_distance_cm"] for e in motion_events if e.get("error_distance_cm") is not None]
    mean_motion_error_cm = round(sum(errors) / len(errors), 3) if errors else None

    return {
        "task":                    task_name,
        "trial_num":               trial_num,
        "trial_id":                trial_id,
        "success":                 success,
        "failure_reason":          failure_reason,
        "total_time_s":            total_time_s,
        "perception_success":      perception_success,
        "perception_confidence":   perception_confidence,
        "perception_inference_ms": perception_infer_ms,
        "perception_error_cm":     perception_error_cm,
        "motion_steps":            motion_steps,
        "mean_motion_error_cm":    mean_motion_error_cm,
    }


# ==============================================================================
# Section 2: Run a single trial
# ==============================================================================

async def run_trial(
    runner: Runner,
    session_id: str,
    user_id: str,
    prompt: str,
    task_name: str,
    trial_num: int,
) -> str:
    """
    Execute one trial. Returns the trial_id string.

    Turn 1: send task prompt → wait for AWAITING APPROVAL
    Turn 2: send "approved"  → wait for execution to finish
    """
    print(f"  [Trial {trial_num}] → \"{prompt}\"")

    # ------------------------------------------------------------------ Turn 1
    plan_text = ""
    approval_seen = False

    try:
        async with asyncio.timeout(TRIAL_TIMEOUT_S):
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=Content(role="user", parts=[Part(text=prompt)]),
            ):
                if event.is_final_response() and event.content:
                    for part in (event.content.parts or []):
                        if hasattr(part, "text") and part.text:
                            plan_text += part.text
                    if any(kw.lower() in plan_text.lower() for kw in APPROVAL_KEYWORDS):
                        approval_seen = True
                        break
    except asyncio.TimeoutError:
        print(f"  [Trial {trial_num}] Timeout waiting for plan — skipping trial")
        return "timeout"

    # Read trial_id written by the most recent load_task() call
    trial_id = TRIAL_ID_FILE.read_text().strip() if TRIAL_ID_FILE.exists() else "unknown"

    # If the agent already executed during Turn 1 (skipped planning due to context),
    # skip Turn 2 to avoid triggering a duplicate execution
    already_executed = ("SUCCESS" in plan_text or "FAILED" in plan_text)

    if already_executed:
        agent_status = "SUCCESS" if "SUCCESS" in plan_text else "FAILED"
        print(f"  [Trial {trial_num}] Agent executed in Turn 1 (no approval prompt) — "
              f"agent: {agent_status} | trial_id: {trial_id}")
        return trial_id

    if not approval_seen:
        print(f"  [Trial {trial_num}] Warning: approval prompt not detected, proceeding anyway")

    print(f"  [Trial {trial_num}] Plan ready — sending 'approved'")

    # ------------------------------------------------------------------ Turn 2
    exec_text = ""

    try:
        async with asyncio.timeout(TRIAL_TIMEOUT_S):
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=Content(role="user", parts=[Part(text="approved")]),
            ):
                if event.is_final_response() and event.content:
                    for part in (event.content.parts or []):
                        if hasattr(part, "text") and part.text:
                            exec_text += part.text
    except asyncio.TimeoutError:
        print(f"  [Trial {trial_num}] Timeout during execution — marking as failed")
        return "timeout"

    # Re-read trial_id after Turn 2 (load_task may have been called again)
    if TRIAL_ID_FILE.exists():
        trial_id = TRIAL_ID_FILE.read_text().strip()

    agent_status = ("SUCCESS" if "SUCCESS" in exec_text
                    else "FAILED" if "FAILED" in exec_text
                    else "unknown")
    print(f"  [Trial {trial_num}] Finished — agent: {agent_status} | trial_id: {trial_id}")

    return trial_id


# ==============================================================================
# Section 3: Summarize and write CSVs
# ==============================================================================

def _mean(values):
    v = [x for x in values if x is not None]
    return round(sum(v) / len(v), 3) if v else None

def _std(values):
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return round((sum((x - m) ** 2 for x in v) / len(v)) ** 0.5, 3)


def write_results(all_results: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Per-trial CSV (append, renumber from last existing trial) ---
    per_trial_path = RESULTS_DIR / "per_trial.csv"
    existing_rows: list[dict] = []
    if per_trial_path.exists():
        with open(per_trial_path, newline="") as f:
            existing_rows = list(csv.DictReader(f))

    # Find the highest trial_num already recorded per task
    last_trial_num: dict[str, int] = {}
    for row in existing_rows:
        task = row["task"]
        try:
            n = int(row["trial_num"])
        except (ValueError, KeyError):
            n = 0
        last_trial_num[task] = max(last_trial_num.get(task, 0), n)

    # Renumber incoming results continuing from last_trial_num
    counters = dict(last_trial_num)
    renumbered = []
    for r in all_results:
        task = r["task"]
        counters[task] = counters.get(task, 0) + 1
        renumbered.append({**r, "trial_num": counters[task]})

    if renumbered:
        write_header = not per_trial_path.exists() or per_trial_path.stat().st_size == 0
        with open(per_trial_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=renumbered[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(renumbered)

    # --- Summary CSV (recalculate from ALL rows: existing + new) ---
    all_rows = existing_rows + renumbered
    # Cast numeric fields back from strings if coming from existing_rows
    for row in all_rows:
        for key in ("total_time_s", "perception_confidence", "perception_inference_ms", "mean_motion_error_cm"):
            if isinstance(row.get(key), str):
                try:
                    row[key] = float(row[key]) if row[key] else None
                except ValueError:
                    row[key] = None
        if isinstance(row.get("success"), str):
            row["success"] = row["success"].strip().lower() == "true"

    summary_rows = []
    tasks_seen = list(dict.fromkeys(r["task"] for r in all_rows))

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for task in tasks_seen:
        t = [r for r in all_rows if r["task"] == task]
        n = len(t)
        n_success = sum(1 for r in t if r["success"])

        failures = [r["failure_reason"] for r in t if not r["success"] and r["failure_reason"]]
        failure_counts = {}
        for f in failures:
            failure_counts[f] = failure_counts.get(f, 0) + 1

        row = {
            "task":                         task,
            "n_trials":                     n,
            "n_success":                    n_success,
            "success_rate_pct":             round(100 * n_success / n, 1) if n else 0,
            "mean_total_time_s":            _mean([r["total_time_s"] for r in t]),
            "std_total_time_s":             _std([r["total_time_s"] for r in t]),
            "mean_perception_confidence":   _mean([r["perception_confidence"] for r in t]),
            "mean_inference_ms":            _mean([r["perception_inference_ms"] for r in t]),
            "mean_motion_error_cm":         _mean([r["mean_motion_error_cm"] for r in t]),
            "fail_perception":              failure_counts.get("perception_failure", 0),
            "fail_ik_error":                failure_counts.get("motion_ik_error", 0),
            "fail_task_incomplete":         failure_counts.get("task_incomplete", 0),
            "fail_timeout":                 failure_counts.get("timeout", 0),
        }
        summary_rows.append(row)

        print(f"\n  {task}")
        print(f"    Success rate      : {n_success}/{n} ({row['success_rate_pct']}%)")
        print(f"    Total time        : {row['mean_total_time_s']}s ± {row['std_total_time_s']}s")
        print(f"    Perception conf   : {row['mean_perception_confidence']}")
        print(f"    Inference time    : {row['mean_inference_ms']}ms")
        print(f"    Motion error      : {row['mean_motion_error_cm']}cm")
        print(f"    Failures          : {failure_counts or 'none'}")

    summary_path = RESULTS_DIR / "summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\n  per_trial.csv → {per_trial_path}")
    print(f"  summary.csv   → {summary_path}")


# ==============================================================================
# Section 4: Main
# ==============================================================================

async def main(tasks: dict, n_trials: int):
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="hammr_eval",
        agent=root_agent,
        session_service=session_service,
    )

    all_results = []
    user_id = "eval_user"

    for task_name, first_prompt in tasks.items():
        print(f"\n{'='*70}")
        print(f"TASK: {task_name}  |  {n_trials} trials")
        print(f"{'='*70}")

        # One session per task — MCP server stays alive across all trials of this task
        session = await session_service.create_session(
            app_name="hammr_eval",
            user_id=user_id,
        )

        for trial_num in range(1, n_trials + 1):
            # Trial 1: full task prompt
            # Trial 2+: explicitly ask for plan first to prevent agent skipping approval
            prompt = (first_prompt if trial_num == 1
                      else f"Please show me the plan for {task_name} again and wait for my approval before executing.")

            trial_id = await run_trial(runner, session.id, user_id, prompt, task_name, trial_num)

            if trial_id in ("timeout", "unknown"):
                result = {
                    "task": task_name, "trial_num": trial_num, "trial_id": trial_id,
                    "success": False, "failure_reason": trial_id,
                    "total_time_s": None, "perception_success": None,
                    "perception_confidence": None, "perception_inference_ms": None,
                    "perception_error_cm": None, "motion_steps": 0,
                    "mean_motion_error_cm": None,
                }
            else:
                result = get_trial_result(trial_id, task_name, trial_num)

            all_results.append(result)

            icon = "✓" if result["success"] else "✗"
            print(f"  [{icon}] Trial {trial_num:2d}: "
                  f"success={result['success']}, "
                  f"time={result['total_time_s']}s, "
                  f"{'ok' if result['success'] else result.get('failure_reason', '?')}")

    write_results(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAMMR batch evaluation runner")
    parser.add_argument(
        "--tasks", nargs="+", choices=list(TASKS.keys()), default=list(TASKS.keys()),
        help="Tasks to evaluate (default: all)"
    )
    parser.add_argument(
        "--n", type=int, default=N_TRIALS,
        help=f"Trials per task (default: {N_TRIALS})"
    )
    args = parser.parse_args()

    selected = {k: TASKS[k] for k in args.tasks}

    # Suppress MCP async generator cleanup warnings (harmless, Python 3.13 + anyio issue)
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*never awaited.*")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(selected, args.n))
    finally:
        # Cancel remaining tasks to suppress MCP stdio cleanup noise
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
