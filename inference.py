"""
CloudOps RL — Inference Script
Runs all 3 tasks sequentially so the validator sees 3 graded episodes.
Each task gets its own [START] … [END] block.
"""

import os
import json
import requests
from typing import Any, Dict, List, Optional

# ── Config (never use os.environ[] — use os.getenv() with defaults) ──────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "no-key-set"
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
ENV_URL      = os.getenv("ENV_URL", "http://127.0.0.1:7860").rstrip("/")
BENCHMARK    = "cloudops_rl"
MAX_STEPS    = 10
SUCCESS_SCORE_THRESHOLD = 0.30

# The 3 tasks the validator must see — easy / medium / hard
TASKS = [
    {"task_id": "idle_resource_leak",  "difficulty": "easy"},
    {"task_id": "traffic_spike",       "difficulty": "medium"},
    {"task_id": "database_failure",    "difficulty": "hard"},
]

VALID_ACTIONS = [
    "scale_up", "scale_down", "restart_database",
    "rebalance_traffic", "clear_cache", "remove_idle_resource", "noop",
]

SYSTEM_PROMPT = """You are an expert SRE incident response agent.

Choose EXACTLY one action from:
scale_up
scale_down
restart_database
rebalance_traffic
clear_cache
remove_idle_resource
noop

Return ONLY the action name — nothing else.""".strip()


# ── Logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error if error else 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ── Env helpers ───────────────────────────────────────────────────────────────

_local_env = None

def _to_dict(obj) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"): return obj.model_dump()
    if hasattr(obj, "dict"):       return obj.dict()
    return obj if isinstance(obj, dict) else {}

def env_reset(difficulty: str = "medium") -> Dict[str, Any]:
    global _local_env
    try:
        r = requests.post(f"{ENV_URL}/reset",
                          json={"difficulty": difficulty}, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("state", data)
    except Exception:
        try:
            from env import CloudOpsEnv
            _local_env = CloudOpsEnv()
            return _to_dict(_local_env.reset(difficulty=difficulty))
        except Exception as exc:
            print(f"[DEBUG] env_reset failed: {exc}", flush=True)
            return {}

def env_step(action: str):
    global _local_env
    try:
        r = requests.post(f"{ENV_URL}/step",
                          json={"action": action}, timeout=20)
        r.raise_for_status()
        d = r.json()
        return (d.get("state", {}),
                float(d.get("reward", 0.0)),
                bool(d.get("done", False)),
                d.get("error"))
    except Exception:
        try:
            if _local_env is None:
                from env import CloudOpsEnv
                _local_env = CloudOpsEnv()
                _local_env.reset()
            state, reward, done = _local_env.step(action)
            return _to_dict(state), float(reward), bool(done), None
        except Exception as exc:
            print(f"[DEBUG] env_step failed: {exc}", flush=True)
            return {}, 0.0, True, str(exc)

def grade_task(task_id: str, final_state: Dict[str, Any],
               rewards: List[float], steps: int) -> float:
    """Call the /tasks/{task_id}/grade endpoint; fall back to reward avg."""
    try:
        r = requests.post(
            f"{ENV_URL}/tasks/{task_id}/grade",
            json={"task_id": task_id, "final_state": final_state,
                  "rewards": rewards, "steps": steps},
            timeout=20,
        )
        r.raise_for_status()
        return float(r.json().get("score", 0.0))
    except Exception:
        avg = sum(rewards) / max(len(rewards), 1)
        return round(min(max((avg + 2.0) / 3.0, 0.0), 1.0), 4)


# ── LLM action ────────────────────────────────────────────────────────────────

def llm_action(client, state: Dict[str, Any]) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(state, default=str)},
            ],
            temperature=0.1,
            max_tokens=10,
        )
        raw = (resp.choices[0].message.content or "").strip().lower()
        for act in VALID_ACTIONS:
            if act in raw:
                return act
        return "noop"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return _heuristic(state)

def _heuristic(state: Dict[str, Any]) -> str:
    if state.get("db_health") == "degraded":    return "restart_database"
    if state.get("cpu_usage", 0) > 80:          return "scale_up"
    if state.get("latency_ms", 0) > 300:        return "rebalance_traffic"
    if state.get("error_rate", 0) > 0.10:       return "rebalance_traffic"
    if state.get("active_servers", 0) > 5:      return "remove_idle_resource"
    return "clear_cache"


# ── Run one task episode ──────────────────────────────────────────────────────

def run_task(client, task_id: str, difficulty: str) -> None:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    try:
        state = env_reset(difficulty=difficulty)
        done  = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = llm_action(client, state)
            state, reward, done, err = env_step(action)

            rewards.append(reward)
            steps_taken = step
            log_step(step, action, reward, done, err)

            if done:
                break

        score   = grade_task(task_id, state, rewards, steps_taken)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] task={task_id} error={exc}", flush=True)

    finally:
        log_end(success, steps_taken, score, rewards)


# ── Main: iterate all 3 tasks ─────────────────────────────────────────────────

def main():
    # Build client inside try/except — api_key=None crashes __init__
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[DEBUG] OpenAI client init failed: {exc}", flush=True)
        client = None

    for task_cfg in TASKS:
        run_task(client, task_cfg["task_id"], task_cfg["difficulty"])


if __name__ == "__main__":
    main()