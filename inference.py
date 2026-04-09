"""
CloudOps Incident Response - Inference Script
=============================================
Uses OpenAI-compatible client routed through the evaluator's LiteLLM proxy.
Env vars injected by grader: API_BASE_URL, API_KEY, MODEL_NAME
"""

import os
import json
import requests
from typing import List, Optional

# ── Env vars ──────────────────────────────────────────────────────────────────
# ALWAYS use os.getenv() (never os.environ[]) so missing keys don't crash.
# api_key must never be None — OpenAI client validates it at __init__ time.
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "no-key-set"
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
ENV_URL      = os.getenv("ENV_URL", "http://127.0.0.1:7860").rstrip("/")

TASK_NAME  = "incident_response"
BENCHMARK  = "cloudops_rl"
MAX_STEPS  = 10
SUCCESS_SCORE_THRESHOLD = 0.30

VALID_ACTIONS = [
    "scale_up",
    "scale_down",
    "restart_database",
    "rebalance_traffic",
    "clear_cache",
    "remove_idle_resource",
    "noop",
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


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Env helpers ───────────────────────────────────────────────────────────────

_local_env = None   # fallback if HTTP env not available


def _state_to_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def env_reset():
    global _local_env
    try:
        r = requests.post(f"{ENV_URL}/reset", timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("state", data)
    except Exception:
        try:
            from env import CloudOpsEnv
            _local_env = CloudOpsEnv()
            return _state_to_dict(_local_env.reset())
        except Exception as exc:
            print(f"[DEBUG] env_reset fallback failed: {exc}", flush=True)
            return {}


def env_step(action: str):
    global _local_env
    try:
        r = requests.post(
            f"{ENV_URL}/step", json={"action": action}, timeout=20
        )
        r.raise_for_status()
        data = r.json()
        return (
            data.get("state", {}),
            float(data.get("reward", 0.0)),
            bool(data.get("done", False)),
            data.get("error"),
        )
    except Exception:
        try:
            if _local_env is None:
                from env import CloudOpsEnv
                _local_env = CloudOpsEnv()
                _local_env.reset()
            state, reward, done = _local_env.step(action)
            return _state_to_dict(state), float(reward), bool(done), None
        except Exception as exc:
            print(f"[DEBUG] env_step fallback failed: {exc}", flush=True)
            return {}, 0.0, True, str(exc)


# ── LLM action ────────────────────────────────────────────────────────────────

def llm_action(client, state: dict, model_name: str) -> str:
    """Ask the LLM which action to take; fall back to heuristic on error."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(state, default=str)},
            ],
            temperature=0.1,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip().lower()
        for act in VALID_ACTIONS:
            if act in raw:
                return act
        return "noop"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return _heuristic(state)


def _heuristic(state: dict) -> str:
    """Simple rule-based fallback so episodes still complete."""
    if state.get("db_health") == "degraded":
        return "restart_database"
    if state.get("cpu_usage", 0) > 80:
        return "scale_up"
    if state.get("latency_ms", 0) > 300:
        return "rebalance_traffic"
    if state.get("error_rate", 0) > 0.10:
        return "rebalance_traffic"
    return "clear_cache"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Build the OpenAI client inside main() inside a try/except.
    # This is the most important fix — the client __init__ can raise if
    # base_url or api_key are invalid, and that must NOT be a top-level crash.
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        # If we truly cannot build a client, log and exit gracefully
        print(f"[DEBUG] OpenAI client init failed: {exc}", flush=True)
        log_start(TASK_NAME, BENCHMARK, MODEL_NAME)
        log_end(False, 0, 0.0, [])
        return

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    rewards: List[float] = []
    score   = 0.0
    success = False
    steps_taken = 0

    try:
        state = env_reset()
        done  = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = llm_action(client, state, MODEL_NAME)
            state, reward, done, err = env_step(action)

            rewards.append(reward)
            steps_taken = step

            log_step(step, action, reward, done, err)

            if done:
                break

        avg_reward = sum(rewards) / max(len(rewards), 1)
        score   = min(max((avg_reward + 1.0) / 2.0, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] fatal={exc}", flush=True)

    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()