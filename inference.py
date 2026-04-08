"""
inference.py — CloudOps-RL Baseline Agent
Compliant with OpenEnv [START]/[STEP]/[END] stdout format.
"""

import os
import json
import requests
import textwrap
from typing import List, Optional
from openai import OpenAI

# ── Mandatory hackathon variables ──────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy_key_to_prevent_crash"

# ── Environment connection ─────────────────────────────────────────────────────
ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:7860").rstrip("/")
LOCAL_ENV = None

# LLM controls
LLM_ENABLED = API_KEY != "dummy_key_to_prevent_crash"
LLM_ERROR_PRINTED = False

# ── Episode config ─────────────────────────────────────────────────────────────
TASK_NAME   = os.getenv("CLOUDOPS_TASK", "incident_response")
BENCHMARK   = os.getenv("CLOUDOPS_BENCHMARK", "cloudops_rl")
MAX_STEPS   = 10
SUCCESS_SCORE_THRESHOLD = 0.3   # normalized score in [0, 1]

VALID_ACTIONS = [
    "scale_up",
    "scale_down",
    "restart_database",
    "rebalance_traffic",
    "clear_cache",
    "remove_idle_resource",
    "noop",
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SRE agent managing a cloud infrastructure environment.
    You will receive the current system telemetry (CPU, memory, latency, error rate, etc.)
    and must choose the single best remediation action.

    Reply with EXACTLY ONE action from this list — nothing else, no explanation:
    scale_up
    scale_down
    restart_database
    rebalance_traffic
    clear_cache
    remove_idle_resource
    noop
""").strip()

# ── Stdout loggers — exact spec format ────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── Env helpers ────────────────────────────────────────────────────────────────
def env_reset() -> dict:
    global LOCAL_ENV
    try:
        resp = requests.post(f"{ENV_URL}/reset", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("state", data)
    except requests.exceptions.RequestException as exc:
        print(f"[DEBUG] Env HTTP unavailable, falling back to local CloudOpsEnv: {exc}", flush=True)
        try:
            from env import CloudOpsEnv
            LOCAL_ENV = CloudOpsEnv()
            state_obj = LOCAL_ENV.reset()
            return state_to_dict(state_obj)
        except Exception as exc2:
            print(f"[DEBUG] Local env fallback failed: {exc2}", flush=True)
            raise

def env_step(action: str) -> tuple:
    global LOCAL_ENV
    if LOCAL_ENV is not None:
        try:
            state_obj, reward, done = LOCAL_ENV.step(action)
            state = state_to_dict(state_obj)
            return state, float(reward), bool(done), None
        except Exception as exc:
            return {}, 0.0, True, str(exc)

    try:
        resp = requests.post(
            f"{ENV_URL}/step",
            json={"action": action},
            timeout=30,
        )
        resp.raise_for_status()
        data    = resp.json()
        state   = data.get("state", {})
        reward  = float(data.get("reward", 0.0))
        done    = bool(data.get("done", False))
        error   = data.get("error") or data.get("last_action_error") or None
        return state, reward, done, error
    except requests.exceptions.RequestException as exc:
        print(f"[DEBUG] Env HTTP step failed, switching to local CloudOpsEnv: {exc}", flush=True)
        try:
            from env import CloudOpsEnv
            LOCAL_ENV = CloudOpsEnv()
            LOCAL_ENV.reset()
            state_obj, reward, done = LOCAL_ENV.step(action)
            state = state_to_dict(state_obj)
            return state, float(reward), bool(done), None
        except Exception as exc2:
            return {}, 0.0, True, str(exc2)

def state_to_dict(state_obj):
    if hasattr(state_obj, "model_dump"):
        try:
            return state_obj.model_dump()
        except Exception:
            pass
    if hasattr(state_obj, "dict"):
        try:
            return state_obj.dict()
        except Exception:
            pass
    return state_obj

# ── LLM decision ──────────────────────────────────────────────────────────────
def get_action(client: OpenAI, state: dict, step: int, history: List[str]) -> str:
    cpu = state.get("cpu_usage", 0)
    latency = state.get("latency_ms", 0)
    error_rate = state.get("error_rate", 0)
    db_health = state.get("db_health", "healthy")
    traffic = state.get("traffic_load", "normal")
    servers = state.get("active_servers", 3)

    # RULE-BASED CRITICAL REASONING (Bulletproof heuristics)
    if db_health == "degraded":
        return "restart_database"
    if cpu >= 80:
        return "scale_up"
    if error_rate >= 0.10:
        return "rebalance_traffic"
    if traffic == "high" and latency >= 300:
        return "rebalance_traffic"
    if latency > 800:
        return "scale_up"
    if traffic == "low" and servers > 3 and cpu < 30:
        return "remove_idle_resource"

    # Fallback to LLM strategic reasoning
    history_block = "\n".join(history[-4:]) if history else "None"
    user_content = f"""
    Step: {step}
    Current state:
    {json.dumps(state, indent=2)}

    History:
    {history_block}

    Choose the best action.
    """

    if client is not None and LLM_ENABLED:
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=20,
            )
            raw = (completion.choices[0].message.content or "").strip().lower()
            for action in VALID_ACTIONS:
                if action in raw:
                    return action
        except Exception as exc:
            global LLM_ERROR_PRINTED
            if not LLM_ERROR_PRINTED:
                print(f"[DEBUG] LLM error: {exc}", flush=True)
                LLM_ERROR_PRINTED = True

    if latency > 300:
        return "rebalance_traffic"
    if cpu > 75:
        return "scale_up"
    return "clear_cache"

# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        client = None
        if LLM_ENABLED:
            try:
                client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            except Exception as exc:
                print(f"[DEBUG] LLM client init failed, disabling LLM: {exc}", flush=True)
                client = None
        
        state = env_reset()
        done  = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = get_action(client, state, step, history)

            try:
                state, reward, done, step_error = env_step(action)
            except Exception as exc:
                step_error = str(exc)
                reward     = 0.0
                done       = True

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=step_error)
            history.append(f"Step {step}: {action} → reward {reward:+.2f}")

        # SRE GRADING RUBRIC: 50% Survival, 50% Optimization
        # If the agent reached max steps without a fatal crash penalty, it survived.
        survived = (steps_taken == MAX_STEPS and rewards[-1] > -2.0)
        survival_score = 0.5 if survived else 0.0
        
        # Calculate optimization efficiency
        avg_reward = sum(rewards) / max(len(rewards), 1)
        optimization_score = max(avg_reward / 2.0, 0.0)
        
        score = min(survival_score + optimization_score, 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Fatal error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()