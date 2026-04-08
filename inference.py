"""
inference.py — FINAL PHASE 2 COMPLIANT
CloudOps-RL baseline agent
"""

import os
import json
import requests
from typing import List, Optional
from openai import OpenAI

# =========================
# REQUIRED VALIDATOR VARIABLES
# =========================
API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://router.huggingface.co/v1"
).strip()

API_KEY = os.environ.get("API_KEY", "dummy").strip()

MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "Qwen/Qwen2.5-72B-Instruct"
)

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Fix malformed URL edge case
if API_BASE_URL and not API_BASE_URL.startswith("http"):
    API_BASE_URL = "http://" + API_BASE_URL

LOCAL_ENV = None
MAX_STEPS = 10
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

SYSTEM_PROMPT = """
You are an expert SRE incident response agent.

Choose EXACTLY one action from:
scale_up
scale_down
restart_database
rebalance_traffic
clear_cache
remove_idle_resource
noop

Return ONLY the action.
""".strip()


# =========================
# LOGGING
# =========================
def log_start(task: str, env: str, model: str):
    print(
        f"[START] task={task} env={env} model={model}",
        flush=True
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str]
):
    error_val = error if error else "null"

    print(
        f"[STEP] step={step} "
        f"action={action} "
        f"reward={reward:.2f} "
        f"done={str(done).lower()} "
        f"error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float]
):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} "
        f"score={score:.3f} "
        f"rewards={rewards_str}",
        flush=True,
    )


# =========================
# ENV HELPERS
# =========================
def state_to_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if hasattr(obj, "dict"):
        return obj.dict()

    return obj


def env_reset():
    global LOCAL_ENV

    env_url = os.environ.get(
        "ENV_URL",
        "http://127.0.0.1:7860"
    ).rstrip("/")

    try:
        r = requests.post(
            f"{env_url}/reset",
            timeout=20
        )
        r.raise_for_status()

        data = r.json()

        return data.get("state", data)

    except Exception:
        from env import CloudOpsEnv

        LOCAL_ENV = CloudOpsEnv()

        return state_to_dict(LOCAL_ENV.reset())


def env_step(action: str):
    global LOCAL_ENV

    env_url = os.environ.get(
        "ENV_URL",
        "http://127.0.0.1:7860"
    ).rstrip("/")

    try:
        r = requests.post(
            f"{env_url}/step",
            json={"action": action},
            timeout=20
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
        if LOCAL_ENV is None:
            from env import CloudOpsEnv

            LOCAL_ENV = CloudOpsEnv()
            LOCAL_ENV.reset()

        state, reward, done = LOCAL_ENV.step(action)

        return (
            state_to_dict(state),
            float(reward),
            bool(done),
            None
        )


# =========================
# LLM ACTION
# =========================
def llm_action(client, state, model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": json.dumps(state)
            },
        ],
        temperature=0.1,
        max_tokens=10,
    )

    raw = (
        response.choices[0].message.content or ""
    ).strip().lower()

    for action in VALID_ACTIONS:
        if action in raw:
            return action

    return "noop"


# =========================
# FALLBACK POLICY
# =========================
def fallback_policy(state):
    cpu = state.get("cpu_usage", 0)
    latency = state.get("latency_ms", 0)
    error_rate = state.get("error_rate", 0)
    db_health = state.get("db_health", "healthy")

    if db_health == "degraded":
        return "restart_database"

    if cpu > 80:
        return "scale_up"

    if latency > 300:
        return "rebalance_traffic"

    if error_rate > 0.10:
        return "rebalance_traffic"

    return "clear_cache"


# =========================
# MAIN
# =========================
def main():
    task_name = "incident_response"
    benchmark = "cloudops_rl"

    log_start(task_name, benchmark, MODEL_NAME)

    rewards = []
    score = 0.0
    success = False
    steps_taken = 0

    try:
        try:
            client = OpenAI(
                base_url=API_BASE_URL,
                api_key=API_KEY
            )

        except Exception as e:
            print(
                f"[DEBUG] OpenAI Init Failed: {e}",
                flush=True
            )
            client = None

        state = env_reset()
        done = False

        for step in range(1, MAX_STEPS + 1):
            try:
                if client is None:
                    raise ValueError(
                        "No LLM client available"
                    )

                action = llm_action(
                    client,
                    state,
                    MODEL_NAME
                )

            except Exception:
                action = fallback_policy(state)

            state, reward, done, err = env_step(action)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step,
                action,
                reward,
                done,
                err
            )

            if done:
                break

        avg_reward = sum(rewards) / max(len(rewards), 1)

        score = min(
            max((avg_reward + 1.0) / 2.0, 0.0),
            1.0
        )

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] fatal={e}", flush=True)

    finally:
        log_end(
            success,
            steps_taken,
            score,
            rewards
        )


if __name__ == "__main__":
    main()