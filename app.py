"""
CloudOps RL Environment — FastAPI Server
OpenEnv spec endpoints:
  GET  /           health ping
  GET  /health     health ping
  POST /reset      start episode  → {state, observation}
  POST /step       advance step   → {state, observation, reward, done, info}
  GET  /state      current state  → {state}
  GET  /tasks      task list      → {tasks: [...]}
  POST /grade      score episode  → {task_id, score, passed}
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import CloudOpsEnv, CloudState
from grader import GRADERS, grade

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="CloudOps RL Environment", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[CloudOpsEnv] = None


def get_env() -> CloudOpsEnv:
    global _env
    if _env is None:
        _env = CloudOpsEnv()
    return _env


# ── Models ────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: str = "medium"
    task: Optional[str] = None


class StepRequest(BaseModel):
    action: str


class GradeRequest(BaseModel):
    task_id: str
    final_state: Dict[str, Any]
    rewards: List[float] = []
    steps: int = 0


class ResetResponse(BaseModel):
    state: Dict[str, Any]
    observation: Dict[str, Any]


class StepResponse(BaseModel):
    state: Dict[str, Any]
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    state: Dict[str, Any]


class GradeResponse(BaseModel):
    task_id: str
    score: float          # always in [0.0, 1.0]
    passed: bool
    details: Dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

_TASK_META = {
    "idle_resource_leak": {
        "name": "Idle Resource Leak",
        "difficulty": "easy",
        "success_threshold": 0.40,
        "description": (
            "Over-provisioned servers are wasting money. "
            "Remove idle resources to cut cost while keeping CPU < 80%."
        ),
    },
    "traffic_spike": {
        "name": "Traffic Spike Response",
        "difficulty": "medium",
        "success_threshold": 0.35,
        "description": (
            "Sudden surge has pushed CPU to 92% and latency to 450 ms. "
            "Stabilise the system within 10 steps."
        ),
    },
    "database_failure": {
        "name": "Database Failure Recovery",
        "difficulty": "hard",
        "success_threshold": 0.30,
        "description": (
            "Database is degraded, error rate 15%, latency 800 ms. "
            "Recover the system under cascading failure conditions."
        ),
    },
}

_DIFFICULTY_MAP = {
    "idle_resource_leak": "easy",
    "traffic_spike":      "medium",
    "database_failure":   "hard",
}


def _to_dict(s: CloudState) -> Dict[str, Any]:
    if hasattr(s, "model_dump"):
        return s.model_dump()
    if hasattr(s, "dict"):
        return s.dict()
    return dict(s)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "env": "cloudops_rl"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = ResetRequest()):
    try:
        env = get_env()
        # If a task_id is passed, map it to the right difficulty
        difficulty = req.difficulty
        if req.task and req.task in _DIFFICULTY_MAP:
            difficulty = _DIFFICULTY_MAP[req.task]
        state_obj = env.reset(difficulty=difficulty)
        d = _to_dict(state_obj)
        return ResetResponse(state=d, observation=d)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    try:
        env = get_env()
        if env.state_data is None:
            env.reset()
        state_obj, reward, done = env.step(req.action)
        d = _to_dict(state_obj)
        return StepResponse(
            state=d,
            observation=d,
            reward=float(reward),
            done=bool(done),
            info={"incident_type": env.incident_type, "step": env.current_step},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", response_model=StateResponse)
def state():
    try:
        env = get_env()
        if env.state_data is None:
            env.reset()
        return StateResponse(state=_to_dict(env.state_data))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tasks")
def tasks():
    """
    Return all tasks with their grader info.
    The validator enumerates this list and expects at least 3 entries,
    each with a grader it can invoke via POST /grade.
    """
    task_list = []
    for task_id, meta in _TASK_META.items():
        task_list.append(
            {
                "id":                task_id,
                "name":              meta["name"],
                "difficulty":        meta["difficulty"],
                "description":       meta["description"],
                "success_threshold": meta["success_threshold"],
                "grader":            f"/grade",   # all tasks share one endpoint
                "grader_endpoint":   "POST /grade",
            }
        )
    return {"tasks": task_list}


@app.post("/grade", response_model=GradeResponse)
def grade_episode(req: GradeRequest):
    """
    Grade a completed episode for a given task.
    Accepts the final environment state + list of rewards collected.
    Returns a score in [0.0, 1.0] and whether the task was passed.
    """
    try:
        if req.task_id not in _TASK_META:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task_id '{req.task_id}'. "
                       f"Valid: {list(_TASK_META.keys())}",
            )

        score = grade(
            task_id=req.task_id,
            final_state=req.final_state,
            rewards=req.rewards,
            steps=req.steps,
        )

        threshold = _TASK_META[req.task_id]["success_threshold"]
        passed = score >= threshold

        return GradeResponse(
            task_id=req.task_id,
            score=score,
            passed=passed,
            details={
                "threshold":  threshold,
                "steps":      req.steps,
                "avg_reward": round(sum(req.rewards) / max(len(req.rewards), 1), 4),
                "difficulty": _TASK_META[req.task_id]["difficulty"],
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))