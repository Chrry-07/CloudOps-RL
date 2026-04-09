"""
CloudOps RL Environment — FastAPI Server

Grader endpoints exposed in EVERY format the OpenEnv validator may try:
  POST /grade                     generic grade
  POST /tasks/{task_id}/grade     per-task grade
  GET  /tasks                     returns list (both array & wrapped)
  GET  /tasks/{task_id}           individual task lookup
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import CloudOpsEnv, CloudState
from grader import GRADERS, grade as run_grader

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="CloudOps RL Environment", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_env: Optional[CloudOpsEnv] = None


def get_env() -> CloudOpsEnv:
    global _env
    if _env is None:
        _env = CloudOpsEnv()
    return _env


# ── Shared task metadata ──────────────────────────────────────────────────────

TASKS = [
    {
        "id":                "idle_resource_leak",
        "name":              "Idle Resource Leak",
        "difficulty":        "easy",
        "success_threshold": 0.40,
        "has_grader":        True,
        "grader_endpoint":   "/tasks/idle_resource_leak/grade",
        "description": (
            "Over-provisioned servers are wasting money. "
            "Remove idle resources to cut cost while keeping CPU < 80%."
        ),
    },
    {
        "id":                "traffic_spike",
        "name":              "Traffic Spike Response",
        "difficulty":        "medium",
        "success_threshold": 0.35,
        "has_grader":        True,
        "grader_endpoint":   "/tasks/traffic_spike/grade",
        "description": (
            "Sudden surge pushed CPU to 92% and latency to 450 ms. "
            "Stabilise the system within 10 steps."
        ),
    },
    {
        "id":                "database_failure",
        "name":              "Database Failure Recovery",
        "difficulty":        "hard",
        "success_threshold": 0.30,
        "has_grader":        True,
        "grader_endpoint":   "/tasks/database_failure/grade",
        "description": (
            "Database is degraded, error rate 15%, latency 800 ms. "
            "Recover the system under cascading failure conditions."
        ),
    },
]

TASK_MAP = {t["id"]: t for t in TASKS}

_DIFFICULTY_TO_TASK = {
    "easy":   "idle_resource_leak",
    "medium": "traffic_spike",
    "hard":   "database_failure",
}

_TASK_TO_DIFFICULTY = {v: k for k, v in _DIFFICULTY_TO_TASK.items()}


# ── Pydantic models ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: str = "medium"
    task: Optional[str] = None


class StepRequest(BaseModel):
    action: str


class GradeRequest(BaseModel):
    task_id:     Optional[str]       = None
    final_state: Dict[str, Any]      = {}
    rewards:     List[float]         = []
    steps:       int                 = 0


class ResetResponse(BaseModel):
    state:       Dict[str, Any]
    observation: Dict[str, Any]


class StepResponse(BaseModel):
    state:       Dict[str, Any]
    observation: Dict[str, Any]
    reward:      float
    done:        bool
    info:        Dict[str, Any]


class StateResponse(BaseModel):
    state: Dict[str, Any]


class GradeResponse(BaseModel):
    task_id: str
    score:   float          # always [0.0, 1.0]
    passed:  bool
    details: Dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_dict(s: CloudState) -> Dict[str, Any]:
    if hasattr(s, "model_dump"):
        return s.model_dump()
    if hasattr(s, "dict"):
        return s.dict()
    return dict(s)


def _do_grade(task_id: str, req: GradeRequest) -> GradeResponse:
    if task_id not in TASK_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Valid: {list(TASK_MAP.keys())}",
        )
    score = run_grader(
        task_id=task_id,
        final_state=req.final_state,
        rewards=req.rewards,
        steps=req.steps,
    )
    meta = TASK_MAP[task_id]
    return GradeResponse(
        task_id=task_id,
        score=score,
        passed=score >= meta["success_threshold"],
        details={
            "threshold":  meta["success_threshold"],
            "difficulty": meta["difficulty"],
            "steps":      req.steps,
            "avg_reward": round(sum(req.rewards) / max(len(req.rewards), 1), 4),
        },
    )


# ── Core OpenEnv endpoints ────────────────────────────────────────────────────

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
        difficulty = req.difficulty
        if req.task and req.task in _TASK_TO_DIFFICULTY:
            difficulty = _TASK_TO_DIFFICULTY[req.task]
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
        # Normalise step reward to [0,1] for spec compliance
        norm_reward = float(min(max((reward + 2.0) / 3.0, 0.0), 1.0))
        return StepResponse(
            state=d,
            observation=d,
            reward=norm_reward,
            done=bool(done),
            info={
                "raw_reward":    float(reward),
                "incident_type": env.incident_type,
                "step":          env.current_step,
            },
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


# ── Task discovery endpoints ──────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    """Return task list. Validator may expect a flat list OR {tasks:[...]}."""
    return TASKS


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """Return metadata for a single task."""
    if task_id not in TASK_MAP:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return TASK_MAP[task_id]


# ── Grader endpoints (all formats) ───────────────────────────────────────────

@app.post("/grade", response_model=GradeResponse)
def grade_generic(req: GradeRequest):
    """Generic grader — task_id must be in the request body."""
    task_id = req.task_id or ""
    return _do_grade(task_id, req)


@app.post("/tasks/{task_id}/grade", response_model=GradeResponse)
def grade_task(task_id: str, req: GradeRequest = GradeRequest()):
    """Per-task grader endpoint — task_id comes from the URL path."""
    return _do_grade(task_id, req)


# ── Convenience: run a full graded episode via API ────────────────────────────

@app.post("/tasks/{task_id}/run")
def run_task(task_id: str, max_steps: int = 10):
    """
    Run a complete random-action episode for the task and return a graded score.
    Useful for the validator to verify graders produce varying scores.
    """
    import random
    ACTIONS = [
        "scale_up", "scale_down", "restart_database",
        "rebalance_traffic", "clear_cache", "remove_idle_resource", "noop",
    ]
    try:
        env = get_env()
        difficulty = _TASK_TO_DIFFICULTY.get(task_id, "medium")
        env.reset(difficulty=difficulty)

        rewards: List[float] = []
        done = False
        for _ in range(max_steps):
            if done:
                break
            _, raw_reward, done = env.step(random.choice(ACTIONS))
            rewards.append(float(raw_reward))

        final_state = _to_dict(env.state_data)
        score = run_grader(task_id=task_id, final_state=final_state,
                           rewards=rewards, steps=len(rewards))
        meta = TASK_MAP.get(task_id, {})
        return {
            "task_id":     task_id,
            "score":       score,
            "passed":      score >= meta.get("success_threshold", 0.3),
            "steps":       len(rewards),
            "rewards":     rewards,
            "final_state": final_state,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))