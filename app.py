"""
CloudOps RL Environment — FastAPI Server
POST /reset  POST /step  GET /state  GET /tasks  POST /grade  GET /health
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import CloudOpsEnv, CloudState

app = FastAPI(title="CloudOps RL Environment", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

class TrajectoryStep(BaseModel):
    action: str
    reward: Optional[float] = None
    state: Optional[Dict[str, Any]] = None
    observation: Optional[Dict[str, Any]] = None
    done: Optional[bool] = None

class GradeRequest(BaseModel):
    task_id: str
    trajectory: List[TrajectoryStep] = []
    final_state: Optional[Dict[str, Any]] = None

class GradeResponse(BaseModel):
    task_id: str
    score: float
    success: bool
    feedback: str

# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_dict(s: CloudState) -> Dict[str, Any]:
    if hasattr(s, "model_dump"):
        return s.model_dump()
    if hasattr(s, "dict"):
        return s.dict()
    return dict(s)

# ── Graders ───────────────────────────────────────────────────────────────────

TASK_SUCCESS_THRESHOLD = {
    "idle_resource_leak": 0.40,
    "traffic_spike":      0.35,
    "database_failure":   0.30,
}

def _grade_idle_resource_leak(traj, final_state):
    actions = [s.action for s in traj]
    good = sum(1 for a in actions if a in ("remove_idle_resource", "scale_down"))
    action_score = min(good / max(len(actions), 1), 1.0)
    cost_score = server_score = 0.0
    if final_state:
        cost = final_state.get("cost_per_hour", 200.0)
        cost_score = max(0.0, min(1.0, (200.0 - cost) / 150.0))
        servers = final_state.get("active_servers", 8)
        server_score = 1.0 if servers <= 4 else max(0.0, (8 - servers) / 4.0)
    score = round(0.4 * action_score + 0.3 * cost_score + 0.3 * server_score, 4)
    return score, f"good_actions={good}/{len(actions)} cost={cost_score:.2f} servers={server_score:.2f}"

def _grade_traffic_spike(traj, final_state):
    rewards = [s.reward for s in traj if s.reward is not None]
    avg_r = sum(rewards) / max(len(rewards), 1)
    reward_score = min(max((avg_r + 1.0) / 2.0, 0.0), 1.0)
    cpu_score = latency_score = 0.0
    if final_state:
        cpu = final_state.get("cpu_usage", 100.0)
        cpu_score = 1.0 if cpu < 85 else max(0.0, (100.0 - cpu) / 15.0)
        lat = final_state.get("latency_ms", 450.0)
        latency_score = 1.0 if lat < 200 else max(0.0, (450.0 - lat) / 250.0)
    score = round(0.5 * reward_score + 0.25 * cpu_score + 0.25 * latency_score, 4)
    return score, f"avg_reward={avg_r:.2f} cpu={cpu_score:.2f} latency={latency_score:.2f}"

def _grade_database_failure(traj, final_state):
    actions = [s.action for s in traj]
    restart_idx = next((i for i, a in enumerate(actions) if a == "restart_database"), None)
    restart_score = 0.0 if restart_idx is None else max(0.0, 1.0 - restart_idx / max(len(actions), 1))
    db_score = error_score = 0.0
    if final_state:
        db_score = 1.0 if final_state.get("db_health") == "healthy" else 0.0
        err = final_state.get("error_rate", 0.5)
        error_score = 1.0 if err < 0.05 else max(0.0, (0.5 - err) / 0.45)
    score = round(0.4 * restart_score + 0.3 * db_score + 0.3 * error_score, 4)
    return score, f"restart_at={restart_idx} db={db_score:.2f} error={error_score:.2f}"

GRADERS = {
    "idle_resource_leak": _grade_idle_resource_leak,
    "traffic_spike":      _grade_traffic_spike,
    "database_failure":   _grade_database_failure,
}

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
        difficulty = req.difficulty
        if req.task == "idle_resource_leak":
            difficulty = "easy"
        elif req.task == "traffic_spike":
            difficulty = "medium"
        elif req.task == "database_failure":
            difficulty = "hard"
        d = _to_dict(env.reset(difficulty=difficulty))
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
            state=d, observation=d,
            reward=float(reward), done=bool(done),
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
    return {
        "tasks": [
            {
                "id": "idle_resource_leak",
                "name": "Idle Resource Leak",
                "difficulty": "easy",
                "description": "Over-provisioned servers waste money. Remove idle resources to cut cost.",
                "success_threshold": 0.40,
                "has_grader": True,
            },
            {
                "id": "traffic_spike",
                "name": "Traffic Spike Response",
                "difficulty": "medium",
                "description": "CPU at 92%, latency 450ms. Stabilise within 10 steps.",
                "success_threshold": 0.35,
                "has_grader": True,
            },
            {
                "id": "database_failure",
                "name": "Database Failure Recovery",
                "difficulty": "hard",
                "description": "DB degraded, error rate 15%, latency 800ms. Recover under cascading failure.",
                "success_threshold": 0.30,
                "has_grader": True,
            },
        ]
    }

@app.post("/grade", response_model=GradeResponse)
def grade(req: GradeRequest):
    """
    Deterministic grader for a completed episode.
    Accepts trajectory + optional final_state. Returns score in [0.0, 1.0].
    """
    if req.task_id not in GRADERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Valid: {list(GRADERS.keys())}",
        )
    try:
        final_state = req.final_state
        if final_state is None and req.trajectory:
            last = req.trajectory[-1]
            final_state = last.state or last.observation

        score, feedback = GRADERS[req.task_id](req.trajectory, final_state)
        score = round(min(max(score, 0.0), 1.0), 4)
        return GradeResponse(
            task_id=req.task_id,
            score=score,
            success=score >= TASK_SUCCESS_THRESHOLD[req.task_id],
            feedback=feedback,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))