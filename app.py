"""
CloudOps RL Environment — FastAPI Server
Implements the full OpenEnv spec:
  POST /reset          → ResetResponse
  POST /step           → StepResponse
  GET  /state          → StateResponse
  GET  /tasks          → list of task dicts
  GET  /health         → {"status": "ok"}
  GET  /               → {"status": "ok"}   (ping / root health)
"""

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import CloudOpsEnv, CloudState

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="CloudOps RL Environment", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared environment instance (stateful per session)
_env: Optional[CloudOpsEnv] = None


def get_env() -> CloudOpsEnv:
    global _env
    if _env is None:
        _env = CloudOpsEnv()
    return _env


# ── Request / Response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: str = "medium"   # "easy" | "medium" | "hard"
    task: Optional[str] = None   # reserved


class StepRequest(BaseModel):
    action: str


class ResetResponse(BaseModel):
    state: Dict[str, Any]
    observation: Dict[str, Any]  # same as state — some validators require both


class StepResponse(BaseModel):
    state: Dict[str, Any]
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    state: Dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_dict(s: CloudState) -> Dict[str, Any]:
    if hasattr(s, "model_dump"):
        return s.model_dump()
    if hasattr(s, "dict"):
        return s.dict()
    return dict(s)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Root health check — the validator pings this first."""
    return {"status": "ok", "env": "cloudops_rl"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest = ResetRequest()):
    """Reset the environment and return the initial state."""
    try:
        env = get_env()
        state_obj = env.reset(difficulty=req.difficulty)
        d = _to_dict(state_obj)
        return ResetResponse(state=d, observation=d)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """Take one action; return new state, reward, done."""
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
    """Return current state without advancing the episode."""
    try:
        env = get_env()
        if env.state_data is None:
            env.reset()
        return StateResponse(state=_to_dict(env.state_data))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tasks")
def tasks():
    """Enumerate supported tasks for the grader."""
    return {
        "tasks": [
            {
                "id": "idle_resource_leak",
                "name": "Idle Resource Leak",
                "difficulty": "easy",
                "description": (
                    "Over-provisioned servers are wasting money. "
                    "Remove idle resources to cut cost while keeping "
                    "CPU usage reasonable."
                ),
                "success_threshold": 0.40,
            },
            {
                "id": "traffic_spike",
                "name": "Traffic Spike Response",
                "difficulty": "medium",
                "description": (
                    "Sudden surge has spiked CPU to 92% and latency "
                    "to 450 ms. Stabilise within 10 steps."
                ),
                "success_threshold": 0.35,
            },
            {
                "id": "database_failure",
                "name": "Database Failure Recovery",
                "difficulty": "hard",
                "description": (
                    "Database is degraded, error rate 15%, latency 800 ms. "
                    "Recover the system under cascading failure."
                ),
                "success_threshold": 0.30,
            },
        ]
    }