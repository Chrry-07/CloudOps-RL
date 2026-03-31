from fastapi import FastAPI
from pydantic import BaseModel
from env import CloudOpsEnv
from models import AgentAction

# Initialize the app and the environment
app = FastAPI(title="CloudOps-RL")
cloud_env = CloudOpsEnv()

# --- ROOT ENDPOINT ---
@app.get("/")
def home():
    return {
        "project": "CloudOps-RL",
        "description": "AI Incident Response Simulator",
        "docs": "/docs",
        "endpoints": ["/reset", "/step", "/state"]
    }

# --- RESET ENDPOINT ---
@app.post("/reset")
def reset_env():
    state = cloud_env.reset()
    # Safely handle the state object returning as a dictionary
    return {"state": state.dict() if hasattr(state, "dict") else state}

# --- STEP ENDPOINT ---
@app.post("/step")
def step_env(request: AgentAction):
    state, reward, done = cloud_env.step(request.action)
    return {
        "state": state.dict() if hasattr(state, "dict") else state,
        "reward": reward,
        "done": done
    }

# --- STATE ENDPOINT ---
@app.get("/state")
def get_state():
    s = cloud_env.state_data
    if s is None:
        return {"state": {}}
    return {"state": s.dict() if hasattr(s, "dict") else s}