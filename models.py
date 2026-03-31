from pydantic import BaseModel
from typing import Literal


class CloudState(BaseModel):
    cpu_usage: int
    memory_usage: int
    latency_ms: int
    error_rate: float
    traffic_load: Literal["low", "medium", "high"]
    active_servers: int
    db_health: Literal["healthy", "degraded", "failed"]
    cost_per_hour: int


class AgentAction(BaseModel):
    action: Literal[
        "scale_up",
        "scale_down",
        "restart_service",
        "restart_database",
        "rebalance_traffic",
        "clear_cache",
        "remove_idle_resource",
        "noop"
    ]