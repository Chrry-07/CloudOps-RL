import random
from pydantic import BaseModel


class CloudState(BaseModel):
    cpu_usage: float
    memory_usage: float
    latency_ms: float
    error_rate: float
    traffic_load: str
    active_servers: int
    db_health: str
    cost_per_hour: float


class CloudOpsEnv:
    def __init__(self):
        self.state_data = None
        self.done = False
        self.current_step = 0
        self.max_steps = 10
        self.incident_type = None

    def reset(self, difficulty="medium"):
        self.current_step = 0
        self.done = False

        # Task difficulty progression
        if difficulty == "easy":
            self.incident_type = "idle_resource_leak"
        elif difficulty == "medium":
            self.incident_type = random.choice(
                ["traffic_spike", "latency_surge"]
            )
        else:  # hard
            self.incident_type = "database_failure"

        print(f"[ENV] Episode started. Incident Type: {self.incident_type}")

        self.state_data = CloudState(
            cpu_usage=45.0,
            memory_usage=50.0,
            latency_ms=120.0,
            error_rate=0.01,
            traffic_load="normal",
            active_servers=3,
            db_health="healthy",
            cost_per_hour=75.0
        )

        # Incident initialization
        if self.incident_type == "traffic_spike":
            self.state_data.traffic_load = "high"
            self.state_data.cpu_usage = 92.0
            self.state_data.latency_ms = 450.0

        elif self.incident_type == "database_failure":
            self.state_data.db_health = "degraded"
            self.state_data.error_rate = 0.15
            self.state_data.latency_ms = 800.0

        elif self.incident_type == "latency_surge":
            self.state_data.latency_ms = 1200.0
            self.state_data.error_rate = 0.05

        elif self.incident_type == "idle_resource_leak":
            self.state_data.traffic_load = "low"
            self.state_data.active_servers = 8
            self.state_data.cost_per_hour = 200.0
            self.state_data.cpu_usage = 15.0

        return self.state_data

    def step(self, action: str):
        if self.done:
            return self.state_data, 0.0, True

        self.current_step += 1

        prev_latency = self.state_data.latency_ms
        prev_error_rate = self.state_data.error_rate

        # Natural drift
        if self.state_data.traffic_load == "high":
            self.state_data.cpu_usage += 8.0
            self.state_data.latency_ms += 40.0
            self.state_data.error_rate += 0.03

        if self.state_data.db_health == "degraded":
            self.state_data.error_rate += 0.05
            self.state_data.latency_ms += 100.0

        # Action effects
        if action == "scale_up":
            self.state_data.active_servers += 1
            self.state_data.cpu_usage = max(
                0.0,
                self.state_data.cpu_usage - 20.0
            )
            self.state_data.latency_ms = max(
                50.0,
                self.state_data.latency_ms - 50.0
            )
            self.state_data.cost_per_hour += 30.0

        elif action in ["scale_down", "remove_idle_resource"]:
            if self.state_data.active_servers > 1:
                self.state_data.active_servers -= 1
                self.state_data.cpu_usage += 15.0
                self.state_data.cost_per_hour -= 25.0

        elif action == "rebalance_traffic":
            self.state_data.latency_ms = max(
                50.0,
                self.state_data.latency_ms - 60.0
            )
            self.state_data.error_rate = max(
                0.0,
                self.state_data.error_rate - 0.02
            )

        elif action == "restart_database":
            if self.state_data.db_health == "degraded":
                self.state_data.db_health = "healthy"
                self.state_data.error_rate = max(
                    0.0,
                    self.state_data.error_rate - 0.10
                )

        elif action == "clear_cache":
            self.state_data.memory_usage = max(
                20.0,
                self.state_data.memory_usage - 40.0
            )
            self.state_data.latency_ms = max(
                50.0,
                self.state_data.latency_ms - 20.0
            )

        # Failure states
        self.state_data.cpu_usage = min(
            100.0,
            self.state_data.cpu_usage
        )

        if (
            self.state_data.cpu_usage >= 100.0
            or self.state_data.error_rate >= 0.50
        ):
            self.done = True

        # Dynamic reward
        if self.done and self.state_data.cpu_usage >= 100.0:
            reward = -2.0
        else:
            uptime_score = (
                1.0
                if (
                    self.state_data.cpu_usage < 85.0
                    and self.state_data.db_health == "healthy"
                )
                else -1.0
            )

            latency_reduction = (
                prev_latency - self.state_data.latency_ms
            ) / 100.0

            error_reduction = (
                prev_error_rate - self.state_data.error_rate
            ) * 100.0

            if (
                self.state_data.traffic_load == "low"
                and self.state_data.active_servers > 3
            ):
                cost_efficiency = -1.0
            else:
                cost_efficiency = (
                    1.0
                    - (self.state_data.cost_per_hour / 200.0)
                )

            reward = (
                (0.4 * uptime_score)
                + (0.2 * latency_reduction)
                + (0.2 * error_reduction)
                + (0.2 * cost_efficiency)
            )

        if self.current_step >= self.max_steps:
            self.done = True

        return self.state_data, round(reward, 3), self.done