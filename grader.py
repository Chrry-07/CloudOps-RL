"""
CloudOps RL — Task Graders
Each grader receives the final CloudState dict + episode rewards list
and returns a deterministic score in [0.0, 1.0].

Tasks:
  idle_resource_leak  (easy)   — cut cost & servers without crashing
  traffic_spike       (medium) — reduce CPU/latency under high traffic
  database_failure    (hard)   — recover from degraded DB + high error rate
"""

from typing import Any, Dict, List


# ── Individual graders ────────────────────────────────────────────────────────

def grade_idle_resource_leak(
    final_state: Dict[str, Any],
    rewards: List[float],
    steps: int,
) -> float:
    """
    Easy task: The agent must remove idle resources.
    Success criteria:
      - active_servers <= 4  (was 8 at start)
      - cost_per_hour <= 120 (was 200 at start)
      - cpu_usage < 80       (not crashed from over-scaling down)
    Score components (each 0-1, weighted):
      40% — server reduction
      40% — cost reduction
      20% — stability (cpu < 80 and no crash)
    """
    initial_servers = 8
    initial_cost = 200.0
    target_servers = 4
    target_cost = 100.0

    servers = final_state.get("active_servers", initial_servers)
    cost = final_state.get("cost_per_hour", initial_cost)
    cpu = final_state.get("cpu_usage", 100.0)

    # Server reduction score: 0 if no change, 1 if at/below target
    server_score = max(
        0.0,
        min(1.0, (initial_servers - servers) / (initial_servers - target_servers)),
    )

    # Cost reduction score
    cost_score = max(
        0.0,
        min(1.0, (initial_cost - cost) / (initial_cost - target_cost)),
    )

    # Stability: cpu must stay below 80
    stability = 1.0 if cpu < 80.0 else max(0.0, 1.0 - (cpu - 80.0) / 20.0)

    score = 0.40 * server_score + 0.40 * cost_score + 0.20 * stability
    return round(float(score), 4)


def grade_traffic_spike(
    final_state: Dict[str, Any],
    rewards: List[float],
    steps: int,
) -> float:
    """
    Medium task: Respond to a traffic spike (cpu=92, latency=450ms).
    Success criteria:
      - cpu_usage < 75
      - latency_ms < 250
      - error_rate < 0.05
    Score components:
      40% — CPU normalised
      35% — latency normalised
      25% — error rate normalised
    """
    cpu = final_state.get("cpu_usage", 92.0)
    latency = final_state.get("latency_ms", 450.0)
    error_rate = final_state.get("error_rate", 0.05)

    # Lower is better; start values are 92, 450, 0.05
    cpu_score = max(0.0, min(1.0, (92.0 - cpu) / (92.0 - 50.0)))
    latency_score = max(0.0, min(1.0, (450.0 - latency) / (450.0 - 100.0)))
    error_score = max(0.0, min(1.0, (0.10 - error_rate) / 0.10))

    score = 0.40 * cpu_score + 0.35 * latency_score + 0.25 * error_score
    return round(float(score), 4)


def grade_database_failure(
    final_state: Dict[str, Any],
    rewards: List[float],
    steps: int,
) -> float:
    """
    Hard task: Recover from degraded DB (error_rate=0.15, latency=800ms).
    Success criteria:
      - db_health == "healthy"  (mandatory — 0 score if not met)
      - error_rate < 0.05
      - latency_ms < 300
    Score components:
      50% — DB recovery (binary)
      30% — error rate reduction
      20% — latency reduction
    """
    db_health = final_state.get("db_health", "degraded")
    error_rate = final_state.get("error_rate", 0.15)
    latency = final_state.get("latency_ms", 800.0)

    db_score = 1.0 if db_health == "healthy" else 0.0

    # Without DB recovery the agent can earn at most ~0.5
    error_score = max(0.0, min(1.0, (0.15 - error_rate) / 0.15))
    latency_score = max(0.0, min(1.0, (800.0 - latency) / (800.0 - 100.0)))

    score = 0.50 * db_score + 0.30 * error_score + 0.20 * latency_score
    return round(float(score), 4)


# ── Registry ──────────────────────────────────────────────────────────────────

GRADERS = {
    "idle_resource_leak": grade_idle_resource_leak,
    "traffic_spike":      grade_traffic_spike,
    "database_failure":   grade_database_failure,
}


def grade(
    task_id: str,
    final_state: Dict[str, Any],
    rewards: List[float],
    steps: int,
) -> float:
    """
    Dispatch to the correct grader. Returns score in [0.0, 1.0].
    Falls back to reward-based scoring if task_id is unknown.
    """
    fn = GRADERS.get(task_id)
    if fn is None:
        # Generic fallback: normalise average reward to [0,1]
        avg = sum(rewards) / max(len(rewards), 1)
        return round(float(min(max((avg + 1.0) / 2.0, 0.0), 1.0)), 4)
    return fn(final_state, rewards, steps)