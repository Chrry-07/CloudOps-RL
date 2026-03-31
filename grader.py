def grade(state):
    score = 0

    if state.error_rate < 0.05:
        score += 0.3

    if state.latency_ms < 200:
        score += 0.3

    if state.cost_per_hour < 120:
        score += 0.2

    if state.cpu_usage < 80:
        score += 0.2

    return round(score, 2)