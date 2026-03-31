# ☁️ CloudOps-RL  
## AI Incident Response Simulator for OpenEnv

CloudOps-RL is a **benchmark-grade OpenEnv environment** where an AI agent monitors cloud infrastructure metrics, diagnoses system incidents, and takes corrective actions to maximize **uptime, performance, and cost efficiency**.

This environment simulates real-world workflows performed by:
- DevOps Engineers
- Site Reliability Engineers (SRE)
- Cloud Infrastructure Teams

---

## 🎯 Problem Statement

Modern cloud systems constantly face issues such as:
- sudden traffic spikes
- overloaded servers
- high latency
- service crashes
- database failures
- unnecessary cloud resource cost

The AI agent must continuously observe system health and take intelligent actions in real time.

---

## 🧠 Objective

The agent learns to:
- detect infrastructure anomalies
- diagnose incidents
- scale resources
- restart failed services
- rebalance traffic
- optimize cost vs uptime trade-off

---

## 🧩 API Endpoints

The environment exposes the following OpenEnv-compatible endpoints:
* **`POST /reset`** - Triggers a random infrastructure incident and returns the initial state.
* **`POST /step`** - Accepts an agent action, calculates the physics/drift, and returns the new state and reward delta.
* **`GET /state`** - Returns the current telemetry of the cloud environment.

---

## 📈 The Dynamic Reward Engine

Unlike static environments, CloudOps-RL features a highly dynamic state engine. Actions do not just yield flat points; the reward is calculated as a **mathematical delta** between the previous state and the new state, forcing the agent to actually optimize the metrics.

The multi-factor reward formula evaluates:
1. **Uptime Score:** Penalizes the agent heavily (-5.0) if CPU hits 100% or error rates exceed 50%.
2. **Latency Reduction:** Rewards proportional drops in millisecond latency.
3. **Error Minimization:** Rewards the stabilization of degraded databases.
4. **Cost Efficiency:** Actively penalizes the agent for keeping idle servers running during low-traffic periods.

---

## 🚀 Quick Start & Testing

You can interact with the live environment directly through the auto-generated Swagger UI! 

1. Navigate to the live docs: [Interactive API Docs](https://chrry07-cloudops-rl.hf.space/docs)
2. Open the `POST /reset` endpoint and click **Try it out** -> **Execute** to trigger a random infrastructure incident (e.g., `traffic_spike` or `database_failure`).
3. Check the response body to see the degraded system state.
4. Use the `POST /step` endpoint to send an action (like `rebalance_traffic` or `scale_up`) and watch the latency and reward metrics update in real-time!
