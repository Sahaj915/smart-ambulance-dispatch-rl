import os
import sys
import json
import time
import numpy as np
from typing import Optional
 
# ── Required environment variables (hackathon checklist) ─────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL",  "https://api.openai.com/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",    "gpt-4o-mini")
HF_TOKEN         = os.getenv("HF_TOKEN",      None)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", None)   # optional: for from_docker_image()
 
# ── OpenAI-compatible client (all LLM calls go through this) ─────────────────
from openai import OpenAI
 
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or os.getenv("OPENAI_API_KEY", "dummy-key"),
)
 
# ── FastAPI server ────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
 
from src.env import AmbulanceDispatchEnv
 
app = FastAPI(
    title="AmbulanceDispatchEnv — OpenEnv Server",
    description="Smart Ambulance Dispatch RL Environment API",
    version="1.0.0",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ── Global environment state ──────────────────────────────────────────────────
_env: Optional[AmbulanceDispatchEnv] = None
_current_obs: Optional[np.ndarray]  = None
_episode_step: int = 0
_total_reward: float = 0.0
_episode_id: int = 0
 
 
# ── Pydantic models ───────────────────────────────────────────────────────────
 
class ResetRequest(BaseModel):
    task: str  = "medium"
    seed: int  = 42
 
class StepRequest(BaseModel):
    action: Optional[int] = None   # if None, LLM policy decides
 
class ResetResponse(BaseModel):
    observation: list
    info: dict
    task: str
    action_space_n: int
    observation_shape: list
 
class StepResponse(BaseModel):
    observation: list
    reward: float
    terminated: bool
    truncated: bool
    done: bool
    info: dict
    action_taken: int
    step: int
    total_reward: float
 
 
# ── LLM Policy ───────────────────────────────────────────────────────────────
 
def llm_policy(obs: np.ndarray, env: AmbulanceDispatchEnv) -> int:
    """
    Use the LLM (via OpenAI-compatible client) to select an action.
    Falls back to greedy heuristic if LLM call fails.
    """
    state = env.state()
 
    # Build a compact prompt describing the current state
    active_calls = state.get("active_calls", [])
    hospitals    = state.get("hospitals", [])
    ambulances   = state.get("ambulances", [])
 
    calls_str = "\n".join([
        f"  Call #{c['id']}: severity={c['severity']}, wait={c['wait']}s, "
        f"pos={c['position']}, survival={c['survival']}"
        for c in active_calls
    ]) or "  None"
 
    hosp_str = "\n".join([
        f"  Hospital #{h['id']} ({h['name']}): "
        f"beds={h['beds']}, icu={h['icu']}, specialty={h['specialty']}"
        for h in hospitals
    ])
 
    amb_str = "\n".join([
        f"  Ambulance #{a['id']}: status={a['status']}, "
        f"pos={a['position']}, dispatches={a['dispatches']}"
        for a in ambulances
    ])
 
    n_amb = env.N_AMB
    n_hos = env.N_HOS
    wait_action = env.n_actions - 1
 
    prompt = f"""You are an AI emergency dispatcher. Choose the best action.
 
CURRENT STATE:
Active emergency calls:
{calls_str}
 
Ambulances:
{amb_str}
 
Hospitals:
{hosp_str}
 
ACTION SPACE (n={env.n_actions}):
- Actions 0 to {wait_action - 1}: dispatch ambulance A to best call, route to hospital H
  Action = ambulance_id * {n_hos} + hospital_id
  (ambulance_id: 0-{n_amb-1}, hospital_id: 0-{n_hos-1})
- Action {wait_action}: WAIT
 
RULES:
1. Prioritize critical (severity=critical) patients above all else
2. Route critical patients to trauma centers (specialty=trauma) when available
3. Only dispatch available ambulances (status=available)
4. Check hospital bed/ICU availability before routing
 
Respond with ONLY a single integer — the action number. Nothing else.
"""
 
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        action_str = response.choices[0].message.content.strip()
        action = int(action_str)
        if 0 <= action < env.n_actions:
            return action
    except Exception as e:
        _log_step(f"LLM policy error: {e} — falling back to greedy")
 
    return _greedy_action(env)
 
 
def _greedy_action(env: AmbulanceDispatchEnv) -> int:
    """Rule-based greedy fallback policy."""
    candidates = [c for c in env.calls if c.active and not c.assigned]
    if not candidates:
        return env.n_actions - 1  # wait
 
    candidates.sort(key=lambda c: (c.severity, c.wait_time), reverse=True)
    target = candidates[0]
 
    avail_ambs = [a for a in env.ambulances if a.is_available()]
    if not avail_ambs:
        return env.n_actions - 1  # wait
 
    import math
    nearest_amb = min(avail_ambs, key=lambda a: a.distance_to(target.x, target.y))
 
    def hospital_score(h):
        if not h.has_capacity(target.severity):
            return -1e9
        spec_bonus = 3.0 if (target.severity == 3 and h.specialty == 1) else 0.0
        dist_penalty = h.distance_to(target.x, target.y) * 0.5
        return h.bed_ratio() * 10 + spec_bonus - dist_penalty
 
    best_hosp = max(env.hospitals, key=hospital_score)
    return nearest_amb.id * env.N_HOS + best_hosp.id
 
 
# ── Structured Logging (START / STEP / END) ───────────────────────────────────
 
def _log_start(episode_id: int, task: str, seed: int):
    print(json.dumps({
        "type":       "START",
        "episode_id": episode_id,
        "task":       task,
        "seed":       seed,
        "timestamp":  time.time(),
        "model":      MODEL_NAME,
        "api_base":   API_BASE_URL,
    }), flush=True)
 
def _log_step(message: str, **kwargs):
    print(json.dumps({
        "type":      "STEP",
        "message":   message,
        "timestamp": time.time(),
        **kwargs,
    }), flush=True)
 
def _log_end(episode_id: int, total_reward: float, steps: int, stats: dict):
    print(json.dumps({
        "type":         "END",
        "episode_id":   episode_id,
        "total_reward": total_reward,
        "steps":        steps,
        "stats":        stats,
        "timestamp":    time.time(),
    }), flush=True)
 
 
# ── API Endpoints ─────────────────────────────────────────────────────────────
 
@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest):
    """Reset the environment. Called by OpenEnv checker to start an episode."""
    global _env, _current_obs, _episode_step, _total_reward, _episode_id
 
    # Validate task
    valid_tasks = ["easy", "medium", "hard"]
    task = request.task if request.task in valid_tasks else "medium"
 
    # Create/reset environment
    _env = AmbulanceDispatchEnv(task=task)
    obs, info = _env.reset(seed=request.seed)
 
    _current_obs  = obs
    _episode_step = 0
    _total_reward = 0.0
    _episode_id  += 1
 
    # Structured log: START
    _log_start(_episode_id, task, request.seed)
 
    return ResetResponse(
        observation=obs.tolist(),
        info=info,
        task=task,
        action_space_n=_env.n_actions,
        observation_shape=list(obs.shape),
    )
 
 
@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """Take a step. Called by OpenEnv checker repeatedly until done."""
    global _env, _current_obs, _episode_step, _total_reward
 
    if _env is None or _current_obs is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
 
    # Determine action
    if request.action is not None:
        action = request.action
    else:
        # Use LLM policy
        action = llm_policy(_current_obs, _env)
 
    # Execute step
    obs, reward, terminated, truncated, info = _env.step(action)
    _current_obs   = obs
    _episode_step += 1
    _total_reward += reward
    done = terminated or truncated
 
    # Structured log: STEP
    _log_step(
        f"step={_episode_step} action={action} reward={reward:.2f} done={done}",
        step=_episode_step,
        action=action,
        reward=round(reward, 4),
        total_reward=round(_total_reward, 4),
        done=done,
        pending_calls=info.get("pending_calls", 0),
        available_ambs=info.get("available_ambs", 0),
    )
 
    # Structured log: END (when episode finishes)
    if done:
        stats = info.get("episode_stats", {})
        _log_end(_episode_id, _total_reward, _episode_step, stats)
 
    return StepResponse(
        observation=obs.tolist(),
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        done=bool(done),
        info=info,
        action_taken=action,
        step=_episode_step,
        total_reward=round(_total_reward, 4),
    )
 
 
@app.get("/")
def root():
    """Health check / environment info."""
    return {
        "name":        "AmbulanceDispatchEnv",
        "version":     "1.0.0",
        "status":      "running",
        "model":       MODEL_NAME,
        "api_base":    API_BASE_URL,
        "tasks":       ["easy", "medium", "hard"],
        "endpoints":   ["/reset", "/step", "/health", "/state"],
    }
 
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}
 
@app.get("/state")
def get_state():
    """Return human-readable current environment state."""
    if _env is None:
        return {"error": "No active episode. Call /reset first."}
    return _env.state()
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(json.dumps({
        "type":      "START",
        "message":   "AmbulanceDispatchEnv server starting",
        "port":      port,
        "model":     MODEL_NAME,
        "api_base":  API_BASE_URL,
        "timestamp": time.time(),
    }), flush=True)
 
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
 