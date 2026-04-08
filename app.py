from fastapi import FastAPI, Body
from src.env import AmbulanceDispatchEnv
import os
import time
import json
import numpy as np
import gradio as gr
import uvicorn
from dotenv import load_dotenv

load_dotenv()

api = FastAPI()
env_instance = None

try:
    from src.env import AmbulanceDispatchEnv, TASK_CONFIGS
    from src.grader import Grader, BENCHMARKS
except ImportError:
    AmbulanceDispatchEnv = None
    Grader = None

# ── Policy Loaders ────────────────────────────────────────────────────────────

def try_load_ppo(task: str):
    try:
        from stable_baselines3 import PPO
        path = f"models/ppo_{task}.zip"
        if os.path.exists(path):
            model = PPO.load(path)
            return lambda obs: int(model.predict(np.array(obs), deterministic=True)[0])
    except Exception:
        pass
    return None

PPO_POLICIES = {t: try_load_ppo(t) for t in ["easy", "medium", "hard"]}

def _make_greedy_policy():
    env_ref = [None]
    def _policy(obs):
        env = env_ref[0]
        if env is None: return 0
        candidates = [c for c in env.calls if c.active and not c.assigned]
        if not candidates: return env.n_actions - 1
        candidates.sort(key=lambda c: (c.severity, c.wait_time), reverse=True)
        target = candidates[0]
        avail_ambs = [a for a in env.ambulances if a.is_available()]
        if not avail_ambs: return env.n_actions - 1
        nearest_amb = min(avail_ambs, key=lambda a: a.distance_to(target.x, target.y))
        def hospital_score(h):
            if not h.has_capacity(target.severity): return -1e9
            spec_bonus = 3.0 if (target.severity == 3 and h.specialty == 1) else 0.0
            dist_penalty = h.distance_to(target.x, target.y) * 0.5
            return h.bed_ratio() * 10 + spec_bonus - dist_penalty
        best_hosp = max(env.hospitals, key=hospital_score)
        return nearest_amb.id * env.N_HOS + best_hosp.id
    return _policy, env_ref

# ── SVG Map Renderer ──────────────────────────────────────────────────────────

def render_real_world_map(env, step):
    W, H = 700, 700
    GRID = env.GRID
    PAD = 40
    scale = (W - 2 * PAD) / GRID
    def gx(x): return PAD + x * scale
    def gy(y): return H - PAD - y * scale

    lines = [
        f'<svg width="100%" height="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
        f'<defs><filter id="glow"><feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>'
        f'<feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>',
        f'<rect width="{W}" height="{H}" fill="#020617"/>',
    ]
    block_size = W // 10
    for i in range(12):
        for j in range(12):
            bx, by = i * block_size, j * block_size
            lines.append(f'<rect x="{bx+4}" y="{by+4}" width="{block_size-8}" height="{block_size-8}" fill="#0f172a" rx="2" opacity="0.5"/>')
    for i in range(0, int(GRID) + 1, 5):
        lines.append(f'<line x1="{gx(i)}" y1="{gy(0)}" x2="{gx(i)}" y2="{gy(GRID)}" stroke="#1e293b" stroke-width="3"/>')
        lines.append(f'<line x1="{gx(0)}" y1="{gy(i)}" x2="{gx(GRID)}" y2="{gy(i)}" stroke="#1e293b" stroke-width="3"/>')
    for h in env.hospitals:
        cx, cy = gx(h.x), gy(h.y)
        color = {0: "#6366f1", 1: "#ef4444", 2: "#06b6d4"}[h.specialty]
        lines.append(f'<rect x="{cx-15}" y="{cy-15}" width="30" height="30" fill="{color}" rx="6" filter="url(#glow)"/>')
        lines.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" fill="white" font-size="16" font-weight="bold">H</text>')
        lines.append(f'<text x="{cx}" y="{cy+38}" text-anchor="middle" fill="#94a3b8" font-size="10">{h.name[:10]}</text>')
    for c in env.calls:
        if c.active and not c.assigned:
            cx, cy = gx(c.x), gy(c.y)
            color = {1: "#f59e0b", 2: "#f97316", 3: "#ef4444"}[c.severity]
            pulse_r = 10 + (step % 10) * 2
            lines.append(f'<circle cx="{cx}" cy="{cy}" r="{pulse_r}" fill="none" stroke="{color}" stroke-width="2" opacity="{1-(step%10)/10}"/>')
            lines.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="{color}" filter="url(#glow)"/>')
            label = "CRITICAL" if c.severity == 3 else f"SEV {c.severity}"
            lines.append(f'<text x="{cx}" y="{cy-15}" text-anchor="middle" fill="{color}" font-size="11" font-weight="bold">{label}</text>')
    for a in env.ambulances:
        cx, cy = gx(a.x), gy(a.y)
        status_color = {0: "#22c55e", 1: "#eab308", 2: "#3b82f6", 3: "#64748b"}[a.status]
        lines.append(f'<rect x="{cx-10}" y="{cy-6}" width="20" height="12" rx="3" fill="{status_color}" filter="url(#glow)"/>')
        lines.append(f'<text x="{cx}" y="{cy+4}" text-anchor="middle" fill="white" font-size="8" font-weight="bold">{a.id}</text>')
    lines.append("</svg>")
    return "\n".join(lines)

# ── Dashboard Utilities ───────────────────────────────────────────────────────

def get_metric_card(title, value, trend="", variant="neutral"):
    color = {"neutral": "#334155", "success": "#065f46", "warning": "#854d0e", "danger": "#7f1d1d"}[variant]
    return f"""
    <div style="background:{color}; padding:12px; border-radius:10px; border:1px solid #475569; margin-bottom:8px">
        <div style="color:#cbd5e1; font-size:0.7rem; font-weight:bold; text-transform:uppercase">{title}</div>
        <div style="color:white; font-size:1.5rem; font-weight:bold">{value}</div>
        <div style="color:#94a3b8; font-size:0.7rem">{trend}</div>
    </div>"""

def run_episode_stream(task, policy_name, seed):
    env = AmbulanceDispatchEnv(task=task)
    if policy_name == "Greedy Heuristic":
        policy_fn, env_ref = _make_greedy_policy()
        env_ref[0] = env
    elif policy_name == "PPO (Trained)" and PPO_POLICIES.get(task):
        policy_fn = PPO_POLICIES[task]
    else:
        policy_fn = lambda obs: env.action_space.sample()

    obs, info = env.reset(seed=seed)
    total_reward, step, done = 0.0, 0, False

    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated

        stats = info.get("episode_stats", {})
        surv = stats.get("patients_survived", 0)
        lost = stats.get("patients_lost", 0)
        rate = f"{(surv/(surv+lost+1e-6))*100:.1f}%"

        metrics_html = f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px">
            {get_metric_card("System Status", "ACTIVE", f"Step {step}/{env.MAX_STEPS}", "success")}
            {get_metric_card("Net Reward", f"{total_reward:.1f}", "Cumulative", "neutral")}
            {get_metric_card("Survival Rate", rate, f"{surv} Lives Saved", "success" if surv > lost else "warning")}
            {get_metric_card("Critical Missed", stats.get('critical_missed', 0), "Patients Lost", "danger")}
            {get_metric_card("Fleet Load", f"{env.N_AMB - info['available_ambs']}/{env.N_AMB}", "Active Dispatches", "neutral")}
            {get_metric_card("Traffic", f"{info['traffic']:.2f}x", "Travel Delay", "warning" if info['traffic'] > 1.5 else "neutral")}
        </div>"""
        yield render_real_world_map(env, step), metrics_html
        time.sleep(0.05)

# ── AI Debrief (Groq) ─────────────────────────────────────────────────────────

from openai import OpenAI
import os
import json

def generate_llama_debrief(stats_json):
    if not stats_json:
        return "⚠️ Please run a Performance Audit first!"

    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    prompt = f"""
You are the Chief Medical Dispatcher reviewing an RL ambulance dispatch system.

Performance audit:
{json.dumps(stats_json, indent=2)}

Write a professional 3-paragraph performance report:
1. Overall performance summary
2. Strengths and weaknesses
3. Final rating and recommendation
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert emergency operations analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Failed to generate debrief: {str(e)}"
## 🚑 AI Performance Debrief
def generate_llama_debrief(stats_json):
    import os
    import json
    import random
    from openai import OpenAI

    if not stats_json or "error" in stats_json:
        return "⚠️ Please run a Performance Audit first!"

    try:
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

        # add randomness to force report variation
        style_options = [
            "professional executive summary",
            "critical operational review",
            "detailed emergency response analysis",
            "medical command performance debrief"
        ]

        chosen_style = random.choice(style_options)

        prompt = f"""
You are a senior emergency response analyst.

Generate a UNIQUE {chosen_style} based on this audit data.

IMPORTANT:
- Do NOT repeat previous wording
- Use different phrasing every time
- Mention strengths and weaknesses from actual numbers
- Give operational suggestions
- Make the tone dynamic

Audit Data:
{json.dumps(stats_json, indent=2)}

Return markdown report with:
# Summary
# Key Metrics
# Weaknesses
# Recommendation
# Final Rating
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Every response must be different in wording and analysis style."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1.2,   # VERY IMPORTANT
            top_p=0.95,
            max_tokens=600
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ API Error: {str(e)}"
# ── Gradio UI ─────────────────────────────────────────────────────────────────

css = """
.gradio-container { background-color: #020617 !important; }
.map-container { background:#0f172a; border:2px solid #1e293b; border-radius:15px; min-height:700px; }
.control-panel { background:#1e293b !important; padding:20px !important; border-radius:15px !important; }
"""

def build_interface():
    with gr.Blocks(theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"), css=css) as demo:
        gr.HTML("<h1 style='color:white; text-align:center; margin:20px 0;'>🚑 Ambulance Dispatch Dashboard</h1>")

        with gr.Tabs():
            with gr.Tab("Dispatch Terminal"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["control-panel"]):
                        task_dd   = gr.Dropdown(choices=["easy","medium","hard"], value="medium", label="District")
                        policy_dd = gr.Dropdown(choices=["PPO (Trained)","Greedy Heuristic","Random Baseline"], value="PPO (Trained)", label="Policy")
                        seed_sl   = gr.Slider(0, 100, value=42, label="Scenario Seed")
                        run_btn   = gr.Button("▶ Start Simulation", variant="primary")
                        gr.HTML("""
                        <div style="background:#0f172a; border:1px solid #334155; padding:12px; border-radius:10px; margin-top:15px">
                            <h4 style="color:white; margin:0 0 10px 0; font-size:0.9rem;">Map Legend</h4>
                            <div style="color:#94a3b8; font-size:0.75rem; font-weight:bold; margin-bottom:5px;">AMBULANCE</div>
                            <div style="color:#22c55e; font-size:0.8rem">■ Available</div>
                            <div style="color:#eab308; font-size:0.8rem">■ En-Route to Scene</div>
                            <div style="color:#3b82f6; font-size:0.8rem">■ Transporting Patient</div>
                            <div style="color:#94a3b8; font-size:0.75rem; font-weight:bold; margin:8px 0 5px 0;">EMERGENCY</div>
                            <div style="color:#f59e0b; font-size:0.8rem">● Severity 1 (Low)</div>
                            <div style="color:#f97316; font-size:0.8rem">● Severity 2 (High)</div>
                            <div style="color:#ef4444; font-size:0.8rem">● CRITICAL</div>
                        </div>""")
                    with gr.Column(scale=2, elem_classes=["map-container"]):
                        map_display = gr.HTML()
                    with gr.Column(scale=1):
                        gr.Markdown("### Live Metrics")
                        stats_display = gr.HTML("Press Start Simulation to begin.")

            with gr.Tab("Analytics & Grading"):
                with gr.Row():
                    with gr.Column(scale=1):
                        eval_task   = gr.Dropdown(choices=["easy","medium","hard"], value="medium", label="Test District")
                        eval_policy = gr.Dropdown(choices=["PPO (Trained)","Greedy Heuristic"], value="PPO (Trained)", label="Policy")
                        eval_eps    = gr.Slider(5, 50, value=10, label="Episodes")
                        eval_btn    = gr.Button("🏅 Run Performance Audit", variant="primary")
                    with gr.Column(scale=1):
                        eval_out = gr.JSON(label="Audit Report (Score 0.0–1.0)")
                    with gr.Column(scale=1):
                        gr.Markdown("### 🤖 AI Performance Report")
                        debrief_btn    = gr.Button("Generate AI Report", variant="secondary")
                        debrief_output = gr.Markdown("> *Run audit first, then click Generate.*")

        run_btn.click(fn=run_episode_stream, inputs=[task_dd, policy_dd, seed_sl], outputs=[map_display, stats_display])

        def run_grader(task, policy_name, n_episodes):
            if not Grader: return {"error": "Grader module not found"}
            if policy_name == "PPO (Trained)" and PPO_POLICIES.get(task):
                policy_fn = PPO_POLICIES[task]
            elif policy_name == "Greedy Heuristic":
                policy_fn, env_ref = _make_greedy_policy()
                env_ref[0] = AmbulanceDispatchEnv(task=task)
            else:
                env_r = AmbulanceDispatchEnv(task=task)
                policy_fn = lambda obs: env_r.action_space.sample()
            return Grader(task=task).evaluate(policy_fn, n_episodes=int(n_episodes), verbose=False)

        eval_btn.click(fn=run_grader, inputs=[eval_task, eval_policy, eval_eps], outputs=eval_out)
        debrief_btn.click(fn=generate_llama_debrief, inputs=[eval_out], outputs=[debrief_output])

    return demo

demo = build_interface()

# ── OpenEnv REST endpoints (/reset and /step) ─────────────────────────────────

@api.post("/reset")
def reset_env(request: dict = Body(default={})):
    global env_instance
    task = request.get("task", "medium")
    seed = request.get("seed", 42)
    env_instance = AmbulanceDispatchEnv(task=task)
    obs, info = env_instance.reset(seed=seed)
    return {
        "obs":  obs.tolist(),
        "info": info,
        "done": False,
        "action_space_n": env_instance.n_actions,
    }

@api.post("/step")
def step_env(request: dict = Body(default={})):
    global env_instance
    if env_instance is None:
        return {"error": "Call /reset first"}
    action = request.get("action", 0)
    obs, reward, terminated, truncated, info = env_instance.step(action)
    done = terminated or truncated
    return {
        "obs":        obs.tolist(),
        "reward":     float(reward),
        "terminated": bool(terminated),
        "truncated":  bool(truncated),
        "done":       bool(done),
        "info":       info,
    }

@api.get("/health")
def health():
    return {"status": "ok"}

# Mount Gradio and launch
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    # Port 7860 is the default for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860)