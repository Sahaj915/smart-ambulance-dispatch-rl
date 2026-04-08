from fastapi import FastAPI, Body
from src.env import AmbulanceDispatchEnv

api = FastAPI()
env_instance = None


@api.post("/reset")
def reset_env(request: dict = Body(default={})):
    global env_instance

    task = request.get("task", "medium")
    seed = request.get("seed", 0)

    env_instance = AmbulanceDispatchEnv(task=task)
    obs, info = env_instance.reset(seed=seed)

    return {
        "observation": obs.tolist() if hasattr(obs, "tolist") else list(obs),
        "info": info
    }


@api.post("/step")
def step_env(request: dict = Body(...)):
    global env_instance

    action = request.get("action", 0)

    obs, reward, terminated, truncated, info = env_instance.step(action)

    return {
        "observation": obs.tolist() if hasattr(obs, "tolist") else list(obs),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    }

import os
import time
import json
import random
import math
import numpy as np
import gradio as gr
from groq import Groq
from dotenv import load_dotenv
load_dotenv()


# importing project modules 
try:
    from src.env import AmbulanceDispatchEnv, TASK_CONFIGS
    from src.grader import Grader, BENCHMARKS
except ImportError:
    # Fallbacks for demonstration if files are missing
    AmbulanceDispatchEnv = None
    Grader = None

#  Policy Loaders

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

#  Advanced "Real-World" SVG Renderer

def render_real_world_map(env, step):
    W, H = 700, 700
    GRID = env.GRID
    PAD = 40
    scale = (W - 2 * PAD) / GRID
    def gx(x): return PAD + x * scale
    def gy(y): return H - PAD - y * scale

    lines = [
        f'<svg width="100%" height="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
        f'<defs><filter id="glow"><feGaussianBlur stdDeviation="2.5" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>',
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
        lines.append(f'<rect x="{cx-18}" y="{cy-18}" width="36" height="36" fill="#000" opacity="0.3" rx="4"/>')
        lines.append(f'<rect x="{cx-15}" y="{cy-15}" width="30" height="30" fill="{color}" rx="6" filter="url(#glow)"/>')
        lines.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" fill="white" font-size="16" font-weight="bold">H</text>')
        fullness = (h.total_beds - h.available_beds) / h.total_beds
        lines.append(f'<circle cx="{cx}" cy="{cy}" r="22" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{138*fullness} 138" transform="rotate(-90 {cx} {cy})"/>')
        lines.append(f'<text x="{cx}" y="{cy+38}" text-anchor="middle" fill="#94a3b8" font-size="10" font-family="sans-serif">{h.name[:10]}</text>')

    for c in env.calls:
        if c.active and not c.assigned:
            cx, cy = gx(c.x), gy(c.y)
            # Define color for Critical Level 3
            color = {1: "#f59e0b", 2: "#f97316", 3: "#ef4444"}[c.severity]
            pulse_r = 10 + (step % 10) * 2
            lines.append(f'<circle cx="{cx}" cy="{cy}" r="{pulse_r}" fill="none" stroke="{color}" stroke-width="2" opacity="{1 - (step%10)/10}"/>')
            lines.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="{color}" filter="url(#glow)"/>')
            
            # Label change: "CRITICAL" for level 3, otherwise "SEV X"
            label_text = "CRITICAL" if c.severity == 3 else f"SEV {c.severity}"
            lines.append(f'<text x="{cx}" y="{cy-15}" text-anchor="middle" fill="{color}" font-size="11" font-weight="bold">{label_text}</text>')

    for a in env.ambulances:
        cx, cy = gx(a.x), gy(a.y)
        status_color = {0: "#22c55e", 1: "#eab308", 2: "#3b82f6", 3: "#64748b"}[a.status]
        lines.append(f'<rect x="{cx-10}" y="{cy-6}" width="20" height="12" rx="3" fill="{status_color}" filter="url(#glow)"/>')
        lines.append(f'<rect x="{cx+2}" y="{cy-4}" width="6" height="8" rx="1" fill="#fff" opacity="0.6"/>')
        lines.append(f'<text x="{cx}" y="{cy+4}" text-anchor="middle" fill="white" font-size="8" font-weight="bold">{a.id}</text>')

    lines.append("</svg>")
    return "\n".join(lines)

#  Dashboard Utilities

def get_metric_card(title, value, trend="", variant="neutral"):
    color = {"neutral": "#334155", "success": "#065f46", "warning": "#854d0e", "danger": "#7f1d1d"}[variant]
    return f"""
    <div style="background:{color}; padding:12px; border-radius:10px; border: 1px solid #475569; margin-bottom:8px">
        <div style="color:#cbd5e1; font-size:0.7rem; font-weight:bold; text-transform:uppercase">{title}</div>
        <div style="color:white; font-size:1.5rem; font-weight:bold">{value}</div>
        <div style="color:#94a3b8; font-size:0.7rem">{trend}</div>
    </div>
    """

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
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px">
            {get_metric_card("System Status", "ACTIVE", f"Step {step}/{env.MAX_STEPS}", "success")}
            {get_metric_card("Net Reward", f"{total_reward:.1f}", "Reward Cumulative", "neutral")}
            {get_metric_card("Survival Rate", rate, f"{surv} Lives Saved", "success" if surv > lost else "warning")}
            {get_metric_card("Critical Alert", stats.get('critical_missed', 0), "Patients Lost", "danger")}
            {get_metric_card("Fleet Load", f"{env.N_AMB - info['available_ambs']}/{env.N_AMB}", "Active Dispatches", "neutral")}
            {get_metric_card("Avg Traffic", f"{info['traffic']:.2f}x", "Travel Delay", "warning" if info['traffic'] > 1.5 else "neutral")}
        </div>
        """
        yield render_real_world_map(env, step), metrics_html
        time.sleep(0.05)


#  Llama 3 GenAI Debrief Engine - FIX APPLIED HERE

def generate_llama_debrief(stats_json):
    if not stats_json:
        return "⚠️ Please run a Performance Audit first!"

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "⚠️ GROQ API key not found"

    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        return f"⚠️ Error connecting to Groq API: {str(e)}"

    prompt = f"""
    Act as the Chief Medical Dispatcher for a city. You are reviewing the performance of an autonomous AI dispatch system based on a recent audit.
    
    Here is the AI's performance audit data (in JSON format):
    {json.dumps(stats_json, indent=2)}
    
    Write a brief, 3-paragraph executive debriefing. 
    1. Acknowledge the overall performance, survival rates, and reward metric.
    2. Analyze the efficiency and point out any bottlenecks or failures from the data.
    3. Give the AI system a final performance rating (e.g., "Outstanding", "Needs Retraining").
    Keep it professional, analytical, and highly engaging.
    """

    try:
        # UPDATED MODEL NAME TO llama-3.1-8b-instant
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=400,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Failed to generate debrief: {str(e)}"


#  UI layout

def build_interface():
    css = """
    .gradio-container { background-color: #020617 !important; }
    .map-container { background: #0f172a;
    border: 2px solid #1e293b;
    border-radius: 15px;
    min-height: 700px;
    height: auto;
    overflow: visible; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
    .control-panel { background: #1e293b !important; padding: 20px !important; border-radius: 15px !important; }
    .legend-box { background: #0f172a; border: 1px solid #334155; padding: 12px; border-radius: 10px; margin-top: 15px; }
    .legend-item { display: flex; align-items: center; margin-bottom: 6px; font-size: 0.85rem; color: #cbd5e1; }
    .dot { width: 12px; height: 12px; border-radius: 3px; margin-right: 10px; }
    """
    
    with gr.Blocks(theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"), css=css) as demo:
        gr.HTML("<h1 style='color:white; text-align:center; margin:20px 0;'>Ambulance Dispatch Dashboard</h1>")
        
        with gr.Tabs():
            with gr.Tab("Dispatch Terminal"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["control-panel"]):
                        task_dd = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="District")
                        policy_dd = gr.Dropdown(choices=["PPO (Trained)", "Greedy Heuristic", "Random Baseline"], value="PPO (Trained)", label="Policy")
                        seed_sl = gr.Slider(0, 100, value=42, label="Scenario Seed")
                        run_btn = gr.Button("Start Simulation", variant="primary")
                        
                        gr.HTML("""
                        <div class="legend-box">
                            <h4 style="color:white; margin-top:0; margin-bottom:10px; font-size:0.9rem;">Map Legend</h4>
                            <p style="color:#94a3b8; font-size:0.75rem; font-weight:bold; margin-bottom:5px;">AMBULANCE STATUS</p>
                            <div class="legend-item"><div class="dot" style="background:#22c55e;"></div>Available / Stationary</div>
                            <div class="legend-item"><div class="dot" style="background:#eab308;"></div>En-Route to Emergency</div>
                            <div class="legend-item"><div class="dot" style="background:#3b82f6;"></div>Transporting to Hospital</div>
                            <div class="legend-item"><div class="dot" style="background:#64748b;"></div>Busy / Out of Service</div>
                            <hr style="border: 0.5px solid #334155; margin: 10px 0;">
                            <p style="color:#94a3b8; font-size:0.75rem; font-weight:bold; margin-bottom:5px;">EMERGENCY SEVERITY</p>
                            <div class="legend-item"><div class="dot" style="background:#f59e0b;"></div>Severity 1 (Low)</div>
                            <div class="legend-item"><div class="dot" style="background:#f97316;"></div>Severity 2 (High)</div>
                            <div class="legend-item"><div class="dot" style="background:#ef4444;"></div>Severity 3 (CRITICAL)</div>
                        </div>
                        """)

                    with gr.Column(scale=2, elem_classes=["map-container"]):
                        map_display = gr.HTML(label="Tactical Map")

                    with gr.Column(scale=1):
                        gr.Markdown("### Live Metrics")
                        stats_display = gr.HTML("Initialize dispatch to see data...")

            with gr.Tab("Analytics & Grading"):
                with gr.Row():
                    with gr.Column(scale=1):
                        eval_task = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Test District")
                        eval_policy = gr.Dropdown(choices=["PPO (Trained)", "Greedy Heuristic"], value="PPO (Trained)", label="Policy")
                        eval_eps = gr.Slider(5, 50, value=10, label="Simulation Episodes")
                        eval_btn = gr.Button("RUN PERFORMANCE AUDIT", variant="primary")
                    
                    with gr.Column(scale=1):
                        eval_out = gr.JSON(label="Audit Report")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🤖 AI Performance Report")
                        debrief_btn = gr.Button("Generate Report", variant="secondary")
                        debrief_output = gr.Markdown("> *Run a Performance Audit first, then click generate.*")

        run_btn.click(fn=run_episode_stream, inputs=[task_dd, policy_dd, seed_sl], outputs=[map_display, stats_display])
        
        def run_grader(task, policy_name, n_episodes):
            if not Grader: return {"error": "Grader module not found"}
            
            if policy_name == "PPO (Trained)" and PPO_POLICIES.get(task):
                policy_fn = PPO_POLICIES[task]
            elif policy_name == "Greedy Heuristic":
                policy_fn, env_ref = _make_greedy_policy()
                env_g = AmbulanceDispatchEnv(task=task)
                env_ref[0] = env_g
            else:
                env_r = AmbulanceDispatchEnv(task=task)
                policy_fn = lambda obs: env_r.action_space.sample()

            grader = Grader(task=task)
            return grader.evaluate(policy_fn, n_episodes=int(n_episodes), verbose=False)

        eval_btn.click(fn=run_grader, inputs=[eval_task, eval_policy, eval_eps], outputs=eval_out)
        debrief_btn.click(fn=generate_llama_debrief, inputs=[eval_out], outputs=[debrief_output])

    return demo


demo = build_interface()

@api.post("/reset")
def reset_env(request: dict = Body(default={})):
    global env_instance

    task = request.get("task", "medium")
    seed = request.get("seed", 0)

    env_instance = AmbulanceDispatchEnv(task=task)
    obs, info = env_instance.reset(seed=seed)

    return {
        "observation": obs.tolist() if hasattr(obs, "tolist") else list(obs),
        "info": info
    }


@api.post("/step")
def step_env(request: dict = Body(default={})):
    global env_instance

    if env_instance is None:
        return {"error": "Call /reset first"}

    action = request.get("action", 0)

    obs, reward, terminated, truncated, info = env_instance.step(action)

    return {
        "observation": obs.tolist() if hasattr(obs, "tolist") else list(obs),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    }


app = gr.mount_gradio_app(api, demo, path="/")
