import os
import argparse
import numpy as np

from typing import Optional, Callable
from fastapi import FastAPI
from pydantic import BaseModel

from src.env import AmbulanceDispatchEnv
from src.grader import Grader, grade_all_tasks


# ==============================
# Submission compatibility vars
# ==============================
API_BASE_URL = os.getenv("API_BASE_URL", None)
MODEL_NAME = os.getenv("MODEL_NAME", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", None)


# ==============================
# FastAPI app for OpenEnv checks
# ==============================
app = FastAPI()

env_instance = None


class ResetRequest(BaseModel):
    task: Optional[str] = "medium"
    seed: Optional[int] = 0


class StepRequest(BaseModel):
    action: int


@app.post("/reset")
def reset_env():
    global env_instance

    env_instance = AmbulanceDispatchEnv(task="medium")
    obs, info = env_instance.reset(seed=0)

    return {
        "observation": obs.tolist() if hasattr(obs, "tolist") else list(obs),
        "info": info
    }


@app.post("/step")
def step_env(request: StepRequest):
    global env_instance

    if env_instance is None:
        return {
            "error": "Environment not initialized. Call /reset first."
        }

    obs, reward, terminated, truncated, info = env_instance.step(request.action)

    return {
        "observation": obs.tolist() if hasattr(obs, "tolist") else list(obs),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    }


# ==============================
# PPO policy wrappers
# ==============================
def load_ppo_policy(model_path: str) -> Callable[[np.ndarray], int]:
    from stable_baselines3 import PPO

    if not model_path.endswith(".zip"):
        model_path += ".zip"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = PPO.load(model_path)
    print(f"Loaded model: {model_path}")

    return lambda obs: int(
        model.predict(np.array(obs), deterministic=True)[0]
    )


def random_policy(env: AmbulanceDispatchEnv) -> Callable[[np.ndarray], int]:
    return lambda obs: env.action_space.sample()


def greedy_policy(task: str):
    env_ref = [None]

    def _policy(obs: np.ndarray) -> int:
        env = env_ref[0]

        if env is None:
            return 0

        candidates = [
            c for c in env.calls
            if c.active and not c.assigned
        ]

        if not candidates:
            return env.n_actions - 1

        candidates.sort(
            key=lambda c: (c.severity, c.wait_time),
            reverse=True
        )

        target = candidates[0]

        avail_ambs = [
            a for a in env.ambulances
            if a.is_available()
        ]

        if not avail_ambs:
            return env.n_actions - 1

        nearest_amb = min(
            avail_ambs,
            key=lambda a: a.distance_to(target.x, target.y)
        )

        def hospital_score(h):
            if not h.has_capacity(target.severity):
                return -1e9

            spec_bonus = (
                3.0
                if target.severity == 3 and h.specialty == 1
                else 0.0
            )

            dist_penalty = h.distance_to(target.x, target.y)

            return (
                h.bed_ratio() * 10
                + spec_bonus
                - dist_penalty * 0.5
            )

        best_hosp = max(env.hospitals, key=hospital_score)

        return nearest_amb.id * env.N_HOS + best_hosp.id

    return _policy, env_ref


# ==============================
# Episode runner
# ==============================
def run_episode(
    task: str,
    policy_fn: Callable[[np.ndarray], int],
    render: bool = True,
    seed: int = 0
):
    env = AmbulanceDispatchEnv(task=task)
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = policy_fn(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        done = terminated or truncated

    env.close()

    return {
        "task": task,
        "steps": steps,
        "total_reward": total_reward
    }


# ==============================
# Policy comparison
# ==============================
def compare_policies(model_path: str, task: str, n_episodes: int = 10):
    grader = Grader(task=task)

    ppo_policy = load_ppo_policy(model_path)
    ppo_result = grader.evaluate(
        ppo_policy,
        n_episodes=n_episodes,
        verbose=False
    )

    env_tmp = AmbulanceDispatchEnv(task=task)
    rand_policy = random_policy(env_tmp)

    rand_result = grader.evaluate(
        rand_policy,
        n_episodes=n_episodes,
        verbose=False
    )

    env_tmp.close()

    return ppo_result, rand_result


# ==============================
# CLI
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="medium"
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--grade", action="store_true")

    args = parser.parse_args()

    policy_fn = load_ppo_policy(args.model)

    tasks = (
        ["easy", "medium", "hard"]
        if args.task == "all"
        else [args.task]
    )

    if args.compare:
        compare_policies(
            args.model,
            task=tasks[0],
            n_episodes=args.episodes
        )
    else:
        for task in tasks:
            for ep in range(args.episodes):
                run_episode(
                    task=task,
                    policy_fn=policy_fn,
                    seed=args.seed + ep
                )

    if args.grade:
        results = grade_all_tasks(
            policy_fn,
            n_episodes=20,
            verbose=True
        )

        print(
            f"Combined Score: "
            f"{results['combined_score']:.4f}"
        )