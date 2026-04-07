# Submission compatibility variables
API_BASE_URL = os.getenv("API_BASE_URL", None)
MODEL_NAME = os.getenv("MODEL_NAME", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", None)


import os
import argparse
import time
import numpy as np
from typing import Optional, Callable

from src.env import AmbulanceDispatchEnv
from src.grader import Grader, grade_all_tasks

#  Policy wrappers

def load_ppo_policy(model_path: str) -> Callable[[np.ndarray], int]:
    """Load a saved SB3 PPO model and return a policy function."""
    from stable_baselines3 import PPO
    if not model_path.endswith(".zip"):
        model_path = model_path + ".zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = PPO.load(model_path)
    print(f"✅ Loaded model: {model_path}")
    return lambda obs: int(model.predict(np.array(obs), deterministic=True)[0])


def random_policy(env: AmbulanceDispatchEnv) -> Callable[[np.ndarray], int]:
    """Random action sampler (baseline)."""
    return lambda obs: env.action_space.sample()


def greedy_policy(task: str) -> Callable[[np.ndarray], int]:
    """
    Rule-based greedy dispatcher (heuristic baseline):
    - Finds nearest available ambulance to highest-priority call
    - Routes to hospital with most available beds
    Does not use the observation vector directly — constructs its own
    decision from a shared environment reference.
    """
    env_ref = [None]

    def _policy(obs: np.ndarray) -> int:
        env = env_ref[0]
        if env is None:
            return 0

        # Find highest-priority unassigned call
        candidates = [c for c in env.calls if c.active and not c.assigned]
        if not candidates:
            return env.n_actions - 1  # wait

        candidates.sort(key=lambda c: (c.severity, c.wait_time), reverse=True)
        target = candidates[0]

        # Find nearest available ambulance
        avail_ambs = [a for a in env.ambulances if a.is_available()]
        if not avail_ambs:
            return env.n_actions - 1  # wait

        nearest_amb = min(avail_ambs, key=lambda a: a.distance_to(target.x, target.y))

        # Find best hospital: prefer capacity + specialty match
        def hospital_score(h):
            if not h.has_capacity(target.severity):
                return -1e9
            spec_bonus = 3.0 if (target.severity == 3 and h.specialty == 1) else 0.0
            dist_penalty = h.distance_to(target.x, target.y)
            return h.bed_ratio() * 10 + spec_bonus - dist_penalty * 0.5

        best_hosp = max(env.hospitals, key=hospital_score)
        action = nearest_amb.id * env.N_HOS + best_hosp.id
        return action

    return _policy, env_ref


#  Single Episode Runner

def run_episode(
    task: str,
    policy_fn: Callable[[np.ndarray], int],
    render: bool = True,
    seed: int = 0,
    env_ref_list: Optional[list] = None,
) -> dict:
    """Run a single episode and return stats."""
    env = AmbulanceDispatchEnv(task=task, render_mode="ansi" if render else None)
    if env_ref_list is not None:
        env_ref_list[0] = env

    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    done  = False

    print(f"\n{'═'*55}")
    print(f"  EMS Dispatch — Task: {task.upper()}   Seed: {seed}")
    print(f"{'═'*55}")

    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        if render and steps % 20 == 0:
            print(env.render())

    # Final render
    if render:
        print(env.render())

    stats = info.get("episode_stats", {})
    survived     = stats.get("patients_survived", 0)
    lost         = stats.get("patients_lost", 0)
    total        = survived + lost
    survival_pct = (survived / total * 100) if total > 0 else 0

    print(f"\n{'─'*55}")
    print(f"  Episode complete — {steps} steps")
    print(f"  Total reward:     {total_reward:.2f}")
    print(f"  Patients:         {survived} survived / {lost} lost ({survival_pct:.1f}% survival)")
    print(f"  Critical served:  {stats.get('critical_served', 0)}")
    print(f"  Critical missed:  {stats.get('critical_missed', 0)}")
    print(f"  Dispatches:       {stats.get('total_dispatches', 0)}")
    print(f"  Failed admissions:{stats.get('failed_admissions', 0)}")
    print(f"{'─'*55}")

    env.close()
    return {
        "task": task, "steps": steps, "total_reward": total_reward,
        "survival_rate": survival_pct / 100,
        **stats,
    }

#  Comparison Benchmark

def compare_policies(model_path: str, task: str, n_episodes: int = 10):
    """Compare trained PPO vs random vs greedy heuristic."""
    print(f"\n{'═'*60}")
    print(f"  Policy Comparison — Task: {task.upper()}  Episodes: {n_episodes}")
    print(f"{'═'*60}")

    grader = Grader(task=task)

    # PPO
    ppo_policy = load_ppo_policy(model_path)
    ppo_result = grader.evaluate(ppo_policy, n_episodes=n_episodes, verbose=False)

    # Random
    env_tmp = AmbulanceDispatchEnv(task=task)
    rand_policy = random_policy(env_tmp)
    rand_result = grader.evaluate(rand_policy, n_episodes=n_episodes, verbose=False)
    env_tmp.close()

    # Greedy heuristic
    greedy_fn, env_ref = greedy_policy(task)
    env_g = AmbulanceDispatchEnv(task=task)
    env_ref[0] = env_g

    def greedy_with_env(obs):
        return greedy_fn(obs)

    greedy_result = grader.evaluate(greedy_with_env, n_episodes=n_episodes, verbose=False)
    env_g.close()

    # Print table
    print(f"\n  {'Policy':<20} {'Score':>8} {'Grade':>6} {'Survival':>10} {'Crit %':>8}")
    print(f"  {'─'*55}")

    for name, result in [("PPO (trained)", ppo_result), ("Greedy heuristic", greedy_result), ("Random", rand_result)]:
        b = result["breakdown"]
        print(
            f"  {name:<20} "
            f"{result['score']:>8.4f} "
            f"{result['grade']:>6} "
            f"{b['survival_rate']:>10.3f} "
            f"{b['critical_success_rate']:>8.3f}"
        )

    print(f"\n  🏆 Best: PPO score = {ppo_result['score']:.4f}")
    return ppo_result, greedy_result, rand_result


#  CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on AmbulanceDispatchEnv")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained PPO model (.zip)")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="medium")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true",
                        help="Suppress step-by-step rendering")
    parser.add_argument("--grade", action="store_true",
                        help="Run the grader after inference")
    parser.add_argument("--compare", action="store_true",
                        help="Compare PPO vs random vs greedy")
    args = parser.parse_args()

    policy_fn = load_ppo_policy(args.model)
    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    if args.compare:
        t = tasks[0]
        compare_policies(args.model, task=t, n_episodes=args.episodes)
    else:
        for t in tasks:
            for ep in range(args.episodes):
                run_episode(
                    task=t,
                    policy_fn=policy_fn,
                    render=not args.no_render,
                    seed=args.seed + ep,
                )

    if args.grade:
        print("\n🏅 Running official grader...")
        results = grade_all_tasks(policy_fn, n_episodes=20, verbose=True)
        print(f"\n🏆 Combined Score: {results['combined_score']:.4f}")
