import os
import argparse
from typing import Optional
import time
import numpy as np
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from src.env import AmbulanceDispatchEnv


#  Training Hyperparameters


PPO_CONFIG = {
    "easy": {
        "policy":           "MlpPolicy",
        "n_steps":          2048,
        "batch_size":       256,
        "n_epochs":         10,
        "gamma":            0.99,
        "gae_lambda":       0.95,
        "clip_range":       0.2,
        "ent_coef":         0.01,
        "vf_coef":          0.5,
        "max_grad_norm":    0.5,
        "learning_rate":    3e-4,
        "policy_kwargs":    {"net_arch": [128, 128]},
        "total_timesteps":  300_000,
    },
    "medium": {
        "policy":           "MlpPolicy",
        "n_steps":          2048,
        "batch_size":       256,
        "n_epochs":         10,
        "gamma":            0.995,
        "gae_lambda":       0.95,
        "clip_range":       0.2,
        "ent_coef":         0.005,
        "vf_coef":          0.5,
        "max_grad_norm":    0.5,
        "learning_rate":    2e-4,
        "policy_kwargs":    {"net_arch": [256, 256]},
        "total_timesteps":  600_000,
    },
    "hard": {
        "policy":           "MlpPolicy",
        "n_steps":          4096,
        "batch_size":       512,
        "n_epochs":         15,
        "gamma":            0.997,
        "gae_lambda":       0.97,
        "clip_range":       0.15,
        "ent_coef":         0.003,
        "vf_coef":          0.5,
        "max_grad_norm":    0.5,
        "learning_rate":    1e-4,
        "policy_kwargs":    {"net_arch": [512, 256, 128]},
        "total_timesteps":  1_000_000,
    },
}

#  Custom Callback: Rich Logging

class DispatchTrainingCallback(BaseCallback):
    """Log training statistics and display EMS-specific metrics."""

    def __init__(self, log_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq   = log_freq
        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            elapsed = time.time() - self._start_time
            fps     = self.n_calls / max(elapsed, 1)
            print(
                f"  [Step {self.n_calls:>8,}] "
                f"FPS={fps:.0f}  "
                f"elapsed={elapsed:.0f}s"
            )
        return True


#  Environment Factory


def make_env(task: str, seed: int = 0, rank: int = 0):
    """Factory function for vectorized environments."""
    def _init():
        env = AmbulanceDispatchEnv(task=task, seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def build_training_envs(task: str, n_envs: int = 4, seed: int = 42):
    env = make_vec_env(
        lambda: AmbulanceDispatchEnv(task=task),
        n_envs=n_envs,
        seed=seed,
    )
    return env


#  Train Single Task

def train_task(
    task: str,
    save_path: str,
    timesteps: Optional[int] = None,
    n_envs: int = 4,
    seed: int = 42,
    pretrained_model: Optional[str] = None,
    verbose: int = 1,
) -> PPO:
    """
    Train PPO on a single task. Optionally load a pretrained model for
    curriculum fine-tuning.

    Returns:
        Trained PPO model
    """
    cfg = PPO_CONFIG[task].copy()
    total_timesteps = timesteps or cfg.pop("total_timesteps")
    policy          = cfg.pop("policy")

    print(f"\n{'═'*60}")
    print(f"  Training Task: {task.upper()}")
    print(f"  Timesteps:     {total_timesteps:,}")
    print(f"  Environments:  {n_envs}")
    print(f"  Save path:     {save_path}")
    print(f"{'═'*60}\n")

    # Training environment
    train_env = build_training_envs(task, n_envs=n_envs, seed=seed)

    # Evaluation environment (single, deterministic)
    eval_env = Monitor(AmbulanceDispatchEnv(task=task, seed=seed + 9999))

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path + "_best",
        log_path=save_path + "_logs",
        eval_freq=max(1000, total_timesteps // 100),
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )
    log_callback = DispatchTrainingCallback(
        log_freq=max(5000, total_timesteps // 50),
        verbose=verbose,
    )

    # Build or load model
    if pretrained_model and os.path.exists(pretrained_model + ".zip"):
        print(f"  Loading pretrained model: {pretrained_model}")
        model = PPO.load(pretrained_model, env=train_env)
        model.set_env(train_env)
    else:
        model = PPO(
            policy=policy,
            env=train_env,
            verbose=verbose,
            seed=seed,
            tensorboard_log=save_path + "_tb",
            **cfg,
        )

    # Train
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, log_callback],
        reset_num_timesteps=(pretrained_model is None),
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.0f}s")

    # Save
    model.save(save_path)
    print(f"  Model saved → {save_path}.zip")

    train_env.close()
    eval_env.close()
    return model


#  Curriculum Training

def train_curriculum(
    base_dir: str = "models",
    n_envs: int = 4,
    seed: int = 42,
    verbose: int = 1,
):
    """
    Progressive curriculum:
      1. Train on easy → save
      2. Fine-tune on medium using easy weights as init
      3. Fine-tune on hard using medium weights as init
    """
    print("\n🎓 Starting Curriculum Training")
    print("  Easy → Medium → Hard\n")

    os.makedirs(base_dir, exist_ok=True)
    paths = {
        "easy":   f"{base_dir}/ppo_easy",
        "medium": f"{base_dir}/ppo_medium",
        "hard":   f"{base_dir}/ppo_hard",
    }

    # Step 1: Easy
    train_task(
        task="easy",
        save_path=paths["easy"],
        n_envs=n_envs, seed=seed, verbose=verbose,
    )

    # Step 2: Medium (init from easy)
    train_task(
        task="medium",
        save_path=paths["medium"],
        n_envs=n_envs, seed=seed, verbose=verbose,
        pretrained_model=paths["easy"],
    )

    # Step 3: Hard (init from medium)
    train_task(
        task="hard",
        save_path=paths["hard"],
        n_envs=n_envs, seed=seed, verbose=verbose,
        pretrained_model=paths["medium"],
    )

    print("\n✅ Curriculum training complete!")
    for task, path in paths.items():
        print(f"   {task:8s} → {path}.zip")

    return paths

#  Quick Sanity Check

def sanity_check(task: str = "easy"):
    """Verify environment works before training."""
    print(f"\n🔍 Sanity check — task: {task}")
    env = AmbulanceDispatchEnv(task=task)
    obs, info = env.reset(seed=0)
    print(f"  obs shape:   {obs.shape}")
    print(f"  action dim:  {env.action_space.n}")
    print(f"  obs sample:  {obs[:8].round(3)}")

    total_reward = 0.0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
    print(f"  10-step reward (random): {total_reward:.2f}")
    print(f"  State: {env.state()['active_calls']}")
    env.close()
    print("  ✅ Environment OK\n")


#  CLI

from typing import Optional

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on AmbulanceDispatchEnv")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="medium",
                        help="Task to train on ('all' = curriculum)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel envs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None,
                        help="Save path (default: models/ppo_{task})")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model for fine-tuning")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run environment sanity check only")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    if args.sanity_check:
        for t in (["easy", "medium", "hard"] if args.task == "all" else [args.task]):
            sanity_check(t)
    elif args.task == "all":
        train_curriculum(
            base_dir="models",
            n_envs=args.n_envs,
            seed=args.seed,
            verbose=args.verbose,
        )
    else:
        save_path = args.save or f"models/ppo_{args.task}"
        train_task(
            task=args.task,
            save_path=save_path,
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            pretrained_model=args.pretrained,
            verbose=args.verbose,
        )
