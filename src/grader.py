from __future__ import annotations
import numpy as np
from typing import Optional, Callable
from src.env import AmbulanceDispatchEnv

#  Task-specific scoring benchmarks

BENCHMARKS = {
    "easy": {
        "survival_rate_max":       0.92,   # Expected max survival rate
        "response_eff_max":        0.85,   # Response efficiency target
        "resource_util_max":       0.80,   # Resource utilization target
        "critical_rate_max":       0.90,   # Critical patient success rate
        "failure_threshold":       0.10,   # Max acceptable failure rate
        "min_episodes":            10,
        "eval_episodes":           20,
    },
    "medium": {
        "survival_rate_max":       0.82,
        "response_eff_max":        0.75,
        "resource_util_max":       0.70,
        "critical_rate_max":       0.75,
        "failure_threshold":       0.20,
        "min_episodes":            10,
        "eval_episodes":           20,
    },
    "hard": {
        "survival_rate_max":       0.68,
        "response_eff_max":        0.62,
        "resource_util_max":       0.60,
        "critical_rate_max":       0.58,
        "failure_threshold":       0.35,
        "min_episodes":            10,
        "eval_episodes":           20,
    },
}

#  Episode Metrics Collector

class EpisodeMetrics:
    def __init__(self):
        self.total_patients   = 0
        self.survived         = 0
        self.lost             = 0
        self.dispatches       = 0
        self.failed_admissions= 0
        self.critical_served  = 0
        self.critical_missed  = 0
        self.total_wait_time  = 0
        self.total_reward     = 0.0
        self.steps            = 0
        self.beds_used_ratio  = 0.0
        self.icu_used_ratio   = 0.0

    def from_info(self, info: dict, env: AmbulanceDispatchEnv, cumulative_reward: float, steps: int):
        stats = info.get("episode_stats", {})
        self.survived          = stats.get("patients_survived", 0)
        self.lost              = stats.get("patients_lost", 0)
        self.dispatches        = stats.get("total_dispatches", 0)
        self.failed_admissions = stats.get("failed_admissions", 0)
        self.critical_served   = stats.get("critical_served", 0)
        self.critical_missed   = stats.get("critical_missed", 0)
        self.total_wait_time   = stats.get("total_wait_time", 0)
        self.total_patients    = self.survived + self.lost
        self.total_reward      = cumulative_reward
        self.steps             = steps
        # Hospital utilization snapshot
        if env.hospitals:
            self.beds_used_ratio = np.mean([
                1.0 - h.bed_ratio() for h in env.hospitals
            ])
            self.icu_used_ratio = np.mean([
                1.0 - h.icu_ratio() for h in env.hospitals
            ])

    @property
    def survival_rate(self) -> float:
        if self.total_patients == 0:
            return 0.0
        return self.survived / self.total_patients

    @property
    def critical_success_rate(self) -> float:
        total = self.critical_served + self.critical_missed
        if total == 0:
            return 1.0  # no critical calls → N/A, no penalty
        return self.critical_served / total

    @property
    def response_efficiency(self) -> float:
        """Fraction of dispatches that resulted in successful admissions."""
        if self.dispatches == 0:
            return 0.0
        return max(0.0, (self.dispatches - self.failed_admissions) / self.dispatches)

    @property
    def avg_wait(self) -> float:
        if self.dispatches == 0:
            return 0.0
        return self.total_wait_time / max(self.dispatches, 1)

#  Grader

class Grader:
    """
    Evaluates an agent policy over multiple episodes and returns a
    normalized score [0.0, 1.0] along with a detailed breakdown.

    Usage:
        grader = Grader(task="medium")
        result = grader.evaluate(policy_fn, n_episodes=20)
        print(result["score"])   # 0.0 – 1.0
    """

    def __init__(self, task: str = "medium", seed: int = 42):
        assert task in BENCHMARKS, f"Unknown task: {task}"
        self.task  = task
        self.bench = BENCHMARKS[task]
        self.seed  = seed

    def evaluate(
        self,
        policy_fn: Callable[[np.ndarray], int],
        n_episodes: Optional[int] = None,
        verbose: bool = False,
    ) -> dict:
        """
        Args:
            policy_fn: callable(obs) → action (int)
            n_episodes: number of evaluation episodes (default from benchmark)
            verbose: print per-episode info

        Returns:
            dict with keys: score, breakdown, raw_metrics, task
        """
        n_eps = n_episodes or self.bench["eval_episodes"]
        env   = AmbulanceDispatchEnv(task=self.task)
        all_metrics: list[EpisodeMetrics] = []

        for ep in range(n_eps):
            obs, info = env.reset(seed=self.seed + ep)
            ep_reward  = 0.0
            ep_steps   = 0
            done = False
            while not done:
                action = policy_fn(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_steps  += 1
                done = terminated or truncated

            m = EpisodeMetrics()
            m.from_info(info, env, ep_reward, ep_steps)
            all_metrics.append(m)

            if verbose:
                print(
                    f"  Ep {ep+1:3d}: surv={m.survival_rate:.2f} "
                    f"crit={m.critical_success_rate:.2f} "
                    f"reward={m.total_reward:.1f}"
                )

        env.close()
        return self._compute_score(all_metrics)

    def _compute_score(self, metrics: list[EpisodeMetrics]) -> dict:
        b = self.bench

        # Aggregate
        mean = lambda f: float(np.mean([f(m) for m in metrics]))
        survival_rate      = mean(lambda m: m.survival_rate)
        response_eff       = mean(lambda m: m.response_efficiency)
        critical_rate      = mean(lambda m: m.critical_success_rate)
        beds_utilized      = mean(lambda m: m.beds_used_ratio)
        avg_reward         = mean(lambda m: m.total_reward)
        failed_rate        = mean(lambda m: (
            m.failed_admissions / max(m.dispatches, 1)
        ))

        # ── Subscores (each 0–1, then weighted) 
        # 1. Survival rate (40%)
        s1 = min(1.0, survival_rate / max(b["survival_rate_max"], 0.01))

        # 2. Response efficiency (25%)
        s2 = min(1.0, response_eff / max(b["response_eff_max"], 0.01))

        # 3. Resource utilization: moderate utilization is good (15%)
        # Penalize both under-use (too many beds unused) and over-use
        s3 = min(1.0, beds_utilized / max(b["resource_util_max"], 0.01))

        # 4. Critical patient handling (15%)
        s4 = min(1.0, critical_rate / max(b["critical_rate_max"], 0.01))

        # 5. Penalty avoidance: low failure rate (5%)
        failure_penalty = max(0.0, failed_rate - b["failure_threshold"])
        s5 = max(0.0, 1.0 - failure_penalty * 5.0)

        # Weighted composite
        score = (
            0.40 * s1 +
            0.25 * s2 +
            0.15 * s3 +
            0.15 * s4 +
            0.05 * s5
        )
        score = float(np.clip(score, 0.0, 1.0))

        # ── Grade label 
        if   score >= 0.90: grade = "A+"
        elif score >= 0.80: grade = "A"
        elif score >= 0.70: grade = "B"
        elif score >= 0.60: grade = "C"
        elif score >= 0.50: grade = "D"
        else:               grade = "F"

        return {
            "score":   round(score, 4),
            "grade":   grade,
            "task":    self.task,
            "n_episodes": len(metrics),
            "breakdown": {
                "survival_rate":        round(survival_rate, 4),
                "response_efficiency":  round(response_eff, 4),
                "critical_success_rate":round(critical_rate, 4),
                "resource_utilization": round(beds_utilized, 4),
                "failure_rate":         round(failed_rate, 4),
                "avg_episode_reward":   round(avg_reward, 2),
            },
            "subscores": {
                "s1_survival_rate":       round(s1, 4),
                "s2_response_efficiency": round(s2, 4),
                "s3_resource_util":       round(s3, 4),
                "s4_critical_handling":   round(s4, 4),
                "s5_penalty_avoidance":   round(s5, 4),
            },
        }

#  Convenience: Grade All Tasks

def grade_all_tasks(
    policy_fn: Callable[[np.ndarray], int],
    n_episodes: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Run the grader on all three tasks and return a combined report.

    Returns:
        {
            "easy":   GradeResult,
            "medium": GradeResult,
            "hard":   GradeResult,
            "combined_score": float,
        }
    """
    results = {}
    for task in ["easy", "medium", "hard"]:
        if verbose:
            print(f"\n{'─'*40}")
            print(f"  Evaluating task: {task.upper()}")
            print(f"{'─'*40}")
        grader = Grader(task=task)
        result = grader.evaluate(policy_fn, n_episodes=n_episodes, verbose=verbose)
        results[task] = result
        if verbose:
            print(f"  Score: {result['score']:.4f}  Grade: {result['grade']}")
            print(f"  Breakdown: {result['breakdown']}")

    # Weighted combined score (hard counts more)
    combined = (
        0.20 * results["easy"]["score"]   +
        0.35 * results["medium"]["score"] +
        0.45 * results["hard"]["score"]
    )
    results["combined_score"] = round(combined, 4)

    if verbose:
        print(f"\n{'═'*40}")
        print(f"  COMBINED SCORE: {results['combined_score']:.4f}")
        print(f"{'═'*40}")

    return results

#  CLI entry point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grade a random or trained agent")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--model", type=str, default=None, help="Path to trained .zip model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Load model or use random policy
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        policy = lambda obs: int(model.predict(obs, deterministic=True)[0])
        print(f"Loaded model from {args.model}")
    else:
        print("No model specified — using random policy as baseline")
        env_tmp = AmbulanceDispatchEnv(task="easy")
        policy = lambda obs: env_tmp.action_space.sample()

    if args.task == "all":
        results = grade_all_tasks(policy, n_episodes=args.episodes, verbose=True)
    else:
        grader = Grader(task=args.task)
        result = grader.evaluate(policy, n_episodes=args.episodes, verbose=args.verbose)
        print(f"\nScore: {result['score']:.4f}  Grade: {result['grade']}")
        print(f"Breakdown: {result['breakdown']}")
