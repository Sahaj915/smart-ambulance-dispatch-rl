"""
Gymnasium environment for ambulance dispatch simulation.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
import random
import math


#  Data Classes

@dataclass
class Ambulance:
    id: int
    x: float
    y: float
    base_x: float
    base_y: float
    status: int = 0          # 0=available, 1=to_scene, 2=to_hospital, 3=busy
    time_remaining: int = 0  # steps until next free
    current_severity: int = 0
    total_dispatches: int = 0

    def is_available(self) -> bool:
        return self.status == 0

    def distance_to(self, x: float, y: float) -> float:
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)


@dataclass
class Hospital:
    id: int
    x: float
    y: float
    name: str
    total_beds: int
    available_beds: int
    total_icu: int
    available_icu: int
    specialty: int = 0       # 0=general, 1=trauma_center, 2=cardiac_center
    patients_admitted: int = 0
    patients_rejected: int = 0

    def has_capacity(self, severity: int) -> bool:
        if severity == 3:    # critical → needs ICU
            return self.available_icu > 0 and self.available_beds > 0
        return self.available_beds > 0

    def admit_patient(self, severity: int) -> bool:
        if not self.has_capacity(severity):
            self.patients_rejected += 1
            return False
        self.available_beds -= 1
        if severity == 3:
            self.available_icu -= 1
        self.patients_admitted += 1
        return True

    def distance_to(self, x: float, y: float) -> float:
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def bed_ratio(self) -> float:
        if self.total_beds == 0:
            return 0.0
        return self.available_beds / self.total_beds

    def icu_ratio(self) -> float:
        if self.total_icu == 0:
            return 0.0
        return self.available_icu / self.total_icu


@dataclass
class EmergencyCall:
    id: int
    x: float
    y: float
    severity: int            # 1=low, 2=medium, 3=critical
    wait_time: int = 0
    active: bool = True
    assigned: bool = False
    assigned_ambulance: int = -1
    assigned_hospital: int = -1

    def survival_probability(self) -> float:
        """Survival probability decreases with wait time and severity."""
        base = {1: 0.99, 2: 0.90, 3: 0.75}[self.severity]
        decay = {1: 0.001, 2: 0.005, 3: 0.015}[self.severity]
        return max(0.0, base - decay * self.wait_time)


#  Task Configurations
TASK_CONFIGS = {
    "easy": {
        "grid_size": 10.0,
        "num_ambulances": 3,
        "num_hospitals": 2,
        "max_active_calls": 3,
        "max_steps": 200,
        "call_spawn_prob": 0.10,
        "traffic_variance": 0.1,
        "bed_refill_rate": 0.05,
        "severity_weights": [0.6, 0.3, 0.1],   # mostly low
        "description": "Small city, few resources, low call volume",
    },
    "medium": {
        "grid_size": 10.0,
        "num_ambulances": 5,
        "num_hospitals": 3,
        "max_active_calls": 6,
        "max_steps": 300,
        "call_spawn_prob": 0.20,
        "traffic_variance": 0.25,
        "bed_refill_rate": 0.03,
        "severity_weights": [0.4, 0.4, 0.2],
        "description": "Medium city, moderate resources, mixed severity calls",
    },
    "hard": {
        "grid_size": 10.0,
        "num_ambulances": 8,
        "num_hospitals": 4,
        "max_active_calls": 10,
        "max_steps": 500,
        "call_spawn_prob": 0.30,
        "traffic_variance": 0.40,
        "bed_refill_rate": 0.02,
        "severity_weights": [0.2, 0.4, 0.4],   # many critical
        "description": "Large city, resource-constrained, high critical call volume",
    },
}

# Pre-defined hospital layouts per task
HOSPITAL_LAYOUTS = {
    "easy": [
        {"name": "City General",    "x": 2.0, "y": 8.0, "beds": 20, "icu": 4,  "specialty": 0},
        {"name": "East Trauma Ctr", "x": 8.0, "y": 2.0, "beds": 15, "icu": 6,  "specialty": 1},
    ],
    "medium": [
        {"name": "Metro General",   "x": 2.0, "y": 8.0, "beds": 25, "icu": 6,  "specialty": 0},
        {"name": "North Trauma",    "x": 8.0, "y": 8.0, "beds": 20, "icu": 8,  "specialty": 1},
        {"name": "South Cardiac",   "x": 5.0, "y": 1.0, "beds": 18, "icu": 5,  "specialty": 2},
    ],
    "hard": [
        {"name": "Central Hospital","x": 5.0, "y": 5.0, "beds": 30, "icu": 8,  "specialty": 0},
        {"name": "North Trauma",    "x": 2.0, "y": 9.0, "beds": 25, "icu": 10, "specialty": 1},
        {"name": "East Cardiac",    "x": 9.0, "y": 5.0, "beds": 20, "icu": 6,  "specialty": 2},
        {"name": "South General",   "x": 5.0, "y": 1.0, "beds": 22, "icu": 4,  "specialty": 0},
    ],
}

AMBULANCE_BASES = {
    "easy":   [(1.0, 5.0), (5.0, 5.0), (9.0, 5.0)],
    "medium": [(1.0, 3.0), (1.0, 7.0), (5.0, 5.0), (9.0, 3.0), (9.0, 7.0)],
    "hard":   [(1.0, 1.0), (1.0, 5.0), (1.0, 9.0), (5.0, 1.0),
               (5.0, 9.0), (9.0, 1.0), (9.0, 5.0), (9.0, 9.0)],
}


#  Main Environment

class AmbulanceDispatchEnv(gym.Env):
    """
    Main RL environment for ambulance dispatch simulation.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"]}

    # Observation feature sizes
    AMBULANCE_FEATURES = 5   # x, y, status, time_remaining, current_severity
    HOSPITAL_FEATURES  = 6   # x, y, bed_ratio, icu_ratio, specialty, distance_to_centroid
    CALL_FEATURES      = 6   # x, y, severity, wait_time, active, survival_prob

    def __init__(self, task: str = "medium", render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()
        assert task in TASK_CONFIGS, f"task must be one of {list(TASK_CONFIGS.keys())}"
        self.task = task
        self.cfg = TASK_CONFIGS[task]
        self.render_mode = render_mode
        self._seed = seed

        # Derived constants
        self.N_AMB  = self.cfg["num_ambulances"]
        self.N_HOS  = self.cfg["num_hospitals"]
        self.N_CALL = self.cfg["max_active_calls"]
        self.GRID   = self.cfg["grid_size"]
        self.MAX_STEPS = self.cfg["max_steps"]

        # Action space: N_AMB × N_HOS dispatch combos + 1 wait
        self.n_actions = self.N_AMB * self.N_HOS + 1
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation space
        obs_size = (
            self.N_AMB  * self.AMBULANCE_FEATURES +
            self.N_HOS  * self.HOSPITAL_FEATURES  +
            self.N_CALL * self.CALL_FEATURES       +
            4  # global: step_ratio, traffic, pending_count, critical_waiting
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Runtime state (populated in reset)
        self.ambulances:  List[Ambulance]     = []
        self.hospitals:   List[Hospital]      = []
        self.calls:       List[EmergencyCall] = []
        self._call_id_counter = 0
        self._step_count = 0
        self._episode_stats: Dict[str, float] = {}
        self._traffic = 1.0

        # Rendering
        self._render_surface = None

    #  Gymnasium API 

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed or self._seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._step_count = 0
        self._call_id_counter = 0
        self._traffic = 1.0 + np.random.uniform(-0.1, 0.1)

        # Initialize ambulances at their base stations
        self.ambulances = []
        for i, (bx, by) in enumerate(AMBULANCE_BASES[self.task]):
            self.ambulances.append(Ambulance(id=i, x=bx, y=by, base_x=bx, base_y=by))

        # Initialize hospitals from layout
        self.hospitals = []
        for i, h in enumerate(HOSPITAL_LAYOUTS[self.task]):
            self.hospitals.append(Hospital(
                id=i, x=h["x"], y=h["y"], name=h["name"],
                total_beds=h["beds"],    available_beds=h["beds"],
                total_icu=h["icu"],      available_icu=h["icu"],
                specialty=h["specialty"]
            ))

        # Seed with 1–2 initial calls
        self.calls = []
        n_initial = random.randint(1, min(2, self.N_CALL))
        for _ in range(n_initial):
            self._spawn_call()

        # Episode stats
        self._episode_stats = {
            "total_dispatches":      0,
            "successful_admissions": 0,
            "failed_admissions":     0,
            "patients_survived":     0,
            "patients_lost":         0,
            "avg_response_time":     0.0,
            "avg_survival_prob":     0.0,
            "critical_served":       0,
            "critical_missed":       0,
            "total_wait_time":       0,
        }

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        # ── 1. Decode action 
        if action == self.n_actions - 1:
            # WAIT action
            reward += self._reward_wait()
        else:
            ambulance_id = action // self.N_HOS
            hospital_id  = action  % self.N_HOS
            reward += self._execute_dispatch(ambulance_id, hospital_id)

        # ── 2. Advance time / tick all active units 
        self._tick_ambulances()

        # ── 3. Age waiting calls (penalty per step) 
        reward += self._age_calls()

        # ── 4. Maybe spawn a new call 
        self._maybe_spawn_call()

        # ── 5. Randomly fluctuate traffic 
        self._update_traffic()

        # ── 6. Refill some hospital beds (patient discharge) ─
        self._refill_beds()

        # ── 7. Check terminal conditions 
        if self._step_count >= self.MAX_STEPS:
            truncated = True

        obs  = self._get_observation()
        info = self._get_info()
        return obs, float(reward), terminated, truncated, info

    def state(self) -> Dict[str, Any]:
        """Return human-readable state dictionary (OpenEnv convention)."""
        return {
            "step": self._step_count,
            "traffic": round(self._traffic, 3),
            "ambulances": [
                {
                    "id":             a.id,
                    "position":       (round(a.x, 2), round(a.y, 2)),
                    "status":         ["available", "to_scene", "to_hospital", "busy"][a.status],
                    "time_remaining": a.time_remaining,
                    "dispatches":     a.total_dispatches,
                }
                for a in self.ambulances
            ],
            "hospitals": [
                {
                    "id":            h.id,
                    "name":          h.name,
                    "beds":          f"{h.available_beds}/{h.total_beds}",
                    "icu":           f"{h.available_icu}/{h.total_icu}",
                    "specialty":     ["general", "trauma", "cardiac"][h.specialty],
                    "admitted":      h.patients_admitted,
                    "rejected":      h.patients_rejected,
                }
                for h in self.hospitals
            ],
            "active_calls": [
                {
                    "id":       c.id,
                    "position": (round(c.x, 2), round(c.y, 2)),
                    "severity": ["", "low", "medium", "critical"][c.severity],
                    "wait":     c.wait_time,
                    "assigned": c.assigned,
                    "survival": round(c.survival_probability(), 3),
                }
                for c in self.calls if c.active
            ],
            "episode_stats": self._episode_stats,
        }

    def render(self) -> Optional[Any]:
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def close(self):
        pass

    # Dispatch Logic 

    def _execute_dispatch(self, ambulance_id: int, hospital_id: int) -> float:
        """
        Attempt to dispatch ambulance_id to the highest-priority unassigned
        call, routing the patient to hospital_id upon pickup.
        Returns shaped reward.
        """
        reward = 0.0

        # Validate ambulance
        if ambulance_id >= self.N_AMB:
            return -1.0

        amb = self.ambulances[ambulance_id]
        if not amb.is_available():
            reward -= 2.0   # penalty for dispatching busy ambulance
            return reward

        # Find highest-priority unassigned call
        target_call = self._get_priority_call()
        if target_call is None:
            reward -= 0.5   # dispatching when no calls pending
            return reward

        hosp = self.hospitals[hospital_id]

        # Reward shaping 

        # 1. Response time reward: closer ambulance → higher reward
        dist_to_scene = amb.distance_to(target_call.x, target_call.y)
        max_dist = self.GRID * math.sqrt(2)
        response_reward = 5.0 * (1.0 - dist_to_scene / max_dist)
        reward += response_reward

        # 2. Severity bonus for dispatching to critical call
        if target_call.severity == 3:
            reward += 5.0
        elif target_call.severity == 2:
            reward += 2.0
        else:
            reward += 0.5

        # 3. Hospital suitability reward
        reward += self._hospital_suitability_reward(target_call, hosp)

        # 4. Hospital capacity reward / penalty
        if hosp.has_capacity(target_call.severity):
            reward += 3.0
        else:
            reward -= 5.0   # routing to full hospital is a bad decision
            # Still proceed — dispatcher doesn't always know real-time capacity perfectly

        # 5. Survival probability at dispatch time
        survival_bonus = target_call.survival_probability() * 3.0
        reward += survival_bonus

        # 6. Penalty for ignoring higher-severity patients if any critical waiting
        reward += self._priority_violation_penalty(target_call)

        # Execute dispatch 
        travel_time = self._compute_travel_time(amb.x, amb.y, target_call.x, target_call.y)
        amb.status = 1   # en route to scene
        amb.time_remaining = travel_time
        amb.current_severity = target_call.severity
        amb.total_dispatches += 1

        target_call.assigned = True
        target_call.assigned_ambulance = ambulance_id
        target_call.assigned_hospital  = hospital_id

        self._episode_stats["total_dispatches"] += 1

        # Schedule hospital arrival (scene + hospital leg)
        dist_to_hosp = hosp.distance_to(target_call.x, target_call.y)
        hosp_travel = self._compute_travel_time(target_call.x, target_call.y, hosp.x, hosp.y)
        total_time = travel_time + hosp_travel

        # Attempt hospital admission
        admitted = hosp.admit_patient(target_call.severity)
        if admitted:
            self._episode_stats["successful_admissions"] += 1
            # Patient survival outcome
            survival_prob = target_call.survival_probability()
            survived = random.random() < survival_prob
            if survived:
                reward += 10.0
                self._episode_stats["patients_survived"] += 1
                if target_call.severity == 3:
                    self._episode_stats["critical_served"] += 1
            else:
                reward -= 3.0
                self._episode_stats["patients_lost"] += 1
        else:
            reward -= 8.0   # patient cannot be admitted
            self._episode_stats["failed_admissions"] += 1
            if target_call.severity == 3:
                self._episode_stats["critical_missed"] += 1

        # Mark call resolved; ambulance will return after total_time steps
        target_call.active = False
        self._episode_stats["total_wait_time"] += target_call.wait_time

        # Set ambulance to busy for round-trip time
        amb.status = 2
        amb.time_remaining = total_time

        return reward

    def _hospital_suitability_reward(self, call: EmergencyCall, hosp: Hospital) -> float:
        """Check hospital suitability."""
        # Trauma center → best for critical
        if call.severity == 3 and hosp.specialty == 1:
            return 4.0
        # Cardiac center good for medium/critical cardiac events (approximated by severity 2/3)
        if call.severity >= 2 and hosp.specialty == 2:
            return 2.0
        # General hospital acceptable for any severity
        if hosp.specialty == 0:
            return 0.5
        # Mismatch: sending low-severity to trauma center wastes resources
        if call.severity == 1 and hosp.specialty == 1:
            return -1.0
        return 0.0

    def _priority_violation_penalty(self, dispatched_call: EmergencyCall) -> float:
        """Penalty for skipping critical calls."""
        for c in self.calls:
            if c.active and not c.assigned and c.id != dispatched_call.id:
                if c.severity > dispatched_call.severity:
                    return -4.0  # skipping a more critical patient
        return 0.0

    def _reward_wait(self) -> float:
        """Reward/penalty for the wait action."""
        # Waiting is only acceptable if all ambulances are busy OR no calls
        all_busy = all(not a.is_available() for a in self.ambulances)
        no_calls = all(not c.active or c.assigned for c in self.calls)

        if all_busy or no_calls:
            return 0.1   # acceptable wait
        # Penalize waiting when there are actionable calls and free ambulances
        has_critical = any(c.active and not c.assigned and c.severity == 3 for c in self.calls)
        if has_critical:
            return -6.0
        return -1.0

    # Time and simulation updates

    def _tick_ambulances(self):
        """Advance ambulance timers; return to base when done."""
        for amb in self.ambulances:
            if amb.status in (1, 2) and amb.time_remaining > 0:
                amb.time_remaining -= 1
                if amb.time_remaining == 0:
                    # Return ambulance to nearest base or hospital vicinity
                    amb.x = amb.base_x
                    amb.y = amb.base_y
                    amb.status = 0
                    amb.current_severity = 0

    def _age_calls(self) -> float:
        """Increment wait times; apply survival decay penalty."""
        reward = 0.0
        expired = []
        for call in self.calls:
            if call.active and not call.assigned:
                call.wait_time += 1
                # Living penalty: each step a critical patient waits
                if call.severity == 3:
                    reward -= 0.5
                elif call.severity == 2:
                    reward -= 0.15
                else:
                    reward -= 0.05
                # Call expires if waited too long (patient deteriorates terminally)
                max_wait = {1: 60, 2: 30, 3: 15}[call.severity]
                if call.wait_time >= max_wait:
                    call.active = False
                    expired.append(call)
                    reward -= {1: 2.0, 2: 5.0, 3: 15.0}[call.severity]
                    self._episode_stats["patients_lost"] += 1
                    if call.severity == 3:
                        self._episode_stats["critical_missed"] += 1
        return reward

    def _maybe_spawn_call(self):
        """Stochastically spawn a new emergency call."""
        active_count = sum(1 for c in self.calls if c.active)
        if active_count < self.N_CALL:
            if random.random() < self.cfg["call_spawn_prob"]:
                self._spawn_call()

    def _spawn_call(self):
        severity = random.choices([1, 2, 3], weights=self.cfg["severity_weights"])[0]
        call = EmergencyCall(
            id=self._call_id_counter,
            x=random.uniform(0.5, self.GRID - 0.5),
            y=random.uniform(0.5, self.GRID - 0.5),
            severity=severity,
        )
        self.calls.append(call)
        self._call_id_counter += 1

    def _update_traffic(self):
        """Random-walk traffic level."""
        delta = np.random.normal(0, self.cfg["traffic_variance"] * 0.1)
        self._traffic = float(np.clip(self._traffic + delta, 0.5, 2.5))

    def _refill_beds(self):
        """Simulate patient discharges — slowly refill beds."""
        for hosp in self.hospitals:
            if random.random() < self.cfg["bed_refill_rate"]:
                hosp.available_beds = min(hosp.total_beds, hosp.available_beds + 1)
            if random.random() < self.cfg["bed_refill_rate"] * 0.5:
                hosp.available_icu = min(hosp.total_icu, hosp.available_icu + 1)

    def _compute_travel_time(self, x1: float, y1: float, x2: float, y2: float) -> int:
        """Calculate travel time."""
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        # 1 unit distance = 2 steps at normal traffic
        base_time = max(1, int(dist * 2 * self._traffic))
        return base_time

# Observation and info helpers
    def _get_observation(self) -> np.ndarray:
        obs = []

        # Ambulance features (N_AMB × 5)
        for i in range(self.N_AMB):
            if i < len(self.ambulances):
                a = self.ambulances[i]
                obs.extend([
                    a.x / self.GRID,
                    a.y / self.GRID,
                    a.status / 3.0,
                    min(a.time_remaining, 50) / 50.0,
                    a.current_severity / 3.0,
                ])
            else:
                obs.extend([0.0] * self.AMBULANCE_FEATURES)

        # Hospital features (N_HOS × 6)
        for i in range(self.N_HOS):
            if i < len(self.hospitals):
                h = self.hospitals[i]
                obs.extend([
                    h.x / self.GRID,
                    h.y / self.GRID,
                    h.bed_ratio(),
                    h.icu_ratio(),
                    h.specialty / 2.0,
                    h.distance_to(5.0, 5.0) / (self.GRID * math.sqrt(2)),
                ])
            else:
                obs.extend([0.0] * self.HOSPITAL_FEATURES)

        # Call slot features (N_CALL × 6)
        active_calls = [c for c in self.calls if c.active and not c.assigned]
        active_calls.sort(
    key=lambda c: (-c.severity, -c.wait_time)) # highest severity first

        for i in range(self.N_CALL):
            if i < len(active_calls):
                c = active_calls[i]
                obs.extend([
                    c.x / self.GRID,
                    c.y / self.GRID,
                    c.severity / 3.0,
                    min(c.wait_time, 100) / 100.0,
                    1.0,   # active=True
                    c.survival_probability(),
                ])
            else:
                obs.extend([0.0] * self.CALL_FEATURES)

        # Global context (4)
        pending_count = sum(1 for c in self.calls if c.active and not c.assigned)
        critical_waiting = sum(1 for c in self.calls if c.active and not c.assigned and c.severity == 3)
        obs.extend([
            self._step_count / self.MAX_STEPS,
            (self._traffic - 0.5) / 2.0,            # normalize [0.5,2.5]→[0,1]
            pending_count / max(self.N_CALL, 1),
            critical_waiting / max(self.N_CALL, 1),
        ])

        return np.array(obs, dtype=np.float32)

    def _get_priority_call(self) -> Optional[EmergencyCall]:
        """Return the highest-priority unassigned active call."""
        candidates = [c for c in self.calls if c.active and not c.assigned]
        if not candidates:
            return None
        # Sort by severity desc, then wait_time desc
        candidates.sort(key=lambda c: (c.severity, c.wait_time), reverse=True)
        return candidates[0]

    def _get_info(self) -> Dict[str, Any]:
        pending = sum(1 for c in self.calls if c.active and not c.assigned)
        available_ambs = sum(1 for a in self.ambulances if a.is_available())
        return {
            "step":             self._step_count,
            "traffic":          round(self._traffic, 3),
            "pending_calls":    pending,
            "available_ambs":   available_ambs,
            "episode_stats":    self._episode_stats.copy(),
            "task":             self.task,
        }

    def _render_ansi(self) -> str:
        lines = [f"╔═══ EMS Dispatch — Step {self._step_count}/{self.MAX_STEPS} ═══╗"]
        lines.append(f"  Traffic: {self._traffic:.2f}x")
        avail = sum(1 for a in self.ambulances if a.is_available())
        lines.append(f"  Ambulances: {avail}/{self.N_AMB} available")
        pending = sum(1 for c in self.calls if c.active and not c.assigned)
        lines.append(f"  Pending calls: {pending}")
        sev_icons = {1: "🟡", 2: "🟠", 3: "🔴"}
        for c in self.calls:
            if c.active and not c.assigned:
                lines.append(f"    {sev_icons[c.severity]} Call #{c.id}: sev={c.severity} wait={c.wait_time}s surv={c.survival_probability():.2f}")
        for h in self.hospitals:
            lines.append(f"  🏥 {h.name}: beds={h.available_beds}/{h.total_beds} icu={h.available_icu}/{h.total_icu}")
        stats = self._episode_stats
        lines.append(f"  ✅ Survived: {stats['patients_survived']}  ❌ Lost: {stats['patients_lost']}")
        lines.append("╚══════════════════════════════════════╝")
        return "\n".join(lines)
