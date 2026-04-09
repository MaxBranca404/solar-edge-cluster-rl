import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from solar_model import solar_power_realistic

# Parametri aggiornati fedelmente alla Table I e Section III del paper
MOD_1_PARAMS = {
    "battery_capacity": 160000.0,
    "idle_power": 2.5,          # P_0
    "sleep_power": 0.1,         # P_s
    "max_power": 6.0,           # P_M
    "epsilon_j": 0.1,           # Energia per processare 1 img (100 mJ)
    "delta_j": 0.005,           # Energia per tx/rx 1 img (5 mJ)
    "xi_j": 0.26,               # Coefficiente energia cooperazione (0.26 J)
    "lambda_rate": 8.0,         # Arrival rate
    "mu_rate": 10.0,            # Processing rate
    "r_rate": 100.0,            # Channel speed (img/s)
    "solar_peak_power": 20.0,
    "E_M": 3600.0,              # Safety margin energy (1 Wh = 3600 J)
    "E_A": 3600.0               # Active margin energy (1 Wh = 3600 J)
}

ACTION_STORE = 0
ACTION_PROCESS = 1

def solar_power_model(t_sec: float, peak_power: float) -> float:
    hours = (t_sec % 86400) / 3600.0
    if hours < 6.0 or hours > 18.0: return 0.0
    return np.sin(np.pi * (hours - 6.0) / 12.0) * peak_power

#def solar_power_model(t_sec: float, peak_power: float) -> float:
#    """
#    Wrapper compatibile con la funzione originale.
#    Limita sempre la potenza a max 20W.
#    """
#
#    # Limite massimo fisso
#    peak = min(peak_power, 20.0)
#
#    power = solar_power_realistic(t_sec, peak)
#
#    # Sicurezza extra
#    return min(power, 20.0)

class SolarEdgeEnv(gym.Env):
    def __init__(self, num_neighbors=2, seed=None):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.sim_seconds = 86400
        self.num_neighbors = num_neighbors

        self.avg_arrival_rate = MOD_1_PARAMS["lambda_rate"]
        self.battery_capacity_j = MOD_1_PARAMS["battery_capacity"]
        self.P0 = MOD_1_PARAMS["idle_power"]
        self.sleep_power = MOD_1_PARAMS["sleep_power"]
        self.P_M = MOD_1_PARAMS["max_power"]

        self.epsilon_j = MOD_1_PARAMS["epsilon_j"]
        self.delta_j = MOD_1_PARAMS["delta_j"]
        self.xi_j = MOD_1_PARAMS["xi_j"]

        self.mu = MOD_1_PARAMS["mu_rate"]
        self.r = MOD_1_PARAMS["r_rate"]

        self.E_M = MOD_1_PARAMS["E_M"]
        self.E_A = MOD_1_PARAMS["E_A"]

        self.solar_peak_power = MOD_1_PARAMS["solar_peak_power"]
        self.backlog_max = 120000

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.t = 0
        self.last_t = 0
        self.battery_j = self.battery_capacity_j
        self.backlog = 0
        self.sleep_mode = False
        self.is_dead = False
        self.neighbor_loads = np.zeros(self.num_neighbors, dtype=np.float32)
        self.neighbor_batteries = np.zeros(self.num_neighbors, dtype=np.float32)

        self.total_frames_arrived = 0
        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        self.total_frames_offloaded = 0
        self.total_received = 0
        self.cumulative_reward = 0.0
        self.sleep_steps = 0
        self.pending_offload = None
        self._init_history()

    def _init_history(self):
        self.history = {
            'time': [], 'battery': [], 'backlog': [], 'action': [],
            'solar_power': [], 'sleep_mode': [], 'is_dead': [],
            'processed_now': [], 'dropped_now': [], 'real_offload_target': []
        }

    def _e_C(self, b: float) -> float:
        return self.E_M + (self.xi_j * b)

    def _e_A(self, b: float) -> float:
        return self.E_A + self._e_C(b)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: self.rng = np.random.RandomState(seed)
        self.t = 0
        self.last_t = 0
        self.battery_j = self.battery_capacity_j
        self.backlog = 0
        self.sleep_mode = False
        self.is_dead = False
        self.pending_offload = None
        self._init_history()
        return self._get_obs(), {}

    def add_external_frames(self, frames: int):
        self.total_received += frames
        if not self.sleep_mode and not self.is_dead:
            new_backlog = self.backlog + frames
            if new_backlog > self.backlog_max:
                overflow = new_backlog - self.backlog_max
                self.backlog = self.backlog_max
                self.total_frames_dropped += overflow
            else:
                self.backlog = new_backlog
        else:
            self.total_frames_dropped += frames

    def step(self, action: int):
        delta_t = int(self.t - self.last_t)
        if delta_t > 3600 or delta_t <= 0:
            delta_t = 1

        self.last_t = self.t

        arrivals = self.rng.poisson(self.avg_arrival_rate * delta_t)
        self.total_frames_arrived += arrivals
        dropped = 0

        solar_power_w = solar_power_model(self.t, self.solar_peak_power)
        harvest_j = solar_power_w * delta_t
        self.battery_j = min(self.battery_j + harvest_j, self.battery_capacity_j)

        current_action = action
        executed_action = -1
        consumption = 0.0
        processed_now = 0
        offloaded_now = 0
        offload_target = -1
        self.pending_offload = None

        if self.battery_j <= 0.001:
            self.is_dead = True
            self.battery_j = 0.0
            executed_action = -99

        else:
            self.is_dead = False

            if self.sleep_mode:
                if self.battery_j >= self._e_A(self.backlog):
                    self.sleep_mode = False
            else:
                if self.battery_j < self._e_C(self.backlog):
                    self.sleep_mode = True

            if self.sleep_mode:
                self.sleep_steps += delta_t
                consumption = self.sleep_power * delta_t
                executed_action = -1
                dropped = arrivals
                self.total_frames_dropped += arrivals

            else:
                self.backlog += arrivals
                if self.backlog > self.backlog_max:
                    overflow = self.backlog - self.backlog_max
                    self.backlog = self.backlog_max
                    dropped += overflow
                    self.total_frames_dropped += overflow

                consumption = self.P0 * delta_t
                executed_action = current_action
                max_variable_energy = (self.P_M - self.P0) * delta_t

                if current_action == ACTION_STORE:
                    pass

                elif current_action == ACTION_PROCESS:
                    if self.backlog > 0:
                        max_proc_rate = int(self.mu * delta_t)
                        max_proc_energy = int(max_variable_energy / self.epsilon_j) if self.epsilon_j > 0 else float('inf')

                        frames_to_process = min(max_proc_rate, max_proc_energy, self.backlog)
                        proc_cost = self.epsilon_j * frames_to_process

                        if (self.battery_j - consumption - proc_cost) >= 0:
                            consumption += proc_cost
                            self.backlog -= frames_to_process
                            processed_now = frames_to_process
                            self.total_frames_processed += frames_to_process

                else:
                    # TENTATIVO DI OFFLOAD
                    offload_success = False
                    neighbor_idx = self._get_best_neighbor()

                    if 0 <= neighbor_idx < self.num_neighbors:
                        neighbor_load = self.neighbor_loads[neighbor_idx]
                        if neighbor_load < 0.8 and self.backlog > 0:
                            max_off_rate = int(self.r * delta_t)
                            max_off_energy = int(max_variable_energy / self.delta_j) if self.delta_j > 0 else float('inf')

                            frames_to_offload = min(max_off_rate, max_off_energy, self.backlog)
                            tx_cost = self.delta_j * frames_to_offload

                            if (self.battery_j - consumption - tx_cost) >= 0:
                                consumption += tx_cost
                                self.backlog -= frames_to_offload
                                offloaded_now = frames_to_offload
                                self.total_frames_offloaded += frames_to_offload
                                self.pending_offload = {"target_idx": neighbor_idx, "frames": frames_to_offload}
                                offload_target = neighbor_idx
                                offload_success = True

                    # --- INIZIO MECCANISMO DI FALLBACK ---
                    # Se l'offload non è andato a buon fine (vicino pieno o non trovato), proviamo a processare localmente
                    if not offload_success and self.backlog > 0:
                        max_proc_rate = int(self.mu * delta_t)
                        max_proc_energy = int(max_variable_energy / self.epsilon_j) if self.epsilon_j > 0 else float('inf')

                        frames_to_process = min(max_proc_rate, max_proc_energy, self.backlog)
                        proc_cost = self.epsilon_j * frames_to_process

                        if (self.battery_j - consumption - proc_cost) >= 0:
                            consumption += proc_cost
                            self.backlog -= frames_to_process
                            processed_now = frames_to_process
                            self.total_frames_processed += frames_to_process
                    # --- FINE MECCANISMO DI FALLBACK ---

        self.battery_j -= consumption
        if self.battery_j < 0:
            self.battery_j = 0.0
            self.is_dead = True

        r_work = (processed_now * 1.0) + (offloaded_now * 0.1)

        r_lazy = 0.0
        if offloaded_now > 0 and self.battery_j >= self._e_A(self.backlog):
            r_lazy = -0.5 * offloaded_now

        r_drop = -1.0 * dropped
        soc = self.battery_j / self.battery_capacity_j
        r_energy_anxiety = -20.0 * ((1.0 - soc) ** 4) * delta_t
        backlog_frac = self.backlog / self.backlog_max
        r_backlog = -2.0 * (backlog_frac ** 2) * delta_t

        r_critical = 0.0
        if self.is_dead: r_critical -= 100.0 * delta_t
        elif self.sleep_mode: r_critical -= 1.0 * delta_t

        reward = (r_work + r_lazy + r_drop + r_energy_anxiety + r_backlog + r_critical) * 0.1
        self.cumulative_reward += reward

        self.history['time'].append(self.t)
        self.history['battery'].append(self.battery_j)
        self.history['backlog'].append(self.backlog)
        self.history['action'].append(executed_action)
        self.history['solar_power'].append(solar_power_w)
        self.history['sleep_mode'].append(self.sleep_mode)
        self.history['is_dead'].append(self.is_dead)
        self.history['processed_now'].append(processed_now)
        self.history['dropped_now'].append(dropped)
        self.history['real_offload_target'].append(offload_target)

        if delta_t == 1 and self.t == self.last_t:
            self.t += 1

        info = {
            "pending_offload": self.pending_offload,
            "solar_power": solar_power_w,
            "battery_j": self.battery_j,
            "backlog": self.backlog,
            "processed": self.total_frames_processed,
            "dropped": self.total_frames_dropped,
            "offloaded": self.total_frames_offloaded,
            "state": "DEAD" if self.is_dead else "SLEEP" if self.sleep_mode else "ACTIVE",
            "action": "DEAD" if executed_action == -99 else "SLEEP" if executed_action == -1 else "STORE" if executed_action == 0 else "PROCESS" if executed_action == 1 else "OFFLOAD"
        }
        return self._get_obs(), reward, False, False, info

    def _get_best_neighbor(self) -> int:
        best_idx, best_score = -1, -1.0
        for i in range(self.num_neighbors):
            bat = self.neighbor_batteries[i]
            load = self.neighbor_loads[i]
            safe_load = max(load, 0.001)
            score = bat / safe_load
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _get_obs(self) -> np.ndarray:
        battery_frac = self.battery_j / self.battery_capacity_j
        backlog_frac = self.backlog / self.backlog_max
        hour_of_day = (self.t % 86400) / 3600.0
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
        solar_power = solar_power_model(self.t, self.solar_peak_power)

        sorted_loads = sorted(self.neighbor_loads)
        n1 = sorted_loads[0] if len(sorted_loads) > 0 else 0.0
        n2 = sorted_loads[1] if len(sorted_loads) > 1 else 0.0

        obs = np.array([
            battery_frac, backlog_frac,
            (hour_sin + 1) / 2, (hour_cos + 1) / 2,
            solar_power / self.solar_peak_power,
            n1, n2
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)
