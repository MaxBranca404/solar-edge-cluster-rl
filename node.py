import os, time, threading, requests, uvicorn, uuid
import matplotlib.pyplot as plt
import numpy as np
import json
from fastapi import FastAPI
from pydantic import BaseModel
from stable_baselines3 import PPO
from solar_env import SolarEdgeEnv

app = FastAPI()
NODE_ID = os.getenv("NODE_ID", "node0")
NEIGHBORS = [n for n in os.getenv("NEIGHBORS", "").split(",") if n]
MODEL_PATH = "ppo_solaredge_model.zip"
STEP_DELAY = float(os.getenv("STEP_DELAY", "1.0"))
SIM_SPEED = 1.0 / STEP_DELAY if STEP_DELAY > 0 else 1.0

# Lettura nuovi parametri Custom
SIM_DURATION_HOURS = float(os.getenv("SIM_DURATION_HOURS", "24.0"))
INIT_BATTERY_MODE = os.getenv("INIT_BATTERY_MODE", "100")
INIT_BACKLOG_MODE = os.getenv("INIT_BACKLOG_MODE", "0")

# --- OROLOGIO GLOBALE DEL CLUSTER ---
START_TIME = time.time()
START_SIM_T = 0
SIM_DURATION_STEPS = int(SIM_DURATION_HOURS * 3600)

env = SolarEdgeEnv(num_neighbors=len(NEIGHBORS))
obs, _ = env.reset()

env.t = START_SIM_T

# Setup Batteria Custom
if INIT_BATTERY_MODE == "100":
    env.battery_j = env.battery_capacity_j
elif INIT_BATTERY_MODE == "50":
    env.battery_j = env.battery_capacity_j * 0.5
elif INIT_BATTERY_MODE == "random":
    env.battery_j = np.random.uniform(0.1, 1.0) * env.battery_capacity_j
else:
    # Se arriva un numero specifico dall'orchestratore, lo usa
    try:
        env.battery_j = float(INIT_BATTERY_MODE)
    except ValueError:
        env.battery_j = env.battery_capacity_j

# Setup Backlog Custom
if INIT_BACKLOG_MODE == "0":
    env.backlog = 0
elif INIT_BACKLOG_MODE == "50":
    env.backlog = int(env.backlog_max * 0.5)
elif INIT_BACKLOG_MODE == "random":
    env.backlog = int(np.random.uniform(0.0, 0.8) * env.backlog_max)
else:
    # Se arriva un numero specifico dall'orchestratore, lo usa
    try:
        env.backlog = int(float(INIT_BACKLOG_MODE))
    except ValueError:
        env.backlog = 0

obs = env._get_obs()

model = PPO.load(MODEL_PATH)
env_lock = threading.Lock()
last_info = {}

received_packets = set()
packet_queue = []
sim_finished = False

TOPOLOGY_NAME = os.getenv("TOPOLOGY_NAME", "")
BASE_REPORT_DIR = f"risultati_topologia/{TOPOLOGY_NAME}" if TOPOLOGY_NAME else "reports"
os.makedirs(BASE_REPORT_DIR, exist_ok=True)

class OffloadData(BaseModel):
    packet_id: str
    frames: int

@app.get("/metrics")
def get_metrics():
    with env_lock:
        real_load_frac = min(env.backlog / env.backlog_max, 1.0)
        simulated_load = 0.2 + (real_load_frac * 0.55)

        return {
            "load": simulated_load, "battery_frac": env.battery_j / env.battery_capacity_j,
            "backlog": env.backlog, "processed": env.total_frames_processed,
            "dropped": env.total_frames_dropped, "offloaded": env.total_frames_offloaded,
            "received": env.total_received, "reward": env.cumulative_reward,
            "sim_time": env.t, "solar_power": last_info.get("solar_power", 0.0),
            "state": "FINISHED" if sim_finished else last_info.get("state", "ACTIVE"),
            "action": last_info.get("action", "STARTING")
        }

@app.post("/offload")
def receive_offload(data: OffloadData):
    if data.packet_id in received_packets:
        return {"status": "already_received"}

    received_packets.add(data.packet_id)
    packet_queue.append(data.packet_id)

    if len(packet_queue) > 500:
        old_pkt = packet_queue.pop(0)
        received_packets.remove(old_pkt)

    with env_lock:
        env.add_external_frames(data.frames)
    return {"status": "ok"}

def generate_24h_report(history, day):
    if len(history['time']) == 0: return
    print(f"[{NODE_ID}] Generazione report {day}...")

    time_hours = np.array(history['time']) / 3600.0
    num_neighbors = len(NEIGHBORS)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # --- A. Batteria ---
    axes[0].plot(time_hours, np.array(history['battery']) / 1000.0, linewidth=1.5, color='green', label='Batteria (kJ)')
    axes[0].set_ylabel('Batteria (kJ)', fontsize=11)
    axes[0].set_title(f'Report {NODE_ID.upper()} - Andamento Batteria', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    crit = env.critical_battery_j / 1000.0 if hasattr(env, 'critical_battery_j') else (env.E_M / 1000.0)
    axes[0].axhline(y=crit, color='r', linestyle='--', label='Critical Threshold (Est.)')
    axes[0].legend()

    # --- B. Backlog ---
    axes[1].plot(time_hours, history['backlog'], linewidth=1.5, color='blue')
    axes[1].set_ylabel('Backlog (frame)', fontsize=11)
    axes[1].set_title('Andamento Backlog', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # --- C. Solare ---
    axes[2].plot(time_hours, history['solar_power'], linewidth=1.5, color='orange')
    axes[2].set_ylabel('Potenza Solare (W)', fontsize=11)
    axes[2].set_title('Energia Solare', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].fill_between(time_hours, 0, history['solar_power'], alpha=0.3, color='orange')

    # --- D. Azioni (SMART) ---
    actions = np.array(history['action'])
    sleep_mask = np.array(history['sleep_mode'])
    real_targets = np.array(history.get('real_offload_target', np.full(len(actions), -1)))

    # STORE
    mask_store = (actions == 0) & (~sleep_mask)
    if mask_store.any():
        axes[3].scatter(time_hours[mask_store], actions[mask_store], label='STORE', color='gray', alpha=0.6, s=15)

    # PROCESS
    mask_proc = (actions == 1) & (~sleep_mask)
    if mask_proc.any():
        axes[3].scatter(time_hours[mask_proc], actions[mask_proc], label='PROCESS', color='blue', alpha=0.6, s=15)

    # OFFLOAD -> Vicini Reali
    colors = ['cyan', 'magenta', 'lime', 'orange', 'purple', 'brown', 'pink', 'olive']
    for i in range(num_neighbors):
        mask_off = (actions >= 2) & (real_targets == i) & (~sleep_mask)
        if mask_off.any():
            y_vals = np.full(mask_off.sum(), 2 + i)
            axes[3].scatter(time_hours[mask_off], y_vals, label=f'OFFLOAD -> N{i+1}', color=colors[i % len(colors)], alpha=0.8, s=20, marker='x')

    # SLEEP
    if sleep_mask.any():
        axes[3].fill_between(time_hours, -0.5, 2 + num_neighbors, where=sleep_mask, alpha=0.2, color='red', label='Sleep Mode')

    axes[3].set_ylabel('Azione / Destinazione', fontsize=11)
    axes[3].set_xlabel('Tempo Assoluto (ore)', fontsize=11)
    axes[3].set_title(f'Azioni Effettive (Routing su {num_neighbors} Vicini)', fontsize=12, fontweight='bold')

    y_ticks = [0, 1] + [2 + i for i in range(num_neighbors)]
    y_labels = ['STORE', 'PROCESS'] + [f'TO N{i+1}' for i in range(num_neighbors)]
    axes[3].set_yticks(y_ticks)
    axes[3].set_yticklabels(y_labels)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(f"{BASE_REPORT_DIR}/report_{NODE_ID}_day{day}.png")
    plt.close()

def agent_loop():
    global obs, last_info, sim_finished, START_TIME

    START_TIME = time.time()

    while True:
        if sim_finished:
            time.sleep(1)
            continue

        req_timeout = max(0.1, min(0.5, STEP_DELAY * 2))
        for idx, neighbor in enumerate(NEIGHBORS):
            try:
                resp = requests.get(f"http://{neighbor}:8000/metrics", timeout=req_timeout).json()
                with env_lock:
                    env.neighbor_loads[idx] = resp["load"]
                    env.neighbor_batteries[idx] = resp["battery_frac"]
            except: pass

        with env_lock:
            elapsed_real = time.time() - START_TIME
            current_global_sim_time = int(START_SIM_T + (elapsed_real * SIM_SPEED))

            if current_global_sim_time >= START_SIM_T + SIM_DURATION_STEPS:
                generate_24h_report(env.history, "FINAL")
                sim_finished = True
                print(f"[{NODE_ID}] Simulazione terminata. Report finale generato.")

                # --- NUOVA AGGIUNTA: Salvataggio metriche per il summary ---
                if TOPOLOGY_NAME:
                    final_stats = {
                        "node_id": NODE_ID,
                        "processed": env.total_frames_processed,
                        "dropped": env.total_frames_dropped,
                        "offloaded": env.total_frames_offloaded,
                        "received": env.total_received,
                        "reward": env.cumulative_reward,
                        "final_battery": float(env.battery_j),
                        "final_backlog": int(env.backlog)
                    }
                    with open(f"{BASE_REPORT_DIR}/.done_{NODE_ID}", "w") as f:
                        json.dump(final_stats, f)
                # ---------------------------------------------------------
                continue

            old_day = env.t // 86400
            new_day = current_global_sim_time // 86400
            if new_day > old_day and env.t > 0:
                generate_24h_report(env.history, f"intermedio_{old_day}")
                env._init_history()

            env.t = current_global_sim_time

            # Chiediamo all'agente di predire l'azione
            action, _ = model.predict(obs, deterministic=False)

            # --- OVERRIDE DI SICUREZZA ---
            # Se la batteria supera l'80%, forziamo l'azione a 1 (PROCESS)
            battery_percentage = env.battery_j / env.battery_capacity_j
            if battery_percentage > 0.80:
                final_action = 1
            else:
                final_action = action.item()
            # -----------------------------

            # Passiamo l'azione finale calcolata all'ambiente
            obs, _, _, _, info = env.step(final_action)
            last_info = info
            pending = info.get("pending_offload")

        if pending and pending["target_idx"] < len(NEIGHBORS):
            target = NEIGHBORS[pending["target_idx"]]
            frames_to_send = pending["frames"]
            packet_id = str(uuid.uuid4())

            retries = 0
            delivered = False

            while not delivered and retries < 3:
                try:
                    res = requests.post(
                        f"http://{target}:8000/offload",
                        json={"packet_id": packet_id, "frames": frames_to_send},
                        timeout=2.0
                    )
                    if res.status_code == 200: delivered = True
                except:
                    retries += 1
                    time.sleep(0.1)

            if not delivered:
                with env_lock:
                    env.backlog = min(env.backlog + frames_to_send, env.backlog_max)
                    env.total_frames_offloaded -= frames_to_send

        if STEP_DELAY > 0: time.sleep(STEP_DELAY)

@app.on_event("startup")
def startup():
    threading.Thread(target=agent_loop, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
