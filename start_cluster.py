import os
import time
import math
import subprocess
import json
import random

def get_topology(n, topo_type):
    """Calcola i vicini per ogni nodo in base alla topologia scelta."""
    nodes = [f"node{i}" for i in range(1, n+1)]
    topology = {node: [] for node in nodes}

    if topo_type == "isolata":
        pass # Nessun vicino

    elif topo_type == "interconnessa":
        for node in nodes:
            topology[node] = [other for other in nodes if other != node]

    elif topo_type == "lineare":
        for i in range(n):
            if i > 0: topology[nodes[i]].append(nodes[i-1])
            if i < n-1: topology[nodes[i]].append(nodes[i+1])

    elif topo_type == "circolare":
        if n > 1:
            for i in range(n):
                topology[nodes[i]].append(nodes[(i-1)%n])
                topology[nodes[i]].append(nodes[(i+1)%n])

    elif topo_type == "griglia":
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        for i in range(n):
            r, c = i // cols, i % cols
            if r > 0: topology[nodes[i]].append(nodes[(r-1)*cols + c])
            if r < rows - 1 and (r+1)*cols + c < n: topology[nodes[i]].append(nodes[(r+1)*cols + c])
            if c > 0: topology[nodes[i]].append(nodes[r*cols + c - 1])
            if c < cols - 1 and r*cols + c + 1 < n: topology[nodes[i]].append(nodes[r*cols + c + 1])

    # Rimuovi eventuali duplicati (es. in reti circolari a 2 nodi)
    for k in topology:
        topology[k] = list(set(topology[k]))

    return topology

def generate_compose(topo_dict, step_delay, duration_h, bat_configs, bkl_configs, topology_name=None):
    """Genera il docker-compose.yml accettando configurazioni specifiche per ogni nodo."""
    compose_content = "services:\n"
    all_nodes = list(topo_dict.keys())

    # Dashboard solo se non siamo in modalità test topologia
    if not topology_name:
        compose_content += f"""
  dashboard:
    build: .
    command: streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - ALL_NODES={','.join(all_nodes)}
"""

    # Generazione Nodi
    for node_name, neighbors in topo_dict.items():
        volume_mount = "./risultati_topologia:/app/risultati_topologia" if topology_name else "./reports:/app/reports"

        # Se riceve un dizionario (Mode 3), prende il valore del singolo nodo, altrimenti usa la stringa generica (Mode 1 e 2)
        bat_val = bat_configs[node_name] if isinstance(bat_configs, dict) else bat_configs
        bkl_val = bkl_configs[node_name] if isinstance(bkl_configs, dict) else bkl_configs

        compose_content += f"""
  {node_name}:
    build: .
    command: python node.py
    volumes:
      - {volume_mount}
    environment:
      - NODE_ID={node_name}
      - NEIGHBORS={','.join(neighbors)}
      - STEP_DELAY={step_delay}
      - SIM_DURATION_HOURS={duration_h}
      - INIT_BATTERY_MODE={bat_val}
      - INIT_BACKLOG_MODE={bkl_val}
      - TOPOLOGY_NAME={topology_name if topology_name else ""}
"""
    return compose_content

if __name__ == "__main__":
    print("="*50)
    print("🚀 CONFIGURAZIONE SOLAREDGE CLUSTER MULTI-NODO")
    print("="*50)

    while True:
        try:
            n = int(input("Quanti nodi vuoi avviare? (Es. 4): "))
            if n > 1: break
        except: pass

    while True:
        try:
            speed = float(input("\nA che velocità vuoi la simulazione? (es. 100 per 100x): "))
            if speed > 0: break
        except: pass

    step_delay = 1.0 / speed

    print("\n--- MODALITÀ DI SIMULAZIONE ---")
    print("1) DEFAULT (24h, Batteria 100%, Backlog 0)")
    print("2) CUSTOM (Durata, Batteria e Backlog personalizzati)")
    print("3) TEST TOPOLOGIA (Automazione multi-topologia senza dashboard)")

    mode = input("Scelta (1, 2 o 3): ").strip()

    if mode == "3":
        topologies = ["lineare", "circolare", "griglia", "interconnessa"]
        duration_h = 24.0

        print("\n⏳ Preparazione della suite di test topologici...")

        # --- NOVITÀ: Generiamo gli stati iniziali casuali UNA SOLA VOLTA ---
        print("\n🎲 Generazione degli stati iniziali randomici (fissi per tutti i test):")
        bat_iniziali = {}
        bkl_iniziali = {}
        for i in range(1, n + 1):
            node_name = f"node{i}"
            # Batteria tra 10% e 100% di 160000.0 J
            bat_iniziali[node_name] = str(random.uniform(0.1, 1.0) * 160000.0)
            # Backlog tra 0% e 80% di 120000 frame
            bkl_iniziali[node_name] = str(int(random.uniform(0.0, 0.8) * 120000))

            bat_perc = (float(bat_iniziali[node_name]) / 160000.0) * 100
            print(f"  - {node_name}: Batteria iniziale {bat_perc:.1f}% | Backlog iniziale {bkl_iniziali[node_name]} frames")
        # -------------------------------------------------------------------

        for topo in topologies:
            print(f"\n" + "-"*40)
            print(f"🔄 AVVIO TEST TOPOLOGIA: {topo.upper()}")
            print("-" * 40)

            topo_dict = get_topology(n, topo)

            # Passiamo i dizionari appena creati invece della stringa "random"
            compose_content = generate_compose(topo_dict, step_delay, duration_h, bat_iniziali, bkl_iniziali, topology_name=topo)

            # ... (il resto del codice rimane identico da qui in poi) ...
            with open("docker-compose.yml", "w") as f:
                f.write(compose_content)

            target_dir = f"risultati_topologia/{topo}"
            os.makedirs(target_dir, exist_ok=True)

            for file in os.listdir(target_dir):
                if file.startswith(".done_"):
                    os.remove(os.path.join(target_dir, file))

            print("Avvio container...")
            subprocess.run(["docker", "compose", "up", "--build", "-d"])

            print(f"Simulazione in corso per la topologia {topo}... attendo i report.")
            while True:
                done_files = [f for f in os.listdir(target_dir) if f.startswith(".done_")]
                if len(done_files) == n:
                    break
                time.sleep(2)

            # --- NUOVA AGGIUNTA: GENERAZIONE SUMMARY.TXT ---
            print("Tutti i nodi hanno terminato. Generazione del summary.txt...")

            total_processed = total_dropped = total_offloaded = total_received = 0
            total_reward = 0.0
            node_stats = []

            for done_file in done_files:
                with open(os.path.join(target_dir, done_file), "r") as f:
                    try:
                        stats = json.load(f)
                        node_stats.append(stats)
                        total_processed += stats.get("processed", 0)
                        total_dropped += stats.get("dropped", 0)
                        total_offloaded += stats.get("offloaded", 0)
                        total_received += stats.get("received", 0)
                        total_reward += stats.get("reward", 0.0)
                    except json.JSONDecodeError:
                        pass # Evita crash se il file non è ancora ben scritto

            summary_lines = [
                f"=== REPORT TOPOLOGIA: {topo.upper()} ===",
                f"Nodi totali: {n}",
                "\n--- MAPPA DELLE CONNESSIONI ---"
            ]
            for node, neighbors in topo_dict.items():
                summary_lines.append(f"{node} comunica con -> {', '.join(neighbors) if neighbors else 'Nessuno (Isolato)'}")

            summary_lines.append("\n--- PERFORMANCE GLOBALI CLUSTER ---")
            summary_lines.append(f"Frame Ricevuti dall'esterno: {total_received}")
            summary_lines.append(f"Frame Processati:            {total_processed}")
            summary_lines.append(f"Frame Droppati (Persi):      {total_dropped}")
            summary_lines.append(f"Frame Offloadati (Rete):     {total_offloaded}")
            summary_lines.append(f"Reward Cumulato Globale:     {total_reward:.2f}")

            eff = (total_processed / (total_processed + total_dropped) * 100) if (total_processed + total_dropped) > 0 else 0
            summary_lines.append(f"Efficienza di Rete:          {eff:.2f}% (Processati / Totali)")

            summary_lines.append("\n--- DETTAGLIO PER NODO ---")
            node_stats.sort(key=lambda x: x["node_id"])
            for stat in node_stats:
                summary_lines.append(f"\n[{stat['node_id'].upper()}]")
                summary_lines.append(f"  - Processati:      {stat['processed']}")
                summary_lines.append(f"  - Droppati:        {stat['dropped']}")
                summary_lines.append(f"  - Offloadati:      {stat['offloaded']}")
                summary_lines.append(f"  - Reward Cumulato: {stat['reward']:.2f}")
                summary_lines.append(f"  - Batteria Finale: {stat['final_battery']:.2f} J")
                summary_lines.append(f"  - Backlog Finale:  {stat['final_backlog']} frames")

            with open(os.path.join(target_dir, "summary.txt"), "w") as f:
                f.write("\n".join(summary_lines))
            # -----------------------------------------------

            print(f"✅ Topologia {topo} completata! Summary generato in {target_dir}/summary.txt")
            print("Spegnimento container...")
            subprocess.run(["docker", "compose", "down"])

        print("\n🎉 TUTTI I TEST TOPOLOGIA COMPLETATI CON SUCCESSO!")

    else:
        # Modalità 1 e 2 classiche
        if mode == "2":
            duration_h = float(input("Durata simulazione in ore (es. 48): "))
            bat_mode = input("Batteria iniziale [100, 50, random]: ").strip().lower()
            bkl_mode = input("Backlog iniziale [0, 50, random]: ").strip().lower()
        else:
            duration_h = 24.0
            bat_mode = "100"
            bkl_mode = "0"

        # Per la modalità standard usiamo la topologia interconnessa
        topo_dict = get_topology(n, "interconnessa")
        compose_content = generate_compose(topo_dict, step_delay, duration_h, bat_mode, bkl_mode)

        with open("docker-compose.yml", "w") as f:
            f.write(compose_content)

        print("\n✅ docker-compose.yml generato con successo!")
        print("Avvio il cluster con dashboard...")
        subprocess.run(["docker", "compose", "up", "--build"])
