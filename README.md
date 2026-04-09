# ☀️ SolarEdge Cluster — Simulatore Multi-Nodo con RL

Simulatore distribuito di una rete di nodi edge alimentati da energia solare. Ogni nodo è controllato da un agente PPO (Proximal Policy Optimization, via Stable-Baselines3) che impara a gestire autonomamente tre risorse:

- **Batteria** — energia raccolta da pannelli solari e consumata per elaborare/trasmettere frame
- **Backlog** — coda locale di frame in attesa di elaborazione
- **Rete** — capacità di fare offload dei frame verso i nodi vicini

I nodi comunicano tra loro tramite API REST (FastAPI) e possono essere monitorati in tempo reale su una dashboard Streamlit.

---

## 📁 Struttura del Progetto

```
.
├── Dockerfile              # Immagine base per ogni container (nodo + dashboard)
├── docker-compose.yml      # Generato automaticamente da start_cluster.py
├── start_cluster.py        # Entry point: configura e lancia il cluster
├── node.py                 # Logica del nodo: agente PPO + server FastAPI
├── solar_env.py            # Ambiente Gymnasium per l'agente RL
├── solar_model.py          # Modello fisico di produzione solare con nuvole
├── dashboard.py            # Dashboard Streamlit per il monitoraggio live
├── ppo_solaredge_model.zip # Modello PPO pre-addestrato (RICHIESTO, vedi sotto)
├── requirements.txt        # Dipendenze Python (per sviluppo/test locale)
├── reports/                # Report PNG generati dalle simulazioni normali
└── risultati_topologia/    # Report e summary delle simulazioni multi-topologia
    ├── lineare/
    ├── circolare/
    ├── griglia/
    └── interconnessa/
```

---

## ⚙️ Prerequisiti

### 1. Docker e Docker Compose
Il sistema è progettato per girare interamente su Docker. Assicurati di avere:

- **Docker Desktop** (Windows/macOS) oppure **Docker Engine + Docker Compose Plugin** (Linux)
- Versione Docker consigliata: ≥ 24.x
- Versione Docker Compose consigliata: ≥ 2.20

Verifica l'installazione:
```bash
docker --version
docker compose version
```

### 2. Python 3.10+ (solo per lo script di avvio locale)
`start_cluster.py` viene eseguito **sulla tua macchina**, non nel container. Richiede Python 3.10 o superiore.

```bash
python --version
# Python 3.10.x o superiore
```

### 3. Modello PPO pre-addestrato
Il file `ppo_solaredge_model.zip` **deve essere presente** nella root del progetto prima di avviare il cluster. Ogni container lo carica all'avvio tramite `PPO.load("ppo_solaredge_model.zip")`.

Se non disponi del modello pre-addestrato, dovrai addestrarlo con Stable-Baselines3 e salvarlo con:
```python
from stable_baselines3 import PPO
from solar_env import SolarEdgeEnv

env = SolarEdgeEnv(num_neighbors=2)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
model.save("ppo_solaredge_model")  # salva come ppo_solaredge_model.zip
```

---

## 🚀 Installazione e Avvio Rapido

### Step 1 — Clona e prepara il progetto

```bash
git clone https://github.com/MaxBranca404/solar-edge-cluster-rl
cd solaredge-cluster
```

Assicurati che nella cartella siano presenti tutti questi file:
```
Dockerfile
start_cluster.py
node.py
solar_env.py
solar_model.py
dashboard.py
ppo_solaredge_model.zip   ← obbligatorio
requirements.txt
```

### Step 2 — (Opzionale) Installa le dipendenze Python locali

Necessario solo per eseguire `start_cluster.py` e per sviluppo/debug locale. Non serve per la simulazione Docker.

```bash
# Crea un ambiente virtuale (consigliato)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Installa le dipendenze
pip install -r requirements.txt
```

### Step 3 — Avvia il cluster

```bash
python start_cluster.py
```

Lo script è interattivo e ti chiederà:

1. **Quanti nodi** vuoi avviare (minimo 2)
2. **La velocità** della simulazione (es. `100` = 100x, 1 ora simulata in ~36 secondi reali)
3. **La modalità** di simulazione (vedi sezione successiva)

---

## 🎮 Modalità di Simulazione

### Modalità 1 — Default
Configurazione standard: 24h simulate, batteria piena (100%), backlog vuoto.

```
Scelta: 1
```

Avvia il cluster con dashboard Streamlit accessibile su http://localhost:8501.

### Modalità 2 — Custom
Permette di personalizzare durata, stato iniziale della batteria e del backlog.

```
Scelta: 2
Durata simulazione in ore: 48
Batteria iniziale [100, 50, random]: random
Backlog iniziale [0, 50, random]: 50
```

Anche questa modalità avvia la dashboard su http://localhost:8501.

### Modalità 3 — Test Topologia (Automazione)
Esegue automaticamente la stessa simulazione su **4 topologie di rete** in sequenza, senza dashboard, e genera un `summary.txt` per ciascuna:

- `lineare` — ogni nodo comunica solo con il precedente e il successivo
- `circolare` — come lineare, ma il primo e l'ultimo nodo sono collegati tra loro
- `griglia` — nodi disposti su griglia 2D, ogni nodo ha al massimo 4 vicini
- `interconnessa` — ogni nodo comunica con tutti gli altri (fully connected)

Gli stati iniziali (batteria e backlog) vengono generati casualmente **una sola volta** e riutilizzati per tutte le topologie, garantendo un confronto equo.

```
Scelta: 3
```

I risultati vengono salvati in:
```
risultati_topologia/
├── lineare/
│   ├── report_node1_dayFINAL.png
│   ├── .done_node1    ← file di sincronizzazione interno
│   └── summary.txt    ← riepilogo delle performance
├── circolare/
│   └── ...
...
```

---

## 📊 Dashboard

Quando si usano le modalità 1 o 2, la dashboard Streamlit è accessibile su:

```
http://localhost:8501
```

Mostra in tempo reale:
- **Orologio simulato** (giorno e ora virtuale)
- **Potenza solare** attuale (W)
- **Frame processati / droppati** su tutto il cluster
- **Backlog globale**
- **Grafico reward** cumulato (curva di apprendimento)
- **Ciclo solare** (andamento nelle ultime ore)
- **Mappa salute** — scatter plot batteria vs backlog per ogni nodo
- **Barre batteria e backlog** per singolo nodo
- **Tabella dettaglio** con stato, azione corrente e metriche di ogni nodo

---

## 🗂️ Output Generati

### Report grafici (`.png`)
Generati al termine di ogni giornata simulata (o al termine della simulazione):
- **Andamento batteria** (kJ nel tempo con soglia critica)
- **Andamento backlog** (frame nel tempo)
- **Ciclo solare** (W con area riempita)
- **Azioni dell'agente** (scatter per azione, con destinazione offload distinta per vicino)

### Summary testuale (`summary.txt`) — solo Modalità 3
Generato automaticamente al termine di ogni topologia:

```
=== REPORT TOPOLOGIA: LINEARE ===
Nodi totali: 4

--- MAPPA DELLE CONNESSIONI ---
node1 comunica con -> node2
node2 comunica con -> node1, node3
...

--- PERFORMANCE GLOBALI CLUSTER ---
Frame Ricevuti dall'esterno: 691200
Frame Processati:            582410
Frame Droppati (Persi):      45210
Frame Offloadati (Rete):     63580
Reward Cumulato Globale:     -1204.33
Efficienza di Rete:          92.80%

--- DETTAGLIO PER NODO ---
[NODE1]
  - Processati:      148200
  ...
```

---

## 📄 Licenza

Progetto a scopo di ricerca e sperimentazione. Fare riferimento al paper originale per i dettagli sul modello energetico e l'algoritmo RL.
