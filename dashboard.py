import streamlit as st
import requests
import time
import os
import pandas as pd
import altair as alt

st.set_page_config(page_title="SolarEdge Dashboard", layout="wide")

NODES = [n for n in os.getenv("ALL_NODES", "").split(",") if n]

if "history_data" not in st.session_state:
    st.session_state.history_data = pd.DataFrame(columns=[
        "Tempo_Ore", "Potenza_Solare_W", "Reward_Cumulato", "Processati_Tot", "Droppati_Tot"
    ])

st.title("☀️ SolarEdge Multi-Node Dashboard")
placeholder = st.empty()

while True:
    cluster_data = []
    tot_processed = tot_dropped = tot_offloaded = tot_backlog = 0
    tot_reward = current_solar = 0.0
    current_sim_time = 0

    for node in NODES:
        try:
            res = requests.get(f"http://{node}:8000/metrics", timeout=0.5).json()
            res["Node"] = node.upper()
            cluster_data.append(res)
            
            tot_processed += res.get("processed", 0)
            tot_dropped += res.get("dropped", 0)
            tot_offloaded += res.get("offloaded", 0)
            tot_backlog += res.get("backlog", 0)
            tot_reward += res.get("reward", 0.0)
            
            current_solar = res.get("solar_power", current_solar)
            if current_sim_time == 0:
                current_sim_time = res.get("sim_time", 0)
        except:
            cluster_data.append({"Node": node.upper(), "state": "OFFLINE", "action": "OFFLINE", "battery_frac": 0.0, "backlog": 0})

    if current_sim_time > 0:
        t_hours = current_sim_time / 3600.0
        new_row = pd.DataFrame({
            "Tempo_Ore": [t_hours],
            "Potenza_Solare_W": [current_solar],
            "Reward_Cumulato": [tot_reward],
            "Processati_Tot": [tot_processed],
            "Droppati_Tot": [tot_dropped]
        })
        st.session_state.history_data = pd.concat([st.session_state.history_data, new_row], ignore_index=True)

    if len(st.session_state.history_data) > 5000:
        st.session_state.history_data = st.session_state.history_data.iloc[-5000:]

    with placeholder.container():
        sim_time_of_day = current_sim_time % 86400
        h = sim_time_of_day // 3600
        m = (sim_time_of_day % 3600) // 60
        s = sim_time_of_day % 60
        giorno = (current_sim_time // 86400) + 1
        
        st.markdown(f"### 🕒 Tempo Ambiente: **Giorno {giorno} - {int(h):02d}:{int(m):02d}:{int(s):02d}**")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("☀️ Potenza Solare", f"{current_solar:.2f} W")
        col2.metric("✅ Processati Totali", f"{tot_processed:,}")
        col3.metric("📥 Backlog Globale", f"{tot_backlog:,}")
        col4.metric("❌ Droppati Totali", f"{tot_dropped:,}")

        st.write("")
        st.write("")
        st.markdown("### 📈 Analisi del Cluster")

        df_nodes = pd.DataFrame(cluster_data)
        if not df_nodes.empty and "battery_frac" in df_nodes.columns:
            df_nodes["Batteria_%"] = df_nodes["battery_frac"].fillna(0) * 100

        # --- RIGA 1: Reward | Efficienza Cluster ---
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            if not st.session_state.history_data.empty:
                chart_reward = alt.Chart(st.session_state.history_data).mark_line(color='blue').encode(
                    x=alt.X('Tempo_Ore', title='Tempo (Ore)'),
                    y=alt.Y('Reward_Cumulato', title='Reward Cumulato'),
                ).properties(height=250, title="Apprendimento (Reward)")
                st.altair_chart(chart_reward, use_container_width=True)
            else:
                st.info("In attesa dei dati...")

        with r1c2:
            if not st.session_state.history_data.empty:
                chart_eff = alt.Chart(st.session_state.history_data).transform_fold(
                    ['Processati_Tot', 'Droppati_Tot'], as_=['Tipo', 'Valore']
                ).mark_line().encode(
                    x=alt.X('Tempo_Ore', title='Tempo (Ore)'),
                    y=alt.Y('Valore:Q', title='Frame'),
                    color=alt.Color('Tipo:N', scale=alt.Scale(domain=['Processati_Tot', 'Droppati_Tot'], range=['green', 'red']))
                ).properties(height=250, title="Efficienza Cluster")
                st.altair_chart(chart_eff, use_container_width=True)
            else:
                st.info("In attesa dei dati...")

        st.write("")

        # --- RIGA 2: Potenza Solare | Mappa Salute ---
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            if not st.session_state.history_data.empty:
                chart_solar = alt.Chart(st.session_state.history_data).mark_line(color='orange').encode(
                    x=alt.X('Tempo_Ore', title='Tempo (Ore)'),
                    y=alt.Y('Potenza_Solare_W', title='Potenza Solare (W)'),
                ).properties(height=250, title="Ciclo Solare")
                st.altair_chart(chart_solar, use_container_width=True)
            else:
                st.info("In attesa dei dati...")

        with r2c2:
            if not df_nodes.empty and "Batteria_%" in df_nodes.columns:
                scatter_health = alt.Chart(df_nodes).mark_circle(size=200).encode(
                    x=alt.X('Batteria_%:Q', scale=alt.Scale(domain=[0, 100]), title="Batteria %"),
                    y=alt.Y('backlog:Q', title="Backlog"),
                    color=alt.Color('Node:N', legend=None),
                    tooltip=['Node', 'Batteria_%', 'backlog', 'state']
                ).properties(height=250, title="Mappa Salute (In basso a dx = Ottimo)")
                st.altair_chart(scatter_health, use_container_width=True)
            else:
                st.info("In attesa dei dati...")

        st.write("")

        # --- RIGA 3: Batteria Nodi | Backlog Nodi ---
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            if not df_nodes.empty and "Batteria_%" in df_nodes.columns:
                bar_batt = alt.Chart(df_nodes).mark_bar().encode(
                    x=alt.X('Node:N', sort='-y', title=""),
                    y=alt.Y('Batteria_%:Q', scale=alt.Scale(domain=[0, 100]), title="Batteria %"),
                    color=alt.condition(alt.datum['Batteria_%'] < 20, alt.value('red'), alt.value('mediumseagreen'))
                ).properties(height=250, title="Livello Batteria per Nodo")
                st.altair_chart(bar_batt, use_container_width=True)
            else:
                st.info("In attesa dei dati...")

        with r3c2:
            if not df_nodes.empty and "backlog" in df_nodes.columns:
                bar_backlog = alt.Chart(df_nodes).mark_bar(color='steelblue').encode(
                    x=alt.X('Node:N', sort='-y', title=""),
                    y=alt.Y('backlog:Q', title="Frames in Coda")
                ).properties(height=250, title="Backlog per Nodo")
                st.altair_chart(bar_backlog, use_container_width=True)
            else:
                st.info("In attesa dei dati...")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- TABELLA STATUS ---
        st.markdown("### 🖥️ Dettaglio Nodi")
        if cluster_data:
            # FIX: Inizializza colonne mancanti se i nodi sono in avvio/offline
            expected_cols = ["state", "action", "Batteria_%", "backlog", "processed", "dropped", "offloaded", "reward"]
            for col in expected_cols:
                if col not in df_nodes.columns:
                    df_nodes[col] = "OFFLINE" if col in ["state", "action"] else 0.0

            df_view = df_nodes[["Node", "state", "action", "Batteria_%", "backlog", "processed", "dropped", "offloaded", "reward"]].copy()
            
            # Formattazione
            df_view["Batteria_%"] = pd.to_numeric(df_view["Batteria_%"]).apply(lambda x: f"{x:.1f}%")
            df_view["Reward"] = pd.to_numeric(df_view["reward"]).apply(lambda x: f"{x:.2f}")
            
            df_display = df_view[["Node", "state", "action", "Batteria_%", "backlog", "processed", "dropped", "offloaded", "Reward"]]
            df_display.columns = ["Nodo", "Stato", "Azione", "Batteria", "Backlog", "Processati", "Droppati", "Inviati (Off)", "Reward"]
            
            st.dataframe(df_display, use_container_width=True)

    time.sleep(0.5)