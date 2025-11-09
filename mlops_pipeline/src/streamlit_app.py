# ================
# streamlit_app.py
# ================

"""
Streamlit app para visualizar resultados de model monitoring:
    - Lee: ./data/results.csv  ./data/train.csv  ./data/test.csv  
    - También ./data/results_history/monitoring_<fecha>.csv.
    - Obtiene predicciones usando FastAPI.

Ejecutar: 
    - streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
from glob import glob
import requests
import io

st.set_page_config(page_title="Model Monitoring", layout="wide")

DATA_DIR = "./data"
RESULTS_PATH = os.path.join(DATA_DIR, "results.csv")
HISTORY_DIR = os.path.join(DATA_DIR, "results_history")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# URL de la API FastAPI
API_URL = "http://host.docker.internal:8000"  # Contenedor FastAPI

st.title("📊 Model Monitoring Dashboard")

# --- 1) Cargar resultados y datasets ---
@st.cache_data
def load_results():
    if os.path.exists(RESULTS_PATH):
        return pd.read_csv(RESULTS_PATH)
    return pd.DataFrame()

@st.cache_data
def load_history():
    files = sorted(glob(os.path.join(HISTORY_DIR, "monitoring_*.csv")), reverse=True)
    if not files:
        return None
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["origen_archivo"] = os.path.basename(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

@st.cache_data
def load_train_test():
    train = pd.read_csv(TRAIN_PATH) if os.path.exists(TRAIN_PATH) else pd.DataFrame()
    test = pd.read_csv(TEST_PATH) if os.path.exists(TEST_PATH) else pd.DataFrame()
    return train, test

results = load_results()
history = load_history()
train_df, test_df = load_train_test()

if results.empty:
    st.warning("⚠️ No se encontró './data/results.csv'. Ejecuta model_monitoring.py primero.")
    st.stop()

# --- 2) Mostrar alertas globales ---
alert_count = (results["alerta"].isin(["🟡 MODERADO", "⚠️ ALTO"])).sum()
if alert_count > 0:
    st.error(f"⚠️ Se detectaron {alert_count} variables con drift significativo. Considera reentrenar el modelo.")
else:
    st.success("🟢 No se detectaron alertas de drift importantes.")

# --- 3) Tabla principal con semáforo ---
st.subheader("📋 Resultados recientes de monitoreo")
st.dataframe(results, use_container_width=True)

# Botón de descarga
csv = results.to_csv(index=False).encode('utf-8')
st.download_button("📥 Descargar results.csv", data=csv, file_name="results.csv", mime="text/csv")

# --- 4) Histórico (si existe) ---
if history is not None:
    st.subheader("📈 Histórico de monitoreos")
    last_dates = history["fecha_ejecucion"].unique()
    sel_date = st.selectbox("Selecciona fecha de monitoreo:", last_dates)
    df_sel = history[history["fecha_ejecucion"] == sel_date]
    st.dataframe(df_sel, use_container_width=True)
else:
    st.info("No hay históricos en './data/results_history/'. Ejecuta varios monitoreos para ver evolución.")

# --- 5) Selector lateral de variable ---
st.sidebar.header("🔍 Análisis por variable")
var_list = results["variable"].tolist()
sel_var = st.sidebar.selectbox("Selecciona variable", var_list)

# Mostrar detalles
row = results[results["variable"] == sel_var].iloc[0]
st.markdown(f"### Variable `{sel_var}` — Tipo: {row['tipo']} — Alerta: {row.get('alerta','')}")
st.write(row.to_frame().T)

# --- 6) Distribuciones baseline vs current ---
st.subheader("Distribución: baseline (train) vs current (test)")

def find_column_in_dfs(name, train, test):
    if name in train.columns and name in test.columns:
        return name
    if "__" in name:
        cand = name.split("__", 1)[1]
        if cand in train.columns and cand in test.columns:
            return cand
    return None

col_name = find_column_in_dfs(sel_var, train_df, test_df)

if col_name:
    if pd.api.types.is_numeric_dtype(train_df[col_name]):
        # numérica → densidad
        train_plot = pd.DataFrame({"value": train_df[col_name], "dataset": "train"})
        test_plot = pd.DataFrame({"value": test_df[col_name], "dataset": "test"})
        plot_df = pd.concat([train_plot, test_plot])
        chart = alt.Chart(plot_df).transform_density(
            "value",
            as_=["value", "density"],
            groupby=["dataset"]
        ).mark_area(opacity=0.5).encode(
            x="value:Q",
            y="density:Q",
            color="dataset:N"
        ).properties(width=700, height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        # categórica → barras
        train_counts = train_df[col_name].value_counts().reset_index()
        test_counts = test_df[col_name].value_counts().reset_index()
        train_counts["dataset"] = "train"
        test_counts["dataset"] = "test"
        train_counts.columns = ["category", "count", "dataset"]
        test_counts.columns = ["category", "count", "dataset"]
        plot_df = pd.concat([train_counts, test_counts])
        chart = alt.Chart(plot_df).mark_bar().encode(
            x="category:N", y="count:Q", color="dataset:N", column="dataset:N"
        )
        st.altair_chart(chart, use_container_width=True)
else:
    st.info("No se encontró la variable original.")

# --- 7) Obtener predicciones desde FastAPI ---
st.subheader("📊 Predicciones usando FastAPI")
uploaded_file = st.file_uploader("Sube un CSV para obtener predicciones", type="csv")
if uploaded_file is not None:
    try:
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{API_URL}/predict", files={"file": uploaded_file})
        if response.status_code == 200:
            preds = response.json()["predicciones"]
            st.success("✅ Predicciones obtenidas correctamente")
            st.dataframe(pd.DataFrame({"Predicciones": preds}))
        else:
            st.error(f"Error API: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error conectando con la API: {e}")

# --- 8) Nota final ---
st.markdown("---")
st.markdown("💡 **Sugerencia:** si observas múltiples variables con PSI > 0.25 o KS p < 0.05, considera reentrenar el modelo.")