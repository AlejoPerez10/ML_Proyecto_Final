# ======================================================
# Monitoreo y detección de Data Drift
# ======================================================

"""
Para correrlo:
    1- Activa tu entorno virtual
    2- Ve a la carpeta src/
    3- Ejecuta:  python model_monitoring.py

Salidas (mlops_pipeline/src/data - /results_history):
    - results.csv
    - monitoring(fecha).csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import LabelEncoder
import os
import datetime as dt
import warnings
import joblib
from pathlib import Path
warnings.filterwarnings("ignore")

# ------------------------------------------------------
# 1. CARGA DE DATOS
# ------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

def load_data():
    baseline_raw = pd.read_csv(DATA_DIR / 'train.csv')
    current_raw  = pd.read_csv(DATA_DIR / 'test.csv')
    baseline_df = baseline_raw.copy()
    current_df  = current_raw.copy()
    return baseline_df, current_df, baseline_raw, current_raw

# ------------------------------------------------------
# 2. FUNCIONES DE MÉTRICAS
# ------------------------------------------------------
def get_hist_probs(arr, bins):
    hist, _ = np.histogram(arr, bins=bins)
    probs = hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist)
    probs = probs + 1e-8
    return probs / probs.sum()

def psi(expected, actual, bins=10):
    bins = np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1))
    exp_probs = get_hist_probs(expected, bins)
    act_probs = get_hist_probs(actual, bins)
    psi_value = np.sum((exp_probs - act_probs) * np.log(exp_probs / act_probs))
    return psi_value

def ks_test(expected, actual):
    ks_stat, p_value = stats.ks_2samp(expected, actual)
    return ks_stat, p_value

def js_divergence(expected, actual, bins=10):
    bins = np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1))
    e_hist = get_hist_probs(expected, bins)
    a_hist = get_hist_probs(actual, bins)
    return jensenshannon(e_hist, a_hist)

def chi_square_test(expected, actual):
    contingency = pd.crosstab(expected, actual)
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    return chi2, p

# ------------------------------------------------------
# 3. PROCESO DE MONITOREO
# ------------------------------------------------------
def monitor_data_drift():
    baseline_df, current_df, baseline_raw, current_raw = load_data()
    common_cols = [col for col in baseline_df.columns if col in current_df.columns]

    preprocessor_path = DATA_DIR / 'pipeline_preprocessor.pkl'
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        try:
            baseline_df = pd.DataFrame(preprocessor.transform(baseline_df), columns=preprocessor.get_feature_names_out())
            current_df = pd.DataFrame(preprocessor.transform(current_df), columns=preprocessor.get_feature_names_out())
            common_cols = baseline_df.columns
        except Exception as e:
            print(f"No se pudo aplicar el preprocesador: {e}")

    results = []

    for col in common_cols:
        try:
            if pd.api.types.is_numeric_dtype(baseline_df[col]):
                psi_val = psi(baseline_df[col].dropna(), current_df[col].dropna())
                ks_stat, ks_p = ks_test(baseline_df[col].dropna(), current_df[col].dropna())
                jsd_val = js_divergence(baseline_df[col].dropna(), current_df[col].dropna())
                results.append({
                    "variable": col,
                    "tipo": "numérica",
                    "PSI": round(psi_val, 3),
                    "KS_stat": round(ks_stat, 3),
                    "KS_p": round(ks_p, 3),
                    "JSD": round(jsd_val, 3),
                    "Chi2": None,
                    "Chi2_p": None
                })
            else:
                base_cat = baseline_raw[col].astype(str).fillna("<<NA>>")
                curr_cat = current_raw[col].astype(str).fillna("<<NA>>")

                le = LabelEncoder()
                le.fit(base_cat)
                base_enc = le.transform(base_cat)
                curr_mapped = [v if v in le.classes_ else "<<UNK>>" for v in curr_cat]
                if "<<UNK>>" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "<<UNK>>")
                curr_enc = le.transform(curr_mapped)

                cont = pd.crosstab(base_enc, curr_enc)
                if cont.shape[0] >= 2 and cont.shape[1] >= 2:
                    chi2_val, chi2_p = stats.chi2_contingency(cont)[:2]
                else:
                    chi2_val, chi2_p = None, None

                results.append({
                    "variable": col,
                    "tipo": "categórica",
                    "PSI": None,
                    "KS_stat": None,
                    "KS_p": None,
                    "JSD": None,
                    "Chi2": round(chi2_val, 3) if chi2_val is not None else None,
                    "Chi2_p": round(chi2_p, 3) if chi2_p is not None else None
                })
        except Exception as e:
            print(f"Error analizando {col}: {e}")

    df_results = pd.DataFrame(results)
    df_results["fecha_ejecucion"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Etiquetas PSI
    def flag_psi(x):
        if x is None or pd.isna(x): return "OK"
        if x > 0.25: return "⚠️ ALTO"
        elif x > 0.1: return "🟡 MODERADO"
        else: return "🟢 OK"
    df_results["alerta"] = df_results["PSI"].apply(flag_psi)

    # Recomendar retrain
    df_results["recomendar_retrain"] = df_results["PSI"].apply(lambda x: True if x is not None and x > 0.25 else False)

    # Guardar resultados
    DATA_DIR.mkdir(exist_ok=True)
    df_results.to_csv(DATA_DIR / "results.csv", index=False)

    # Guardar histórico con fecha
    (DATA_DIR / "results_history").mkdir(exist_ok=True)
    fecha = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df_results.to_csv(DATA_DIR / f"results_history/monitoring_{fecha}.csv", index=False)

    print("✅ Monitoreo completado. Resultados guardados en './data/results.csv' y carpeta /results_history/")
    return df_results

# ------------------------------------------------------
# 4. EJECUCIÓN
# ------------------------------------------------------
if __name__ == "__main__":
    df_metrics = monitor_data_drift()
    print(df_metrics.head())