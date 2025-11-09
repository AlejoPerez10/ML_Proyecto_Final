# test_model_monitoring.py
import pytest
import pandas as pd
import numpy as np
from mlops_pipeline.src import model_monitoring as mm
import tempfile
from pathlib import Path
import os

# 1) Test de psi con datos simples
def test_psi_simple():
    expected = pd.Series([1,2,3,4])
    actual = pd.Series([1,2,3,5])
    val = mm.psi(expected, actual, bins=2)
    assert isinstance(val, float)
    assert val >= 0

# 2) Test KS
def test_ks_test():
    a = pd.Series([1,2,3])
    b = pd.Series([1,2,4])
    stat, p = mm.ks_test(a, b)
    assert 0 <= stat <= 1
    assert 0 <= p <= 1

# 3) Test js_divergence
def test_js_divergence():
    a = pd.Series([1,2,3])
    b = pd.Series([1,2,4])
    val = mm.js_divergence(a, b, bins=2)
    assert isinstance(val, float)
    assert 0 <= val <= 1

# 4) Test chi_square_test con datos simples
def test_chi_square_test():
    a = pd.Series(["A","B","A","B"])
    b = pd.Series(["X","X","Y","Y"])
    chi, p = mm.chi_square_test(a, b)
    assert chi >= 0
    assert 0 <= p <= 1

# 5) Test monitor_data_drift con datasets de prueba
def test_monitor_data_drift(tmp_path):
    # Crear CSVs temporales
    train = pd.DataFrame({
        "num1":[1,2,3],
        "num2":[4,5,6],
        "cat1":["A","B","A"],
        "target":[0,1,0]
    })
    test = pd.DataFrame({
        "num1":[1,2,3],
        "num2":[4,5,7],
        "cat1":["A","B","B"],
        "target":[0,1,0]
    })
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)

    # Guardar preprocessor dummy
    from sklearn.preprocessing import StandardScaler
    import joblib
    preproc = StandardScaler()
    preproc.fit(train[["num1","num2"]])
    joblib.dump(preproc, data_dir / "pipeline_preprocessor.pkl")

    # Cambiar rutas dentro del módulo temporalmente
    mm.DATA_DIR = data_dir
    df_results = mm.monitor_data_drift()

    # Validar columnas clave
    assert "variable" in df_results.columns
    assert "alerta" in df_results.columns
    assert df_results.shape[0] > 0

    # Validar que se hayan creado archivos
    assert (data_dir / "results.csv").exists()
    history_files = list((data_dir / "results_history").glob("monitoring_*.csv"))
    assert len(history_files) > 0