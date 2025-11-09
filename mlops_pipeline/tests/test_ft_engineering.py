import pytest
import pandas as pd
import numpy as np
from mlops_pipeline.src.ft_engineering import create_features, build_preprocessor

# 1) Test de create_features
def test_create_features():
    df = pd.DataFrame({
        "age": [30, 40],
        "thalach": [150, 160],
        "oldpeak": [1.0, 2.0]
    })
    df_out = create_features(df)
    assert "age_x_thalach" in df_out.columns
    assert "thalach_minus_age" in df_out.columns
    assert "oldpeak_ratio" in df_out.columns
    assert df_out["age_x_thalach"].iloc[0] == 30 * 150
    assert df_out["thalach_minus_age"].iloc[1] == 160 - 40
    # Ajuste: usar el mismo cálculo que create_features
    assert np.isclose(df_out["oldpeak_ratio"].iloc[0], 1.0 / (150 + 1e-6))
    assert np.isclose(df_out["oldpeak_ratio"].iloc[1], 2.0 / (160 + 1e-6))

# 2) Test de build_preprocessor 
def test_build_preprocessor():
    num_cols = ["num1", "num2"]
    cat_cols = ["cat1"]
    df = pd.DataFrame({
        "num1": [1, 2],
        "num2": [3, 4],
        "cat1": ["A", "B"]
    })
    preproc = build_preprocessor(num_cols, cat_cols, scaler=False)
    result = preproc.fit_transform(df)
    assert result.shape[1] >= 3  # columnas numéricas + one-hot
    assert not np.any(pd.isnull(result))  # no debe haber NaNs