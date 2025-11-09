# test_model_training_evaluation.py
import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from mlops_pipeline.src.model_training_evaluation import build_model, summarize_classification, run
import tempfile
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

# 1) Test build_model
def test_build_model():
    X = np.array([[1,2],[3,4]])
    y = np.array([0,1])
    model = LogisticRegression()
    trained = build_model(model, X, y)
    assert trained.coef_.shape[1] == X.shape[1]

# 2) Test summarize_classification
def test_summarize_classification():
    X = np.array([[1,2],[3,4]])
    y = np.array([0,1])
    model = LogisticRegression()
    model.fit(X, y)
    metrics = summarize_classification(model, X, y)
    assert 'accuracy' in metrics
    assert 'f1_score' in metrics
    assert 'roc_auc' in metrics
    assert 0 <= metrics['accuracy'] <= 1

# 3) Test run con dataset pequeño y preprocessor simulado
def test_run(tmp_path):
    # Crear CSV de entrenamiento y prueba
    df = pd.DataFrame({
        'num1': [1,2,3,4],
        'num2': [4,3,2,1],
        'target': [0,1,0,1]
    })
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    df.to_csv(train_path, index=False)
    df.to_csv(test_path, index=False)

    # Crear un preprocessor simulado y guardarlo
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    preproc = Pipeline([('scaler', StandardScaler())])
    preproc.fit(df[['num1','num2']])
    joblib.dump(preproc, tmp_path / "pipeline_preprocessor.pkl")

    # Ejecutar run
    run(train_path, test_path, out_dir=tmp_path)

    # Verificar archivos generados
    assert (tmp_path / "best_model.pkl").exists()
    assert (tmp_path / "model_metrics.csv").exists()
    assert (tmp_path / "model_comparison.png").exists()