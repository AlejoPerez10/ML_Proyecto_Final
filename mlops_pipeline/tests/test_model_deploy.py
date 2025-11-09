# test_model_deploy.py
import pytest
from fastapi.testclient import TestClient
from mlops_pipeline.src.model_deploy import app
import pandas as pd
import io

client = TestClient(app)

# 1) Test GET root
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"mensaje": "API funcionando correctamente"}

# 2) Test POST predict con CSV válido
def test_predict_valid_csv(tmp_path):
    # Crear un CSV de ejemplo
    df = pd.DataFrame({
        "age": [30, 40],
        "thalach": [150, 160],
        "oldpeak": [1.0, 2.0],
        "trestbps": [120, 130],
        "chol": [200, 210],
        "sex": [1, 0],
        "cp": [3, 2],
        "fbs": [0, 1],
        "restecg": [1, 0],
        "exang": [0, 1],
        "thal": [2, 3],
        "slope": [2, 1],
        "ca": [0, 1],
        "target": [1, 0]
    })
    csv_bytes = io.BytesIO()
    df.to_csv(csv_bytes, index=False)
    csv_bytes.seek(0)

    response = client.post("/predict", files={"file": ("test.csv", csv_bytes, "text/csv")})
    assert response.status_code == 200
    data = response.json()
    assert "predicciones" in data
    assert len(data["predicciones"]) == len(df)

# 3) Test POST predict con CSV inválido
def test_predict_invalid_csv():
    response = client.post("/predict", files={"file": ("invalid.csv", io.BytesIO(b"no,colums,here"), "text/csv")})
    assert response.status_code == 400
