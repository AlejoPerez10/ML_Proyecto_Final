# model_deploy.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
import io
from pathlib import Path
from mlops_pipeline.src.ft_engineering import create_features

app = FastAPI(title="Modelo Predictivo API")

# Rutas absolutas relativas al script
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Cargar modelo y preprocesador
PREP = joblib.load(DATA_DIR / "pipeline_preprocessor.pkl")
MODEL = joblib.load(DATA_DIR / "best_model.pkl")

@app.get("/")
def root():
    return {"mensaje": "API funcionando correctamente"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Sube un archivo CSV y devuelve las predicciones en lote.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df = create_features(df)  # crea las columnas derivadas
        X = PREP.transform(df)
        preds = MODEL.predict(X)
        return {"predicciones": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
