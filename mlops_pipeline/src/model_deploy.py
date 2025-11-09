# model_deploy.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
import io
from ft_engineering import create_features

app = FastAPI(title="Modelo Predictivo API")

# Cargar modelo y preprocesador
PREP = joblib.load("./data/pipeline_preprocessor.pkl")
MODEL = joblib.load("./data/best_model.pkl")

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
        df = create_features(df)  # <<--- crea las columnas derivadas
        X = PREP.transform(df)
        preds = MODEL.predict(X)
        return {"predicciones": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))