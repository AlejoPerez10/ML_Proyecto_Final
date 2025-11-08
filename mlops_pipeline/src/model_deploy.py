import joblib
import pandas as pd

PREP = joblib.load("./data/pipeline_preprocessor.pkl")
MODEL = joblib.load("./data/best_model.pkl")

def predict(df: pd.DataFrame):
    X = PREP.transform(df)
    preds = MODEL.predict(X)
    return preds
