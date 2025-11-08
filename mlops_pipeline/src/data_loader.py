import pandas as pd

def load_df():
    df = pd.read_csv('../../Base_de_datos.csv', encoding="utf-8")
    return df