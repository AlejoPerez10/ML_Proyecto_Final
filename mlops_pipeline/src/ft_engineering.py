"""
Salidas:
    - mlops_pipeline/src/data/feature_list.txt
    - mlops_pipeline/src/data/pipeline_preprocessor.pkl
    - mlops_pipeline/src/data/test.csv
    - mlops_pipeline/src/data/train.csv

Uso:
    python ft_engineering.py --input ../../Base_de_datos.csv --out_dir ./data --test_size 0.2 --random_state 42
"""

from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif

def create_features(df):
    df = df.copy()
    df['age_x_thalach'] = df['age'] * df['thalach']
    df['thalach_minus_age'] = df['thalach'] - df['age']
    df['oldpeak_ratio'] = df['oldpeak'] / (df['thalach'] + 1e-6)
    return df

def build_preprocessor(num_cols, cat_cols, scaler=True):
    # Pipeline numérico
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) if scaler else ('passthrough','passthrough')
    ])
    # Pipeline Categórico
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='drop', sparse_threshold=0)
    return preprocessor

def run(input_path, out_dir, test_size=0.2, random_state=42, scaler=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(input_path)

    data = create_features(data)

    num_cols = ['age','trestbps','chol','thalach','oldpeak','age_x_thalach','thalach_minus_age','oldpeak_ratio']
    cat_cols = ['sex','cp','fbs','restecg','exang','thal','slope','ca']

    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols, scaler=scaler)

    selector = SelectKBest(score_func=f_classif, k='all')

    full_pipeline = Pipeline([
        ('preproc', preprocessor),
        ('select', selector)
    ])

    full_pipeline.fit(X_train, y_train)

    X_train_tr = full_pipeline.named_steps['preproc'].transform(X_train)
    try:
        feat_names = full_pipeline.named_steps['preproc'].get_feature_names_out()
    except Exception:
        feat_names = [f"f_{i}" for i in range(X_train_tr.shape[1])]

    joblib.dump(full_pipeline, out_dir / 'pipeline_preprocessor.pkl')
    pd.Series(feat_names).to_csv(out_dir / 'feature_list.txt', index=False, header=False)

    train = X_train.copy(); train['target'] = y_train
    test = X_test.copy(); test['target'] = y_test
    train.to_csv(out_dir / 'train.csv', index=False)
    test.to_csv(out_dir / 'test.csv', index=False)


    print("Saved pipeline and datasets to:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    run(args.input, args.out_dir, args.test_size, args.random_state)