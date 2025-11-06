"""
Salidas (mlops_pipeline/data/):
    - best_model.pkl
    - model_metrics.csv
    - model_comparison.png
Uso:
    python model_training_evaluation.py --train ./data/train.csv --test ./data/test.csv --out_dir ./data
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def build_model(model, X_train, y_train):
    """Fit a model and return it"""
    model.fit(X_train, y_train)
    return model

def summarize_classification(model, X_test, y_test):
    """Return key metrics of a model"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    return {'accuracy': acc, 'f1_score': f1, 'roc_auc': roc}

def run(train_path, test_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop(columns=['target'])
    y_train = train['target']
    X_test = test.drop(columns=['target'])
    y_test = test['target']

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    metrics_list = []
    trained_models = {}

    for name, model in models.items():
        trained_model = build_model(model, X_train, y_train)
        trained_models[name] = trained_model
        metrics = summarize_classification(trained_model, X_test, y_test)
        metrics['model'] = name
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(out_dir / 'model_metrics.csv', index=False)
    print("Model metrics saved to:", out_dir / 'model_metrics.csv')

    best_model_name = metrics_df.sort_values('f1_score', ascending=False).iloc[0]['model']
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, out_dir / 'best_model.pkl')
    print(f"Best model: {best_model_name} saved to:", out_dir / 'best_model.pkl')

    metrics_df.set_index('model')[['accuracy','f1_score','roc_auc']].plot(kind='bar', figsize=(8,5))
    plt.title("Model comparison")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / 'model_comparison.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out_dir", default="./data")
    args = parser.parse_args()
    run(args.train, args.test, args.out_dir)