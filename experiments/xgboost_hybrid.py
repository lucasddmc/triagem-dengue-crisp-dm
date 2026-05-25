"""
Sanity check rápido: XGBoost + hybrid sampling (under + over) sobre o dataset LIMPO.

Objetivo: medir recall da classe 2 (Alerta/Grave) após o cleanup das colunas
de leak. Comparar com o run antigo no MLflow (`cm_xgboost_hybrid.png`) que
rodava sobre dataset com vazamento.

Sem GridSearch — só hiperparâmetros default razoáveis e CV 5-fold.

Execução: python experiments/xgboost_hybrid.py
"""
import sys
from pathlib import Path

# Adiciona raiz do projeto ao sys.path pra `from src.X import Y` funcionar
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb

from src.data_loader import load_train, RANDOM_STATE

# Constantes
EXPERIMENT_NAME = "triagem-dengue"
N_SPLITS = 5
ALGORITHM = "xgboost"
VARIANT = "v2_hybrid_postcleanup"

# Hybrid sampling: under classes 0/1 a 5k, over classe 2 a 5k
SAMPLING_UNDER = {0: 5000, 1: 5000}
SAMPLING_OVER = {2: 5000}


def build_pipeline(X_train):
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    # imblearn Pipeline pra suportar resamplers (só aplicados no fit, não no predict)
    pipe = ImbPipeline([
        ("preprocess", preprocess),
        ("under", RandomUnderSampler(sampling_strategy=SAMPLING_UNDER, random_state=RANDOM_STATE)),
        ("over",  RandomOverSampler(sampling_strategy=SAMPLING_OVER, random_state=RANDOM_STATE)),
        ("model", xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
        )),
    ])
    return pipe


def main():
    project_root = Path(__file__).resolve().parent.parent
    mlflow.set_tracking_uri(f"sqlite:///{project_root / 'mlflow_dengue.db'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Carregando treino...")
    X_train, y_train = load_train()
    print(f"  X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"  Classes: {y_train.value_counts().to_dict()}")

    pipe = build_pipeline(X_train)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\nRodando CV {N_SPLITS}-fold (XGBoost + under/over) sobre o treino...")
    # cross_val_predict gera predições pra cada amostra usando o fold em que ela
    # ficou de fora. Permite calcular métricas honestas sem leak.
    y_pred = cross_val_predict(pipe, X_train, y_train, cv=cv, n_jobs=1)

    f1_macro = f1_score(y_train, y_pred, average="macro")
    recall_per_class = recall_score(y_train, y_pred, average=None, labels=[0, 1, 2])

    print(f"\n{'='*60}")
    print(f"RESULTADOS — {ALGORITHM} + {VARIANT}")
    print(f"{'='*60}")
    print(f"\nF1-macro (CV): {f1_macro:.4f}")
    print(f"\nRecall por classe:")
    print(f"  0 (Descartado):    {recall_per_class[0]:.4f}")
    print(f"  1 (Comum):         {recall_per_class[1]:.4f}")
    print(f"  2 (Alerta/Grave):  {recall_per_class[2]:.4f}   ← métrica chave")

    print(f"\nMatriz de confusão (rows=verdade, cols=predito):")
    cm = confusion_matrix(y_train, y_pred, labels=[0, 1, 2])
    print(pd.DataFrame(cm, index=["0_real", "1_real", "2_real"], columns=["0_pred", "1_pred", "2_pred"]))

    print(f"\nClassification report:")
    print(classification_report(y_train, y_pred, target_names=["Descartado", "Comum", "Alerta/Grave"], zero_division=0))

    # Log MLflow
    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run(run_name=f"{ALGORITHM}_{VARIANT}"):
        mlflow.set_tag("algoritmo", ALGORITHM)
        mlflow.set_tag("variante", VARIANT)
        mlflow.set_tag("dataset_state", "post_cleanup_26cols")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("undersampling", str(SAMPLING_UNDER))
        mlflow.log_param("oversampling", str(SAMPLING_OVER))
        mlflow.log_metric("f1_macro_cv", float(f1_macro))
        mlflow.log_metric("recall_descartado", float(recall_per_class[0]))
        mlflow.log_metric("recall_comum",      float(recall_per_class[1]))
        mlflow.log_metric("recall_alerta_grave", float(recall_per_class[2]))

    print(f"\n✓ Run logado no MLflow (experiment '{EXPERIMENT_NAME}')")


if __name__ == "__main__":
    main()
