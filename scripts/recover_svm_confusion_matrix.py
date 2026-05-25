"""
Re-deriva a matriz de confusão do SVM (LinearSVC) v1_baseline localmente
e grava num `.out` no formato do `experiment_runner`. O log original
(`triagem_svm_742.out`) foi perdido num cleanup do Apuana.

Lê os best_params do run `svm_v1_baseline` no MLflow local, reconstrói o
pipeline padrão (OneHotEncoder + StandardScaler + LinearSVC com C/loss
escolhidos), roda `cross_val_predict` 5-fold idêntico à rodada original
(RANDOM_STATE=42) e printa no MESMO formato dos jobs do Apuana — assim
o parser do `build_confusion_matrices.py` pega sem mudança.

Tempo esperado: ~5min (LinearSVC é o algoritmo mais rápido dos 10).

Saída:
    mlflow_apuana/logs/triagem_svm_recovered.out
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import RANDOM_STATE, load_train  # noqa: E402
from src.experiment_runner import build_pipeline  # noqa: E402

DB_PATH = PROJECT_ROOT / "mlflow_dengue.db"
LOGS_DIR = PROJECT_ROOT / "mlflow_apuana" / "logs"
OUT_PATH = LOGS_DIR / "triagem_svm_recovered.out"


def fetch_best_svm_params() -> dict:
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    c = MlflowClient()
    exp = c.get_experiment_by_name("triagem-dengue")
    runs = c.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.algoritmo = 'svm' AND tags.source = 'apuana'",
    )
    if not runs:
        raise SystemExit("Run svm_v1_baseline não encontrado no MLflow.")
    return runs[0].data.params


def main():
    print("[recover_svm] Lendo best_params do MLflow...")
    params = fetch_best_svm_params()
    C = float(params["C"])
    loss = params["loss"]
    print(f"  best: C={C:.6g}, loss={loss}")

    print("\n[recover_svm] Carregando treino...")
    X_train, y_train = load_train()
    classes_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    print(f"  X_train={X_train.shape}, classes={classes_counts}")

    print("\n[recover_svm] Construindo pipeline (LinearSVC + best_params)...")
    model = LinearSVC(random_state=RANDOM_STATE, dual="auto", max_iter=5000,
                      C=C, loss=loss)
    pipe = build_pipeline(X_train, model, extra_steps=None)

    print("\n[recover_svm] cross_val_predict (5 folds, RANDOM_STATE=42)...")
    t0 = time.time()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(pipe, X_train, y_train, cv=cv, n_jobs=-1)
    elapsed = time.time() - t0
    print(f"  ✓ feito em {elapsed:.1f}s")

    # Calcula métricas e printa NO MESMO FORMATO do experiment_runner (linhas 293-300)
    f1_macro = f1_score(y_train, y_pred, average="macro")
    recall_per_class = recall_score(y_train, y_pred, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(y_train, y_pred, labels=[0, 1, 2])

    out_lines = []
    out_lines.append(f"[{time.strftime('%a %b %d %I:%M:%S %p %Z %Y')}] "
                     "Iniciando SVM recovery (LinearSVC, params do MLflow)")
    out_lines.append("")
    out_lines.append("=" * 70)
    out_lines.append("EXPERIMENTO: svm | variante: v1_baseline (recovered)")
    out_lines.append("=" * 70)
    out_lines.append(f"  best params: C={C:.6g}, loss={loss}")
    out_lines.append("")
    out_lines.append(f"--- Métricas (svm | v1_baseline | recovered) ---")
    out_lines.append(f"F1-macro (CV):     {f1_macro:.4f}")
    out_lines.append(f"Recall Descartado: {recall_per_class[0]:.4f}")
    out_lines.append(f"Recall Comum:      {recall_per_class[1]:.4f}")
    out_lines.append(f"Recall Alerta/G.:  {recall_per_class[2]:.4f}   ← chave")
    out_lines.append("")
    out_lines.append("Matriz de confusão:")
    out_lines.append(str(pd.DataFrame(cm,
                                      index=["0_real", "1_real", "2_real"],
                                      columns=["0_pred", "1_pred", "2_pred"])))

    output = "\n".join(out_lines)
    print()
    print(output)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(output + "\n")
    print(f"\n  ✓ Saída → {OUT_PATH.relative_to(PROJECT_ROOT)}")

    # Sanity check
    if abs(f1_macro - 0.3921) > 0.01:
        print(f"\n  [warn] F1-macro {f1_macro:.4f} difere do registrado no MLflow (0.3921). "
              "Pode indicar config diferente.")
    else:
        print(f"  ✓ F1-macro confere com MLflow (esperado 0.3921, obtido {f1_macro:.4f})")


if __name__ == "__main__":
    main()
