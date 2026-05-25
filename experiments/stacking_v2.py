"""
Stacking heterogêneo (v2_smote) — balanceamento via SMOTE no train fold de cada CV.

Mesmo grid do v1_baseline (base = KNN + RF + MLP, meta = LogisticRegression).
SMOTE inserido via `extra_pipeline_steps` — aplica antes do StackingClassifier
treinar tanto os base estimators quanto o meta-learner. Hipótese: stacking
ficou em 5º lugar do v1 (F1=0.4188), abaixo dos componentes individuais
lightgbm/xgboost/DT. Com classes balanceadas, os base estimators devem
divergir mais em suas predições (mais informação útil pro meta-learner) —
esperamos stacking subir no ranking.

Execução: python experiments/stacking_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def _mk_stacking():
    estimators = [
        ("knn", KNeighborsClassifier(n_neighbors=15, n_jobs=-1)),
        ("rf",  RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64,), max_iter=100,
                               early_stopping=True, random_state=RANDOM_STATE)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        cv=3,
        n_jobs=1,
    )


def main():
    run_experiment(
        algorithm="stacking",
        variant="v2_smote",
        model_factory=_mk_stacking,
        param_grid={
            "model__passthrough": [True, False],
            "model__final_estimator__C": [0.1, 1.0, 10.0],
        },
        search_method="grid",
        main_hp_for_curve=None,
        extra_pipeline_steps=[("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5))],
    )


if __name__ == "__main__":
    main()
