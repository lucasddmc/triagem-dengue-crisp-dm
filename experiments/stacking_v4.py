"""
Stacking heterogêneo (v4_selectk) — feature selection via Mutual Information (k=15).

Hipótese: base estimators (KNN, RF, MLP) recebem o mesmo espaço reduzido de 15
features. KNN é o mais beneficiado (ver lvq_v4 / knn_v4 hipóteses).
Esperamos stacking subir relativamente comparado ao v1 (5º lugar).

Execução: python experiments/stacking_v4.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
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
        variant="v4_selectk",
        model_factory=_mk_stacking,
        param_grid={
            "model__passthrough": [True, False],
            "model__final_estimator__C": [0.1, 1.0, 10.0],
        },
        search_method="grid",
        main_hp_for_curve=None,
        extra_pipeline_steps=[
            ("select", SelectKBest(score_func=mutual_info_classif, k=15)),
        ],
    )


if __name__ == "__main__":
    main()
