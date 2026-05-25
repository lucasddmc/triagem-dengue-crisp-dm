"""
Stacking heterogêneo (v3_target_enc) — TargetEncoder nas categóricas alta cardinalidade.

Hipótese: base estimators (KNN, RF, MLP) recebem o mesmo espaço densificado.
KNN é o mais beneficiado (ver knn_v3). Esperamos stacking subir um pouco.

Execução: python experiments/stacking_v3.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment, build_preprocessor_target_encoding


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
        variant="v3_target_enc",
        model_factory=_mk_stacking,
        param_grid={
            "model__passthrough": [True, False],
            "model__final_estimator__C": [0.1, 1.0, 10.0],
        },
        search_method="grid",
        main_hp_for_curve=None,
        preprocessor_builder=build_preprocessor_target_encoding,
    )


if __name__ == "__main__":
    main()
