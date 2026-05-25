"""
Stacking heterogêneo (v1_baseline).

Combina classificadores das partes 1 e 2 com um meta-modelo (LogisticRegression).
Exigência (seção 8): base estimators a partir do baseline (KNN, RF, SVM, MLP),
final_estimator {LogReg, MLP_pequena}, passthrough {True, False}.

Execução: python experiments/stacking.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        cv=3,  # CV interno do stacking (separado do CV externo do GridSearch)
        n_jobs=1,  # evita aninhar paralelismo
    )


def main():
    run_experiment(
        algorithm="stacking",
        variant="v1_baseline",
        model_factory=_mk_stacking,
        param_grid={
            "model__passthrough": [True, False],
            "model__final_estimator__C": [0.1, 1.0, 10.0],
        },
        search_method="grid",   # espaço pequeno (6 combos)
        main_hp_for_curve=None,  # não há HP escalar significativo pra curva
    )


if __name__ == "__main__":
    main()
