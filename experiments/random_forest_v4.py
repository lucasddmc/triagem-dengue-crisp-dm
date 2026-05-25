"""
Random Forest (v4_selectk) — feature selection via Mutual Information (k=15).

Hipótese: RF foi o pior overfit no v1 (gap 0.27). SelectKBest deve reduzir
ruído nas features menos informativas e estreitar o gap modestamente.
Provavelmente abaixo do efeito do v2 (SMOTE) — RF já faz feature subset por
árvore via `max_features`.

Execução: python experiments/random_forest_v4.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="random_forest",
        variant="v4_selectk",
        model_factory=lambda: RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        param_grid={
            "model__n_estimators": [100, 200, 400, 600, 800],
            "model__max_features": ["sqrt", "log2", 0.5],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_leaf": [1, 5, 10],
        },
        search_method="random",
        n_iter=30,
        main_hp_for_curve="model__n_estimators",
        curve_range=[100, 200, 400, 600, 800],
        extra_pipeline_steps=[
            ("select", SelectKBest(score_func=mutual_info_classif, k=15)),
        ],
    )


if __name__ == "__main__":
    main()
