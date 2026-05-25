"""
Random Forest (v1_baseline).

Grid (Randomized, seção 8): n_estimators 100-800, max_features {sqrt, log2, 0.5},
max_depth {None, 10-30}, min_samples_leaf 1-10.

Execução: python experiments/random_forest.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="random_forest",
        variant="v1_baseline",
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
    )


if __name__ == "__main__":
    main()
