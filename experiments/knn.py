"""
K-NN (v1_baseline).

Grid completo conforme exigência (seção 8): n_neighbors ímpar 3-31,
weights {uniform, distance}, metric {euclidean, manhattan, minkowski}.
Total: 15 × 2 × 3 = 90 combos × 5 folds = 450 fits.

Execução: python experiments/knn.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.neighbors import KNeighborsClassifier
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="knn",
        variant="v1_baseline",
        model_factory=lambda: KNeighborsClassifier(n_jobs=-1),
        param_grid={
            "model__n_neighbors": list(range(3, 32, 2)),
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan", "minkowski"],
        },
        search_method="grid",
        main_hp_for_curve="model__n_neighbors",
        curve_range=[3, 5, 7, 11, 15, 21, 31],
    )


if __name__ == "__main__":
    main()
