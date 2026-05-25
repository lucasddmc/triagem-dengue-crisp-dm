"""
K-NN (v4_selectk) — feature selection via Mutual Information (k=15).

Hipótese: KNN é o algoritmo onde feature selection deveria ter MAIOR efeito
após v2 (SMOTE) — KNN sofre muito com curse of dimensionality em 49 features.
Reduzir pra 15 features mais informativas deve aproximar pontos de mesma
classe no espaço euclidiano/manhattan, melhorando o voto dos vizinhos.

Execução: python experiments/knn_v4.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="knn",
        variant="v4_selectk",
        model_factory=lambda: KNeighborsClassifier(n_jobs=-1),
        param_grid={
            "model__n_neighbors": list(range(3, 32, 2)),
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan", "minkowski"],
        },
        search_method="grid",
        main_hp_for_curve="model__n_neighbors",
        curve_range=[3, 5, 7, 11, 15, 21, 31],
        extra_pipeline_steps=[
            ("select", SelectKBest(score_func=mutual_info_classif, k=15)),
        ],
    )


if __name__ == "__main__":
    main()
