"""
K-NN (v2_smote) — balanceamento via SMOTE no train fold de cada CV.

Mesmo grid do v1_baseline. SMOTE inserido via `extra_pipeline_steps`. Hipótese:
KNN é o algoritmo onde SMOTE deveria ter MAIOR efeito — KNN classifica por
votação dos vizinhos no train, e no v1 a vizinhança de qualquer ponto era ~99%
de classes majoritárias (a minoritária é 0.6% da base). Com SMOTE, a densidade
local de cada classe fica equilibrada, então a votação muda dramaticamente.
recall_alerta_grave do v1 era 0.0000 — esperamos salto grande.

Execução: python experiments/knn_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="knn",
        variant="v2_smote",
        model_factory=lambda: KNeighborsClassifier(n_jobs=-1),
        param_grid={
            "model__n_neighbors": list(range(3, 32, 2)),
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan", "minkowski"],
        },
        search_method="grid",
        main_hp_for_curve="model__n_neighbors",
        curve_range=[3, 5, 7, 11, 15, 21, 31],
        extra_pipeline_steps=[("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5))],
    )


if __name__ == "__main__":
    main()
