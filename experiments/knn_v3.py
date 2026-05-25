"""
K-NN (v3_target_enc) — TargetEncoder nas categóricas alta cardinalidade.

Hipótese: KNN sofre muito com OneHot esparso (distância euclidiana fica
dominada pelas dimensões binárias). Com TargetEncoder, cada categórica vira 3
colunas contínuas (P(classe|nível)) — mais informativas e densas. Esperamos
melhora notável no F1 e talvez sair do recall_grave=0 do v1 (sem precisar de
SMOTE).

Execução: python experiments/knn_v3.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.neighbors import KNeighborsClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment, build_preprocessor_target_encoding


def main():
    run_experiment(
        algorithm="knn",
        variant="v3_target_enc",
        model_factory=lambda: KNeighborsClassifier(n_jobs=-1),
        param_grid={
            "model__n_neighbors": list(range(3, 32, 2)),
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan", "minkowski"],
        },
        search_method="grid",
        main_hp_for_curve="model__n_neighbors",
        curve_range=[3, 5, 7, 11, 15, 21, 31],
        preprocessor_builder=build_preprocessor_target_encoding,
    )


if __name__ == "__main__":
    main()
