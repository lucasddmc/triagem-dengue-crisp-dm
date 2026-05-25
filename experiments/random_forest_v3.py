"""
Random Forest (v3_target_enc) — TargetEncoder nas categóricas alta cardinalidade.

Hipótese: árvores são pouco sensíveis a codificação (one-hot vs target) porque
splits funcionam em ambos. Efeito esperado: marginal — talvez ligeiro ganho
por reduzir dimensionalidade. Não esperamos mudança grande no recall_grave
(v2 SMOTE é mais direto pra isso).

Execução: python experiments/random_forest_v3.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment, build_preprocessor_target_encoding


def main():
    run_experiment(
        algorithm="random_forest",
        variant="v3_target_enc",
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
        preprocessor_builder=build_preprocessor_target_encoding,
    )


if __name__ == "__main__":
    main()
