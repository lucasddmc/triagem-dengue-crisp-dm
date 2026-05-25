"""
Random Forest (v2_smote) — balanceamento via SMOTE no train fold de cada CV.

Mesmo grid do v1_baseline. SMOTE(k_neighbors=5, random_state=42) inserido entre
preprocessor e model via `extra_pipeline_steps`. Hipótese: o gap train-val de
0.27 no v1 (maior overfit dos 10 modelos) deve diminuir significativamente —
RF estava memorizando estrutura da majoritária; com SMOTE forçamos atenção
proporcional à minoritária no treino.

Execução: python experiments/random_forest_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="random_forest",
        variant="v2_smote",
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
        extra_pipeline_steps=[("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5))],
    )


if __name__ == "__main__":
    main()
