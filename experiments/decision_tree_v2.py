"""
Árvore de Decisão (v2_smote) — balanceamento via SMOTE no train fold de cada CV.

Mesmo grid do v1_baseline. SMOTE(k_neighbors=5, random_state=42) inserido entre
preprocessor e model via `extra_pipeline_steps`; o imblearn Pipeline garante
que SMOTE só roda no fit (não na predição). Hipótese: rebalancear o train via
amostras sintéticas da minoritária deve aumentar `recall_alerta_grave` e
estreitar o gap train-val (que no v1 ficou em 0.22 — overfit à majoritária).

Execução: python experiments/decision_tree_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="decision_tree",
        variant="v2_smote",
        model_factory=lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid={
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [None, 5, 10, 15, 20, 30],
            "model__min_samples_leaf": [1, 5, 10, 20],
            "model__ccp_alpha": [0.0, 0.001, 0.01, 0.05],
        },
        search_method="grid",
        main_hp_for_curve="model__max_depth",
        curve_range=[5, 10, 15, 20, 30, 50],
        extra_pipeline_steps=[("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5))],
    )


if __name__ == "__main__":
    main()
