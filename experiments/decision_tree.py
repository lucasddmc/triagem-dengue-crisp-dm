"""
Árvore de Decisão (v1_baseline). CART, critérios Gini + Entropia.

Grid (seção 8): criterion {gini, entropy}, max_depth {None, 5-30},
min_samples_leaf 1-20, ccp_alpha 0-0.05.

Execução: python experiments/decision_tree.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.tree import DecisionTreeClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="decision_tree",
        variant="v1_baseline",
        model_factory=lambda: DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid={
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [None, 5, 10, 15, 20, 30],
            "model__min_samples_leaf": [1, 5, 10, 20],
            "model__ccp_alpha": [0.0, 0.001, 0.01, 0.05],
        },
        search_method="grid",
        main_hp_for_curve="model__max_depth",
        curve_range=[5, 10, 15, 20, 30, 50],  # int range pra plot numérico
    )


if __name__ == "__main__":
    main()
