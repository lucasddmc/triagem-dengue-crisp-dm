"""
Árvore de Decisão (v4_selectk) — feature selection via Mutual Information (k=15).

Mesmo grid do v1_baseline. Adiciona `SelectKBest(mutual_info_classif, k=15)`
entre preprocessor (OneHot → 49 features) e model via `extra_pipeline_steps`.
Hipótese: árvores fazem feature selection implícita, então SelectKBest deve
ter efeito pequeno em DT — provavelmente leve melhora por reduzir ruído.

Execução: python experiments/decision_tree_v4.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="decision_tree",
        variant="v4_selectk",
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
        extra_pipeline_steps=[
            ("select", SelectKBest(score_func=mutual_info_classif, k=15)),
        ],
    )


if __name__ == "__main__":
    main()
