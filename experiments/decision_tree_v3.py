"""
Árvore de Decisão (v3_target_enc) — codificação via TargetEncoder pras categóricas.

Mesmo grid do v1_baseline. Substitui `OneHotEncoder` por `TargetEncoder`
(`target_type=multiclass`) nas 3 categóricas com cardinalidade ≥ 5
(tp_gestante, tp_raca_cor, tp_escolaridade). Mantém OneHot pra tp_sexo (3 levels).
Reduz dimensionalidade de 49 → 34 features. Anti-leakage: TargetEncoder é
sklearn-compatible, refita-se dentro de cada fold do CV automaticamente.

Execução: python experiments/decision_tree_v3.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.tree import DecisionTreeClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment, build_preprocessor_target_encoding


def main():
    run_experiment(
        algorithm="decision_tree",
        variant="v3_target_enc",
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
        preprocessor_builder=build_preprocessor_target_encoding,
    )


if __name__ == "__main__":
    main()
