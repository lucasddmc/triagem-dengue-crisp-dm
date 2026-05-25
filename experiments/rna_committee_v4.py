"""
Comitê de Redes Neurais (v4_selectk) — feature selection via Mutual Information (k=15).

Hipótese: cada MLP base no Bagging vê o mesmo espaço reduzido de 15 features
(SelectKBest é determinístico). Diversidade do ensemble vem só do bagging
de amostras + init dos pesos. Efeito esperado: melhora modesta similar ao
mlp_v4, talvez ligeiramente maior pelo voting.

Execução: python experiments/rna_committee_v4.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.neural_network import MLPClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def _mk_committee():
    return BaggingClassifier(
        estimator=MLPClassifier(
            hidden_layer_sizes=(64,),
            max_iter=100,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def main():
    run_experiment(
        algorithm="rna_committee",
        variant="v4_selectk",
        model_factory=_mk_committee,
        param_grid={
            "model__n_estimators": [5, 10, 20, 30],
            "model__estimator__hidden_layer_sizes": [(64,), (128,), (64, 64)],
            "model__estimator__activation": ["relu", "tanh"],
            "model__max_samples": [0.5, 0.7, 1.0],
        },
        search_method="random",
        n_iter=15,
        main_hp_for_curve="model__n_estimators",
        curve_range=[5, 10, 20, 30],
        extra_pipeline_steps=[
            ("select", SelectKBest(score_func=mutual_info_classif, k=15)),
        ],
    )


if __name__ == "__main__":
    main()
