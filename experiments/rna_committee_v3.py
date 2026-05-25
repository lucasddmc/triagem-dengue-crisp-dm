"""
Comitê de Redes Neurais (v3_target_enc) — TargetEncoder nas categóricas alta cardinalidade.

Hipótese: similar a mlp_v3 — entrada mais densa + menos features ajuda
ligeiramente cada MLP base.

Execução: python experiments/rna_committee_v3.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment, build_preprocessor_target_encoding


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
        variant="v3_target_enc",
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
        preprocessor_builder=build_preprocessor_target_encoding,
    )


if __name__ == "__main__":
    main()
