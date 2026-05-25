"""
Comitê de Redes Neurais (v2_smote) — balanceamento via SMOTE no train fold de cada CV.

Mesmo grid do v1_baseline (BaggingClassifier de MLPs). SMOTE inserido via
`extra_pipeline_steps`. Hipótese: rna_committee teve recall_alerta_grave =
0.0000 EXATO no v1 (ensemble herda viés dos componentes individuais). Com
SMOTE, cada MLP base recebe um treino balanceado em sua amostra de bagging,
e o agregado deve começar a predizer Alerta/Grave.

Execução: python experiments/rna_committee_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier
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
        variant="v2_smote",
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
        extra_pipeline_steps=[("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5))],
    )


if __name__ == "__main__":
    main()
