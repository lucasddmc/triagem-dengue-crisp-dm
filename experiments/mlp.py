"""
MLP (v1_baseline).

Grid (Randomized, seção 8): hidden_layer_sizes {(64,), (128,), (64,64), (128,64)},
activation {relu, tanh}, alpha 1e-5 a 1e-1 (log),
learning_rate_init 1e-4 a 1e-1 (log), batch_size {32, 64, 128}.

Execução: python experiments/mlp.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import loguniform
from sklearn.neural_network import MLPClassifier

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="mlp",
        variant="v1_baseline",
        model_factory=lambda: MLPClassifier(
            random_state=RANDOM_STATE,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
        ),
        param_grid={
            "model__hidden_layer_sizes": [(64,), (128,), (64, 64), (128, 64)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": loguniform(1e-5, 1e-1),
            "model__learning_rate_init": loguniform(1e-4, 1e-1),
            "model__batch_size": [32, 64, 128],
        },
        search_method="random",
        n_iter=25,
        main_hp_for_curve="model__alpha",
        curve_range=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    )


if __name__ == "__main__":
    main()
