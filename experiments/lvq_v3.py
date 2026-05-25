"""
LVQ (v3_target_enc) — TargetEncoder nas categóricas alta cardinalidade.

Mesmo shim `GLVQCompat` (sklvq 0.1.2 × sklearn ≥1.6 — ver doc no v1). Hipótese
similar a knn_v3: protótipos LVQ vivem em espaço euclidiano; codificação densa
deve ajudar mais que OneHot esparso.

Execução: python experiments/lvq_v3.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from sklvq import GLVQ
except ImportError as e:
    raise ImportError(
        "sklvq não está instalado. Rode: pip install sklvq\n"
        f"Erro original: {e}"
    )

from sklearn.utils.validation import validate_data as _sk_validate_data


class GLVQCompat(GLVQ):
    """GLVQ com shim pra sklearn ≥1.6 (ver doc no v1)."""

    def _validate_data(self, X, y=None, **kwargs):
        return _sk_validate_data(self, X, y=y, **kwargs)


from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment, build_preprocessor_target_encoding


def main():
    run_experiment(
        algorithm="lvq",
        variant="v3_target_enc",
        model_factory=lambda: GLVQCompat(
            distance_type="squared-euclidean",
            random_state=RANDOM_STATE,
        ),
        param_grid={
            "model__prototype_n_per_class": [1, 2, 3, 5],
            "model__activation_type": ["sigmoid", "swish", "identity"],
        },
        search_method="grid",
        main_hp_for_curve="model__prototype_n_per_class",
        curve_range=[1, 2, 3, 5],
        preprocessor_builder=build_preprocessor_target_encoding,
    )


if __name__ == "__main__":
    main()
