"""
LVQ (v2_smote) — balanceamento via SMOTE no train fold de cada CV.

Mesmo grid do v1_baseline + shim de compatibilidade `GLVQCompat` (sklvq 0.1.2
× sklearn ≥1.6 — ver doc no v1). SMOTE inserido via `extra_pipeline_steps`.
Hipótese: LVQ já tinha o maior recall_alerta_grave do v1 (0.0374) — com SMOTE
deve subir mais. Os protótipos da classe minoritária terão mais "exemplos"
sintéticos pra ajustar suas posições.

Execução: python experiments/lvq_v2.py
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

from imblearn.over_sampling import SMOTE
from sklearn.utils.validation import validate_data as _sk_validate_data


class GLVQCompat(GLVQ):
    """GLVQ com shim pra sklearn ≥1.6 (ver doc no v1)."""

    def _validate_data(self, X, y=None, **kwargs):
        return _sk_validate_data(self, X, y=y, **kwargs)


from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="lvq",
        variant="v2_smote",
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
        extra_pipeline_steps=[("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5))],
    )


if __name__ == "__main__":
    main()
