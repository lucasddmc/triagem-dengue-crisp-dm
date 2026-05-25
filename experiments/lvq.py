"""
LVQ (Learning Vector Quantization) via pacote sklvq.

Grid (seção 8): distance_type {squared-euclidean},
prototypes_per_class 1-5, activation_type {sigmoid, swish, identity}.

⚠️ Requer `pip install sklvq`. Pode ter problemas de instalação em alguns
ambientes (depende de scipy/sklearn versions específicas). Validar primeiro
o import antes de rodar.

Execução: python experiments/lvq.py
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

# Shim de compatibilidade: sklvq 0.1.2 (último release, 2020) chama
# `self._validate_data(...)` herdado de sklearn.BaseEstimator. Esse método foi
# removido em sklearn ≥1.6 (substituído pela função module-level
# `sklearn.utils.validation.validate_data`).
#
# Solução: subclasse com `_validate_data` definido. Monkey-patch direto na
# classe GLVQ NÃO sobrevive ao GridSearchCV(n_jobs>1), porque os workers do
# joblib (backend `loky`) reimportam `sklvq` em subprocessos novos e perdem o
# patch. Já a subclasse define o método como parte da classe — pickle/unpickle
# preservam o atributo de classe.
from sklearn.utils.validation import validate_data as _sk_validate_data


class GLVQCompat(GLVQ):
    """GLVQ com shim pra sklearn ≥1.6."""

    def _validate_data(self, X, y=None, **kwargs):
        return _sk_validate_data(self, X, y=y, **kwargs)


from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="lvq",
        variant="v1_baseline",
        model_factory=lambda: GLVQCompat(
            distance_type="squared-euclidean",
            random_state=RANDOM_STATE,
        ),
        param_grid={
            "model__prototype_n_per_class": [1, 2, 3, 5],
            "model__activation_type": ["sigmoid", "swish", "identity"],
        },
        search_method="grid",  # 12 combos, viável grid
        main_hp_for_curve="model__prototype_n_per_class",
        curve_range=[1, 2, 3, 5],
    )


if __name__ == "__main__":
    main()
