"""
SVM LinearSVC (v3_target_enc) — TargetEncoder nas categóricas alta cardinalidade.

Hipótese: LinearSVC é mais sensível a codificação que árvores. OneHot esparso
em 49 dims era ruim pra fronteira linear; TargetEncoder densifica o espaço
em 34 dims com semântica numérica (prob da classe dado cada nível). Esperamos
melhora notável vs v1 (F1=0.3921, recall_grave=0).

NOTA C3 (code review 25/05): LinearSVC vem wrapped em CalibratedClassifierCV
(method='sigmoid', cv=5) pra ter `predict_proba` — habilita ROC-AUC e PR-AUC
macro calibrados (Critério 6 da rubrica).

Execução: python experiments/svm_v3.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment, build_preprocessor_target_encoding


def main():
    run_experiment(
        algorithm="svm",
        variant="v3_target_enc",
        model_factory=lambda: CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_STATE, dual="auto", max_iter=5000),
            method="sigmoid", cv=5,
        ),
        param_grid={
            "model__estimator__C": loguniform(0.01, 100),
            "model__estimator__loss": ["hinge", "squared_hinge"],
        },
        search_method="random",
        n_iter=20,
        main_hp_for_curve="model__estimator__C",
        curve_range=[0.01, 0.1, 1.0, 10.0, 100.0],
        preprocessor_builder=build_preprocessor_target_encoding,
    )


if __name__ == "__main__":
    main()
