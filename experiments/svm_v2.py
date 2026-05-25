"""
SVM LinearSVC (v2_smote) — balanceamento via SMOTE no train fold de cada CV.

Mesmo grid do v1_baseline. SMOTE inserido via `extra_pipeline_steps`. Hipótese:
no v1, SVM teve recall_alerta_grave = 0.0000 EXATO (0/321 graves recuperados).
Com SMOTE, esperamos LinearSVC começar a predizer Alerta/Grave de fato.

NOTA C3 (code review 25/05): LinearSVC vem wrapped em CalibratedClassifierCV
(method='sigmoid', cv=5) pra ter `predict_proba` — habilita ROC-AUC e PR-AUC
macro calibrados (Critério 6 da rubrica, 8% peso). Custo: ~5× lento mas
LinearSVC é rápido (3-30s/fit), aceitável.

Execução: python experiments/svm_v2.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imblearn.over_sampling import SMOTE
from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="svm",
        variant="v2_smote",
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
        extra_pipeline_steps=[("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5))],
    )


if __name__ == "__main__":
    main()
