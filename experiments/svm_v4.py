"""
SVM LinearSVC (v4_selectk) — feature selection via Mutual Information (k=15).

Hipótese: LinearSVC é sensível a curse of dimensionality (49 features pós-OneHot
em 52k amostras, classe minoritária com 321 samples = ~6.5 samples/feature na
rara). Reduzir pra k=15 features mais informativas deve melhorar F1-macro e
talvez ativar recall_alerta_grave (que no v1 era 0.0000).

NOTA C3 (code review 25/05): LinearSVC vem wrapped em CalibratedClassifierCV
(method='sigmoid', cv=5) pra ter `predict_proba` — habilita ROC-AUC e PR-AUC
macro calibrados (Critério 6 da rubrica).

Execução: python experiments/svm_v4.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import LinearSVC

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="svm",
        variant="v4_selectk",
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
        extra_pipeline_steps=[
            ("select", SelectKBest(score_func=mutual_info_classif, k=15)),
        ],
    )


if __name__ == "__main__":
    main()
