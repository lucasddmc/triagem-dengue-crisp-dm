"""
XGBoost (v4_selectk) — feature selection via Mutual Information (k=15).

Hipótese: XGBoost já tem `colsample_bytree` no grid (0.6-1.0). SelectKBest é
intervenção complementar (fixo global vs estocástico por árvore). Efeito
esperado: modesto.

Execução: python experiments/xgboost_v4.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import loguniform, uniform
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="xgboost",
        variant="v4_selectk",
        model_factory=lambda: xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
        ),
        param_grid={
            "model__n_estimators": [200, 400, 600, 800, 1000],
            "model__max_depth": [3, 5, 7, 9, 12],
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_lambda": loguniform(0.1, 10),
        },
        search_method="random",
        n_iter=30,
        main_hp_for_curve="model__max_depth",
        curve_range=[3, 5, 7, 9, 12],
        extra_pipeline_steps=[
            ("select", SelectKBest(score_func=mutual_info_classif, k=15)),
        ],
    )


if __name__ == "__main__":
    main()
