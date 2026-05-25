"""
XGBoost (v3_target_enc) — TargetEncoder nas categóricas alta cardinalidade.

Hipótese: similar a lightgbm_v3, ganho marginal por redução de dimensionalidade.

Execução: python experiments/xgboost_v3.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import loguniform, uniform
import xgboost as xgb

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment, build_preprocessor_target_encoding


def main():
    run_experiment(
        algorithm="xgboost",
        variant="v3_target_enc",
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
        preprocessor_builder=build_preprocessor_target_encoding,
    )


if __name__ == "__main__":
    main()
