"""
LightGBM (v3_target_enc) — TargetEncoder nas categóricas alta cardinalidade.

Mesmo grid do v1_baseline (com fix n_jobs=1). Hipótese: gradient boosting é
robusto a codificação. Ganho marginal de reduzir dimensionalidade (49→34).

Execução: python experiments/run_lightgbm_v3.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import loguniform, randint, uniform
import lightgbm as lgb

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment, build_preprocessor_target_encoding


def main():
    run_experiment(
        algorithm="lightgbm",
        variant="v3_target_enc",
        model_factory=lambda: lgb.LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
            objective="multiclass",
            num_class=3,
            verbose=-1,
        ),
        param_grid={
            "model__num_leaves": randint(15, 256),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__min_data_in_leaf": randint(5, 100),
            "model__feature_fraction": uniform(0.6, 0.4),
            "model__bagging_fraction": uniform(0.6, 0.4),
            "model__n_estimators": [100, 200, 400],
        },
        search_method="random",
        n_iter=30,
        main_hp_for_curve="model__num_leaves",
        curve_range=[15, 31, 63, 127, 255],
        preprocessor_builder=build_preprocessor_target_encoding,
    )


if __name__ == "__main__":
    main()
