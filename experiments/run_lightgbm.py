"""
LightGBM (v1_baseline).

Grid (Randomized, seção 8): num_leaves 15-255, learning_rate 0.01-0.3 (log),
min_data_in_leaf 5-100, feature_fraction 0.6-1.0, bagging_fraction 0.6-1.0.

DECISÃO DE EXECUÇÃO (2026-05-21): primeira submissão usava `n_jobs=-1` no
LGBMClassifier E `n_jobs=-1` no `RandomizedSearchCV` (default do runner). Isso
causa OVERSUBSCRIPTION: 8 workers do search × 8 threads do LightGBM = 64
threads brigando por 8 cores. Resultado: cada fit ficou ~70min, atingiu
timeout (4h em short-simple E 12h em long-simple). Fix: `n_jobs=1` no
classifier (paralelismo SÓ no search) + n_estimators reduzido pra grid mais
viável.

Execução: python experiments/run_lightgbm.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import loguniform, randint, uniform
import lightgbm as lgb

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="lightgbm",
        variant="v1_baseline",
        model_factory=lambda: lgb.LGBMClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,  # FIX oversubscription — search faz o paralelismo
            objective="multiclass",
            num_class=3,
            verbose=-1,
        ),
        param_grid={
            "model__num_leaves": randint(15, 256),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__min_data_in_leaf": randint(5, 100),
            "model__feature_fraction": uniform(0.6, 0.4),  # 0.6 a 1.0
            "model__bagging_fraction": uniform(0.6, 0.4),
            "model__n_estimators": [100, 200, 400],  # reduzido pra grid viável
        },
        search_method="random",
        n_iter=30,
        main_hp_for_curve="model__num_leaves",
        curve_range=[15, 31, 63, 127, 255],
    )


if __name__ == "__main__":
    main()
