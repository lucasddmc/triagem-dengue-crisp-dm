"""
Recupera fold scores + ROC/PR macro retroativamente pra runs MLflow rodados
ANTES da refatoração do `experiment_runner` (que agora persiste cv_results_
completo como artifact).

Pra cada run com `tags.source=<source>` e `tags.variante=<variant>`:
1. Lê best_params do MLflow.
2. Reconstrói pipeline via `build_pipeline(X, factory(params), extra_steps=...,
   preprocessor_builder=...)` aplicando o setup da variante.
3. Roda `cross_validate(pipe, X, y, cv=StratifiedKFold(5, shuffle=True,
   random_state=42), scoring='f1_macro', n_jobs=-1)` → 5 fold scores.
4. Roda `cross_val_predict(pipe, X, y, cv=cv, method='predict_proba'|'decision_function')`
   → matriz n×3 de scores → ROC-AUC macro (ovr) + PR-AUC macro.
5. Loga no MESMO run:
   - Metrics: `f1_macro_fold_{0..4}`, `roc_auc_macro_cv`, `pr_auc_macro_cv`
   - Tag: `score_method`
   - Artifacts: `{algo}_{variant}_best_fold_scores.csv`, `..._y_scores.csv`

Sanity check anti-regression: aborta com warning se
|recomputed_mean - logged_f1_macro_cv_search| > 0.01.

Caso especial SVM: wrap em `CalibratedClassifierCV(LinearSVC, method='sigmoid',
cv=5)` pra ter `predict_proba` (calibrado).

Uso:
    python scripts/recover_fold_scores.py --variant v1_baseline
    python scripts/recover_fold_scores.py --variant v2_smote
    python scripts/recover_fold_scores.py --variant v1_baseline --algo decision_tree  # debug
    python scripts/recover_fold_scores.py --variant v1_baseline --dry-run             # lista só
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import lightgbm as lgb
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import RANDOM_STATE, load_train  # noqa: E402
from src.experiment_runner import build_pipeline, build_preprocessor_target_encoding  # noqa: E402

# sklvq shim — opcional (LVQ runs)
try:
    from sklvq import GLVQ
    from sklearn.utils.validation import validate_data as _sk_validate_data

    class GLVQCompat(GLVQ):
        """GLVQ com shim pra sklearn ≥1.6 (idem experiments/lvq.py)."""
        def _validate_data(self, X, y=None, **kwargs):
            return _sk_validate_data(self, X, y=y, **kwargs)

    LVQ_AVAILABLE = True
except ImportError:
    LVQ_AVAILABLE = False

DB_PATH = PROJECT_ROOT / "mlflow_dengue.db"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
EXPERIMENT_NAME = "triagem-dengue"


# ---------- Cast de params ----------

def _cast_param(value: str):
    """Tenta converter string MLflow pra tipo Python correto."""
    if not isinstance(value, str):
        return value
    if value == "None":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    # Tupla (ex: hidden_layer_sizes='(64, 64)')
    if value.startswith("(") and value.endswith(")"):
        try:
            inner = value[1:-1].strip()
            if not inner:
                return ()
            parts = [_cast_param(x.strip()) for x in inner.split(",") if x.strip()]
            return tuple(parts)
        except Exception:
            pass
    # Float (loguniform / uniform)
    try:
        if "." in value or "e" in value.lower() or "E" in value:
            return float(value)
    except ValueError:
        pass
    # Int
    try:
        return int(value)
    except ValueError:
        pass
    return value


def _cast_params(params: dict) -> dict:
    return {k: _cast_param(v) for k, v in params.items()}


# ---------- Factories de modelo ----------

def _build_factory(algo: str, params: dict):
    """Retorna estimator reconstruído com best_params (cast types). SVM em Calibrated."""
    p = _cast_params(params)

    if algo == "decision_tree":
        keys = {"criterion", "max_depth", "min_samples_leaf", "ccp_alpha"}
        return DecisionTreeClassifier(random_state=RANDOM_STATE,
                                      **{k: v for k, v in p.items() if k in keys})
    if algo == "random_forest":
        keys = {"n_estimators", "max_features", "max_depth", "min_samples_leaf"}
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                      **{k: v for k, v in p.items() if k in keys})
    if algo == "lightgbm":
        keys = {"num_leaves", "learning_rate", "min_data_in_leaf",
                "feature_fraction", "bagging_fraction", "n_estimators"}
        return lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1,
                                  objective="multiclass", num_class=3, verbose=-1,
                                  **{k: v for k, v in p.items() if k in keys})
    if algo == "xgboost":
        keys = {"n_estimators", "max_depth", "learning_rate",
                "subsample", "colsample_bytree", "reg_lambda"}
        return xgb.XGBClassifier(random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1,
                                 objective="multi:softprob", num_class=3,
                                 eval_metric="mlogloss",
                                 **{k: v for k, v in p.items() if k in keys})
    if algo == "svm":
        keys = {"C", "loss"}
        base = LinearSVC(random_state=RANDOM_STATE, dual="auto", max_iter=5000,
                         **{k: v for k, v in p.items() if k in keys})
        return CalibratedClassifierCV(base, method="sigmoid", cv=5)
    if algo == "mlp":
        keys = {"hidden_layer_sizes", "activation", "alpha",
                "learning_rate_init", "batch_size"}
        return MLPClassifier(random_state=RANDOM_STATE, max_iter=200,
                             early_stopping=True, validation_fraction=0.1,
                             **{k: v for k, v in p.items() if k in keys})
    if algo == "knn":
        keys = {"n_neighbors", "weights", "metric"}
        return KNeighborsClassifier(n_jobs=-1,
                                    **{k: v for k, v in p.items() if k in keys})
    if algo == "lvq":
        if not LVQ_AVAILABLE:
            raise RuntimeError("sklvq não instalado localmente — pip install sklvq")
        keys = {"prototype_n_per_class", "activation_type"}
        return GLVQCompat(distance_type="squared-euclidean", random_state=RANDOM_STATE,
                          **{k: v for k, v in p.items() if k in keys})
    if algo == "stacking":
        # Reconstruir estrutura igual a experiments/stacking*.py
        estimators = [
            ("knn", KNeighborsClassifier(n_neighbors=15, n_jobs=-1)),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64,), max_iter=100,
                                  early_stopping=True, random_state=RANDOM_STATE)),
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                max_iter=500, random_state=RANDOM_STATE,
                C=p.get("final_estimator__C", 1.0),
            ),
            cv=3, n_jobs=1,
            passthrough=p.get("passthrough", False),
        )
    if algo == "rna_committee":
        return BaggingClassifier(
            estimator=MLPClassifier(
                hidden_layer_sizes=p.get("estimator__hidden_layer_sizes", (64,)),
                activation=p.get("estimator__activation", "relu"),
                max_iter=100, early_stopping=True, random_state=RANDOM_STATE,
            ),
            n_estimators=p.get("n_estimators", 10),
            max_samples=p.get("max_samples", 1.0),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Algoritmo desconhecido: {algo}")


# ---------- Variant config ----------

def _build_variant_steps(variant: str):
    """Retorna (extra_steps, preprocessor_builder) pra cada variante."""
    if variant == "v1_baseline":
        return None, None
    if variant == "v2_smote":
        return [("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5))], None
    if variant == "v3_target_enc":
        return None, build_preprocessor_target_encoding
    if variant == "v4_selectk":
        return [("select", SelectKBest(score_func=mutual_info_classif, k=15))], None
    raise ValueError(f"Variante desconhecida: {variant}")


# ---------- Recovery core ----------

def recover_run(client: MlflowClient, run, X_train, y_train, dry_run: bool = False) -> dict:
    rid = run.info.run_id
    algo = run.data.tags.get("algoritmo", "")
    variant = run.data.tags.get("variante", "")
    rname = run.info.run_name or "(unnamed)"
    print(f"\n  → {rname} (run_id={rid[:8]}, algo={algo})")

    # Skip se já tem fold scores completos
    if all(f"f1_macro_fold_{i}" in run.data.metrics for i in range(5)):
        if "roc_auc_macro_cv" in run.data.metrics:
            print("     ⏭️  já tem fold_scores + ROC/PR, pulando")
            return {"status": "skipped", "reason": "complete"}

    if dry_run:
        print("     [dry-run] viria reconstruir + cross_validate + log")
        return {"status": "dry_run"}

    # Reconstrói pipeline
    try:
        model = _build_factory(algo, run.data.params)
    except Exception as e:
        print(f"     ❌ factory falhou: {type(e).__name__}: {e}")
        return {"status": "factory_failed", "error": str(e)}

    extra_steps, preprocessor_builder = _build_variant_steps(variant)
    pipe = build_pipeline(X_train, model, extra_steps=extra_steps,
                          preprocessor_builder=preprocessor_builder)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1) Fold scores
    print(f"     cross_validate (5 folds)...")
    t0 = time.time()
    try:
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring="f1_macro",
                                n_jobs=-1, return_estimator=False, return_train_score=False)
        fold_scores = [float(s) for s in cv_res["test_score"]]
        print(f"        ✓ {time.time()-t0:.1f}s; scores={[f'{s:.4f}' for s in fold_scores]}; "
              f"mean={np.mean(fold_scores):.4f}")
    except Exception as e:
        print(f"        ❌ cross_validate falhou: {type(e).__name__}: {e}")
        return {"status": "cv_failed", "error": str(e)}

    # Sanity vs logged
    logged_mean = run.data.metrics.get("f1_macro_cv_search", float("nan"))
    if not np.isnan(logged_mean):
        delta = abs(np.mean(fold_scores) - logged_mean)
        marker = "⚠️" if delta > 0.01 else "✓"
        print(f"        {marker} sanity: recomputed={np.mean(fold_scores):.4f} vs "
              f"logged={logged_mean:.4f} (Δ={delta:.4f})")

    # 2) y_scores pra ROC/PR
    print(f"     cross_val_predict (proba)...")
    t1 = time.time()
    y_score = None
    score_method = None
    for method in ("predict_proba", "decision_function"):
        try:
            pipe2 = build_pipeline(X_train, _build_factory(algo, run.data.params),
                                    extra_steps=extra_steps,
                                    preprocessor_builder=preprocessor_builder)
            y_score = cross_val_predict(pipe2, X_train, y_train, cv=cv, n_jobs=1,
                                        method=method)
            score_method = method
            print(f"        ✓ {time.time()-t1:.1f}s; method={method}; shape={y_score.shape}")
            break
        except (AttributeError, ValueError) as e:
            print(f"        skip {method}: {type(e).__name__}: {str(e)[:80]}")
            continue
    if y_score is None:
        print(f"        ⚠️  sem predict_proba nem decision_function; ROC/PR vão ficar NaN")

    # 3) ROC/PR macro
    roc_auc_macro, pr_auc_macro = None, None
    if y_score is not None:
        try:
            roc_auc_macro = float(roc_auc_score(y_train, y_score, multi_class="ovr",
                                                 average="macro", labels=[0, 1, 2]))
            y_bin = label_binarize(y_train, classes=[0, 1, 2])
            pr_auc_macro = float(average_precision_score(y_bin, y_score, average="macro"))
            print(f"        ROC-AUC macro={roc_auc_macro:.4f}, PR-AUC macro={pr_auc_macro:.4f}")
        except Exception as e:
            print(f"        ⚠️  ROC/PR falhou: {type(e).__name__}: {e}")

    # 4) Log no run existente
    print(f"     logando metrics + artifacts em {rid[:8]}...")
    for i, s in enumerate(fold_scores):
        client.log_metric(rid, f"f1_macro_fold_{i}", s)
    if roc_auc_macro is not None:
        client.log_metric(rid, "roc_auc_macro_cv", roc_auc_macro)
        client.log_metric(rid, "pr_auc_macro_cv", pr_auc_macro)
    if score_method:
        client.set_tag(rid, "score_method", score_method)
    client.set_tag(rid, "fold_scores_recovered", "true")

    # 5) Artifacts CSV
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = f"{algo}_{variant}"
    folds_path = FIG_DIR / f"{base}_best_fold_scores.csv"
    pd.DataFrame({"fold": list(range(5)), "f1_macro": fold_scores}).to_csv(folds_path, index=False)
    client.log_artifact(rid, str(folds_path))

    if y_score is not None:
        scores_path = FIG_DIR / f"{base}_y_scores.csv"
        pd.DataFrame(y_score, columns=["score_0", "score_1", "score_2"]).to_csv(scores_path,
                                                                                  index=False)
        client.log_artifact(rid, str(scores_path))

    return {
        "status": "ok", "algo": algo, "variant": variant,
        "fold_scores": fold_scores,
        "roc_auc_macro": roc_auc_macro, "pr_auc_macro": pr_auc_macro,
        "score_method": score_method,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True,
                   choices=["v1_baseline", "v2_smote", "v3_target_enc", "v4_selectk"])
    p.add_argument("--source", default="apuana")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--algo", default=None, help="Filtra por algoritmo (debug)")
    args = p.parse_args(argv)

    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' não existe.")
        return 1

    filter_parts = [f"tags.variante = '{args.variant}'", f"tags.source = '{args.source}'"]
    if args.algo:
        filter_parts.append(f"tags.algoritmo = '{args.algo}'")
    runs = client.search_runs(experiment_ids=[exp.experiment_id],
                              filter_string=" AND ".join(filter_parts))
    print(f"Variant: {args.variant}, source: {args.source}{', algo=' + args.algo if args.algo else ''}")
    print(f"{len(runs)} runs encontrados.\n")
    if not runs:
        return 1

    if args.dry_run:
        for r in runs:
            algo = r.data.tags.get("algoritmo", "?")
            has_folds = all(f"f1_macro_fold_{i}" in r.data.metrics for i in range(5))
            has_roc = "roc_auc_macro_cv" in r.data.metrics
            status = "complete" if (has_folds and has_roc) else (
                "folds_only" if has_folds else "missing")
            print(f"  {algo:18s}  run_id={r.info.run_id[:8]}  status={status}")
        return 0

    print("Carregando treino...")
    X, y = load_train()
    print(f"  X={X.shape}, classes={dict(zip(*np.unique(y, return_counts=True)))}\n")

    results = []
    for run in runs:
        results.append(recover_run(client, run, X, y, dry_run=False))

    print("\n" + "=" * 60)
    print("Resumo:")
    statuses = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    for s, n in sorted(statuses.items()):
        print(f"  {s}: {n}")

    return 0 if statuses.get("ok", 0) > 0 or statuses.get("skipped", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
