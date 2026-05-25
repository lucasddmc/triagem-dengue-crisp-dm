"""
Runner abstrato pra experimentos de mineração de dados.

Centraliza:
- Construção do preprocessor (ColumnTransformer adaptativo aos tipos de coluna)
- Construção do Pipeline (com ou sem resamplers via imblearn)
- Busca de hiperparâmetros (Grid ou Randomized) com folds fixos
- validation_curve pra um HP principal
- Log MLflow padronizado (tags, params, métricas, figura, modelo)

Uso típico dentro de experiments/<algo>.py:

    from src.experiment_runner import run_experiment
    from sklearn.neighbors import KNeighborsClassifier

    def main():
        run_experiment(
            algorithm="knn",
            variant="v1_baseline",
            model_factory=lambda: KNeighborsClassifier(n_jobs=-1),
            param_grid={
                "model__n_neighbors": list(range(3, 32, 2)),
                "model__weights": ["uniform", "distance"],
                "model__metric": ["euclidean", "manhattan", "minkowski"],
            },
            search_method="grid",
            main_hp_for_curve="model__n_neighbors",
            curve_range=[3, 5, 7, 11, 15, 21, 31],
        )

    if __name__ == "__main__":
        main()
"""
from pathlib import Path
from typing import Callable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # backend não-interativo (background-safe)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix, f1_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
    cross_val_predict, validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, TargetEncoder, label_binarize,
)

from src.data_loader import load_train, RANDOM_STATE

# ---------- Constantes ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# File backend (não SQLite) — evita `database is locked` quando >1 jobs do array
# escrevem em paralelo. mlruns/ é uma pasta plana, cada run num subdir; sem
# contenção de I/O. Pra UI: `mlflow ui --backend-store-uri file://./mlruns`.
TRACKING_URI = f"file://{PROJECT_ROOT / 'mlruns'}"
EXPERIMENT_NAME = "triagem-dengue"
N_SPLITS = 5

# Module-level cache do experiment_id pra evitar lookup repetido + lidar com
# race condition no file backend (vários processos paralelos criando o exp).
_MLFLOW_EXPERIMENT_ID: Optional[str] = None


def _get_experiment_id() -> str:
    """Resolve experiment_id idempotentemente; cache module-level.

    `mlflow.set_experiment(name)` tem race condition no file backend quando
    vários processos rodam em paralelo (alguns caem pro Default id=0). Aqui
    usamos `MlflowClient.get_experiment_by_name` + `create_experiment` com
    re-fetch em caso de MlflowException (outro processo criou no meio).
    """
    global _MLFLOW_EXPERIMENT_ID
    if _MLFLOW_EXPERIMENT_ID is not None:
        return _MLFLOW_EXPERIMENT_ID
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        try:
            exp_id = client.create_experiment(EXPERIMENT_NAME)
        except mlflow.exceptions.MlflowException:
            # Outro processo criou no meio do caminho — re-fetch
            exp = client.get_experiment_by_name(EXPERIMENT_NAME)
            if exp is None:
                raise
            exp_id = exp.experiment_id
    else:
        exp_id = exp.experiment_id
    _MLFLOW_EXPERIMENT_ID = exp_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


# ---------- Building blocks ----------

def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """ColumnTransformer adaptativo: numéricas + categóricas auto-detectadas (OneHot)."""
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def build_preprocessor_target_encoding(X_train: pd.DataFrame,
                                       high_card_threshold: int = 5) -> ColumnTransformer:
    """Preprocessor com TargetEncoder pras categóricas de cardinalidade ≥ threshold.

    Categóricas com cardinalidade < threshold ficam com OneHot (mais estável).
    Numéricas: StandardScaler. Anti-leakage por design: o TargetEncoder do sklearn
    é sklearn-compatible (tem fit/transform) — quando dentro de um Pipeline + CV,
    cada fold treina o encoder só no train fold.

    Usado pela variante v3_target_enc.

    Dimensionalidade output (no dataset Triagem-Dengue 2026.1, pós-cleanup):
    - 3 categóricas alta cardinalidade (tp_gestante, tp_raca_cor, tp_escolaridade):
      TargetEncoder(target_type=multiclass) → 3 colunas por feature (1 por classe)
      = 3 × 3 = 9 colunas.
    - 1 categórica baixa cardinalidade (tp_sexo, 3 levels): OneHotEncoder → 3 colunas.
    - 22 numéricas (incluindo binárias 0/1 pós-cleanup): StandardScaler → 22 colunas.
    - **Total esperado: ~34 colunas** (vs 49 com OneHot puro do `build_preprocessor`).
    """
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    high_card = [c for c in categorical_cols if X_train[c].nunique() >= high_card_threshold]
    low_card = [c for c in categorical_cols if c not in high_card]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    high_card_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("target", TargetEncoder(target_type="multiclass", smooth="auto",
                                 cv=5, random_state=RANDOM_STATE)),
    ])
    low_card_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [("num", numeric_pipe, numeric_cols)]
    if high_card:
        transformers.append(("cat_high", high_card_pipe, high_card))
    if low_card:
        transformers.append(("cat_low", low_card_pipe, low_card))

    return ColumnTransformer(transformers)


def build_pipeline(
    X_train: pd.DataFrame,
    model: BaseEstimator,
    extra_steps: Optional[Sequence[tuple]] = None,
    preprocessor_builder: Optional[Callable[[pd.DataFrame], ColumnTransformer]] = None,
) -> Pipeline:
    """
    Monta [preprocessor] + [extra_steps] + [model].

    extra_steps é lista de tuplas (nome, transformer/resampler) inseridas
    ENTRE preprocessor e model. Usar pra resamplers (SMOTE, RandomUnderSampler)
    que devem rodar só na fase de fit. Se houver, usamos imblearn Pipeline.

    preprocessor_builder: callable opcional pra trocar o preprocessor default
    (OneHot). Ex: `build_preprocessor_target_encoding` pra v3_target_enc.
    Se None, usa build_preprocessor.
    """
    if preprocessor_builder is None:
        preprocessor_builder = build_preprocessor
    preprocess = preprocessor_builder(X_train)
    steps = [("preprocess", preprocess)]
    if extra_steps:
        steps.extend(extra_steps)
    steps.append(("model", model))

    PipelineClass = ImbPipeline if extra_steps else Pipeline
    return PipelineClass(steps)


def plot_validation_curve_fig(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    param_name: str,
    param_range: Sequence,
    title_suffix: str = "",
    n_jobs: int = -1,
) -> tuple:
    """Roda validation_curve e retorna (fig, dict com mean/std de train e val)."""
    train_scores, val_scores = validation_curve(
        pipe, X_train, y_train,
        param_name=param_name, param_range=param_range,
        cv=cv, scoring="f1_macro", n_jobs=n_jobs,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    # X axis: pra valores numéricos plota linha; pra strings/tuplas, usa índice
    try:
        xs = [float(v) for v in param_range]
        xticks = None
    except (TypeError, ValueError):
        xs = list(range(len(param_range)))
        xticks = [str(v) for v in param_range]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, train_mean, "o-", label="treino", color="C0")
    ax.fill_between(xs, train_mean - train_std, train_mean + train_std, alpha=0.15, color="C0")
    ax.plot(xs, val_mean, "o-", label="validação", color="C1")
    ax.fill_between(xs, val_mean - val_std, val_mean + val_std, alpha=0.15, color="C1")
    ax.set_xlabel(param_name.replace("model__", ""))
    ax.set_ylabel("F1-macro")
    title = f"Curva treino vs validação — {title_suffix}" if title_suffix else "Curva treino vs validação"
    ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(xs)
        ax.set_xticklabels(xticks, rotation=45 if max(len(s) for s in xticks) > 6 else 0)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig, {
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
        "val_mean": val_mean.tolist(),
        "val_std": val_std.tolist(),
        "param_range": list(param_range),
    }


# ---------- Runner principal ----------

def run_experiment(
    *,
    algorithm: str,
    variant: str,
    model_factory: Callable[[], BaseEstimator],
    param_grid: dict,
    search_method: str = "grid",
    n_iter: int = 30,
    extra_pipeline_steps: Optional[Sequence[tuple]] = None,
    preprocessor_builder: Optional[Callable[[pd.DataFrame], ColumnTransformer]] = None,
    main_hp_for_curve: Optional[str] = None,
    curve_range: Optional[Sequence] = None,
    cv_predict_n_jobs: int = 1,
    search_n_jobs: int = -1,
    extra_tags: Optional[dict] = None,
):
    """
    Roda 1 experimento end-to-end e loga tudo no MLflow.

    Parâmetros chave:
    - algorithm: nome do algoritmo (vai pro run_name e tag).
    - variant: v1_baseline / v2_balanceamento / v3_escala_codificacao / v4_fe.
    - model_factory: função que retorna instância nova do modelo (sem fit).
    - param_grid: dict pro GridSearchCV/RandomizedSearchCV.
    - search_method: "grid" | "random".
    - n_iter: número de combos pra RandomizedSearch (ignorado em grid).
    - extra_pipeline_steps: lista de tuplas (nome, step) inseridas ENTRE
      preprocess e model (pra resamplers — força imblearn Pipeline).
    - preprocessor_builder: callable opcional pra trocar o preprocessor default
      (build_preprocessor). Ex: build_preprocessor_target_encoding pra v3.
    - main_hp_for_curve: nome do HP pra validation_curve. Se None, pula a curva.
    - curve_range: valores do main_hp pra plotar (≤ ~10 pra ler bem).
    - cv_predict_n_jobs: 1 evita conflito de paralelismo com search_n_jobs=-1.
    - extra_tags: dict de tags adicionais pro MLflow.
    """
    # Setup MLflow — resolve experiment_id de forma idempotente.
    # Lógica extraída pra função module-level com cache (ver _get_experiment_id).
    _exp_id = _get_experiment_id()

    print(f"\n{'='*70}")
    print(f"EXPERIMENTO: {algorithm} | variante: {variant}")
    print(f"{'='*70}")

    # Load
    print("Carregando treino...")
    X_train, y_train = load_train()
    print(f"  X_train={X_train.shape}, classes={y_train.value_counts().to_dict()}")

    # Pipeline
    pipe = build_pipeline(X_train, model_factory(), extra_steps=extra_pipeline_steps, preprocessor_builder=preprocessor_builder)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Hyperparameter search
    n_combos = (
        np.prod([len(v) for v in param_grid.values()])
        if search_method == "grid"
        else n_iter
    )
    n_fits = n_combos * N_SPLITS
    print(f"\n{search_method.title()}SearchCV: {n_combos} combos × {N_SPLITS} folds = {n_fits} fits")

    if search_method == "grid":
        search = GridSearchCV(
            pipe, param_grid, cv=cv, scoring="f1_macro",
            n_jobs=search_n_jobs, return_train_score=True, verbose=1,
        )
    elif search_method == "random":
        search = RandomizedSearchCV(
            pipe, param_grid, n_iter=n_iter, cv=cv, scoring="f1_macro",
            n_jobs=search_n_jobs, return_train_score=True, verbose=1,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"search_method desconhecido: {search_method}")

    search.fit(X_train, y_train)
    best = search.best_params_
    best_score = search.best_score_
    best_std = search.cv_results_["std_test_score"][search.best_index_]
    best_idx = search.best_index_
    print(f"  Melhor F1-macro CV (search): {best_score:.4f} ± {best_std:.4f}")
    print(f"  Best params: {best}")

    # Scores por fold do best combo — habilita Wilcoxon pareado downstream
    # (precisamos comparar baseline vs variante usando os MESMOS folds).
    best_fold_scores = [
        float(search.cv_results_[f"split{i}_test_score"][best_idx])
        for i in range(N_SPLITS)
    ]
    print(f"  Best fold scores: {[f'{s:.4f}' for s in best_fold_scores]}")

    # Validation curve (opcional)
    curve_data = None
    fig_path = None
    if main_hp_for_curve and curve_range:
        print(f"\nValidation curve pra {main_hp_for_curve}...")
        # Pipeline com best params fixos exceto o main_hp
        best_pipe = build_pipeline(X_train, model_factory(), extra_steps=extra_pipeline_steps, preprocessor_builder=preprocessor_builder)
        params_for_curve = {k: v for k, v in best.items() if k != main_hp_for_curve}
        best_pipe.set_params(**params_for_curve)

        fig, curve_data = plot_validation_curve_fig(
            best_pipe, X_train, y_train, cv,
            param_name=main_hp_for_curve, param_range=curve_range,
            title_suffix=f"{algorithm} ({variant})",
        )
        fig_path = PROJECT_ROOT / "reports" / "figures" / f"{algorithm}_{variant}_validation_curve.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(f"  Curva salva: {fig_path}")

    # Predições CV pra métricas por classe (honesto)
    print(f"\ncross_val_predict pra métricas por classe...")
    best_pipe_final = build_pipeline(X_train, model_factory(), extra_steps=extra_pipeline_steps, preprocessor_builder=preprocessor_builder)
    best_pipe_final.set_params(**best)
    y_pred = cross_val_predict(best_pipe_final, X_train, y_train, cv=cv, n_jobs=cv_predict_n_jobs)

    f1_macro = f1_score(y_train, y_pred, average="macro")
    recall_per_class = recall_score(y_train, y_pred, average=None, labels=[0, 1, 2])

    print(f"\n--- Métricas ({algorithm} | {variant}) ---")
    print(f"F1-macro (CV):     {f1_macro:.4f}")
    print(f"Recall Descartado: {recall_per_class[0]:.4f}")
    print(f"Recall Comum:      {recall_per_class[1]:.4f}")
    print(f"Recall Alerta/G.:  {recall_per_class[2]:.4f}   ← chave")
    cm = confusion_matrix(y_train, y_pred, labels=[0, 1, 2])
    print(f"\nMatriz de confusão:")
    print(pd.DataFrame(cm, index=["0_real", "1_real", "2_real"], columns=["0_pred", "1_pred", "2_pred"]))

    # ROC-AUC e PR-AUC macro multiclasse (Critério 6 da rubrica, 8% peso).
    # Tenta predict_proba (preferido); fallback decision_function pra modelos como LinearSVC
    # NÃO wrapped em CalibratedClassifierCV.
    print(f"\nROC-AUC / PR-AUC macro (cross_val_predict probas)...")
    y_score = None
    score_method = None
    for method in ("predict_proba", "decision_function"):
        try:
            pipe_for_proba = build_pipeline(X_train, model_factory(),
                                            extra_steps=extra_pipeline_steps,
                                            preprocessor_builder=preprocessor_builder)
            pipe_for_proba.set_params(**best)
            y_score = cross_val_predict(pipe_for_proba, X_train, y_train, cv=cv,
                                        n_jobs=cv_predict_n_jobs, method=method)
            score_method = method
            break
        except (AttributeError, ValueError) as e:
            print(f"  skip {method}: {type(e).__name__}: {str(e)[:80]}")
            continue

    roc_auc_macro = None
    pr_auc_macro = None
    if y_score is not None:
        try:
            roc_auc_macro = float(roc_auc_score(y_train, y_score, multi_class="ovr",
                                                 average="macro", labels=[0, 1, 2]))
            y_bin = label_binarize(y_train, classes=[0, 1, 2])
            pr_auc_macro = float(average_precision_score(y_bin, y_score, average="macro"))
            print(f"  ROC-AUC macro (ovr): {roc_auc_macro:.4f}")
            print(f"  PR-AUC macro:        {pr_auc_macro:.4f}")
            print(f"  score_method:        {score_method}")
        except Exception as e:
            print(f"  ⚠️  ROC/PR falhou: {type(e).__name__}: {e}")
    else:
        print(f"  ⚠️  Sem predict_proba nem decision_function — ROC/PR não calculados")

    # MLflow log
    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run(experiment_id=_exp_id,
                          run_name=f"{algorithm}_{variant}"):
        mlflow.set_tag("algoritmo", algorithm)
        mlflow.set_tag("variante", variant)
        mlflow.set_tag("dataset_state", "post_cleanup_49feat")
        mlflow.set_tag("search_method", search_method)
        if extra_tags:
            for k, v in extra_tags.items():
                mlflow.set_tag(k, str(v))
        mlflow.log_params({k.replace("model__", ""): str(v) for k, v in best.items()})
        mlflow.log_metric("f1_macro_cv_search", float(best_score))
        mlflow.log_metric("f1_macro_cv_search_std", float(best_std))
        mlflow.log_metric("f1_macro_cv_predict", float(f1_macro))
        mlflow.log_metric("recall_descartado", float(recall_per_class[0]))
        mlflow.log_metric("recall_comum", float(recall_per_class[1]))
        mlflow.log_metric("recall_alerta_grave", float(recall_per_class[2]))
        # F1-macro por fold (best combo) — métricas individuais facilitam leitura
        # via MlflowClient sem precisar parsear o cv_results CSV.
        for i, s in enumerate(best_fold_scores):
            mlflow.log_metric(f"f1_macro_fold_{i}", s)
        # ROC-AUC e PR-AUC macro multiclasse (Critério 6 da rubrica).
        if roc_auc_macro is not None:
            mlflow.log_metric("roc_auc_macro_cv", roc_auc_macro)
            mlflow.log_metric("pr_auc_macro_cv", pr_auc_macro)
        if score_method:
            mlflow.set_tag("score_method", score_method)

        # Artifacts persistidos pro relatório / análise downstream
        figures_dir = PROJECT_ROOT / "reports" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # 1. cv_results_ completo — todos os combos × todos os folds + params
        cv_results_path = figures_dir / f"{algorithm}_{variant}_cv_results.csv"
        pd.DataFrame(search.cv_results_).to_csv(cv_results_path, index=False)
        mlflow.log_artifact(str(cv_results_path))

        # 2. Best fold scores como CSV simples — input direto pro Wilcoxon
        best_folds_path = figures_dir / f"{algorithm}_{variant}_best_fold_scores.csv"
        pd.DataFrame({"fold": list(range(N_SPLITS)),
                      "f1_macro": best_fold_scores}).to_csv(best_folds_path, index=False)
        mlflow.log_artifact(str(best_folds_path))

        # 3. Matriz de confusão como CSV (substitui parser de logs .out)
        cm_path = figures_dir / f"{algorithm}_{variant}_confusion_matrix.csv"
        pd.DataFrame(cm,
                     index=["0_real", "1_real", "2_real"],
                     columns=["0_pred", "1_pred", "2_pred"]).to_csv(cm_path)
        mlflow.log_artifact(str(cm_path))

        # 4. y_scores (probas multiclasse) — permite refit ROC/PR offline
        if y_score is not None:
            scores_path = figures_dir / f"{algorithm}_{variant}_y_scores.csv"
            pd.DataFrame(y_score, columns=["score_0", "score_1", "score_2"]).to_csv(
                scores_path, index=False)
            mlflow.log_artifact(str(scores_path))

        # 5. Validation curve (se houver)
        if fig_path:
            mlflow.log_artifact(str(fig_path))
        if curve_data:
            csv_path = figures_dir / f"{algorithm}_{variant}_validation_curve.csv"
            pd.DataFrame(curve_data).to_csv(csv_path, index=False)
            mlflow.log_artifact(str(csv_path))

        mlflow.sklearn.log_model(search.best_estimator_, name="model")

    print(f"\n✓ MLflow run logado: {algorithm}_{variant}")
    return {
        "best_params": best,
        "f1_macro_search": best_score,
        "f1_macro_predict": f1_macro,
        "best_fold_scores": best_fold_scores,
        "recall_per_class": recall_per_class.tolist(),
        "roc_auc_macro": roc_auc_macro,
        "pr_auc_macro": pr_auc_macro,
        "score_method": score_method,
        "fig_path": str(fig_path) if fig_path else None,
    }
