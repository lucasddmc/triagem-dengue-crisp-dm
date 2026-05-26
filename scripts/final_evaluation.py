"""
Avaliação final no conjunto de TESTE.

Esta é a única chamada autorizada de `load_test(UNLOCK_TOKEN)` em todo o
projeto — passo 11 do workflow CRISP-DM exigido pelo edital, critérios 7 e 8
da rubrica (Avaliação final + Deployment).

Fluxo:
1. Resolve o run MLflow do modelo final (default: lightgbm + v2_smote, vencedor
   empírico em F1-macro CV — 0.4413, rank_biserial = +1.000 vs v1_baseline).
2. Reconstrói o pipeline (preprocessor + extra_steps da variante + modelo com
   best_params) reusando helpers de `scripts/recover_fold_scores.py`.
3. Fita no `X_train` completo (sem CV — modelo final usa todo o treino).
4. Libera sentinel: `X_test, y_test = load_test(UNLOCK_TOKEN)`.
5. Computa métricas (F1-macro, balanced_accuracy, accuracy, ROC-AUC macro,
   PR-AUC macro, métricas per-class) e gera artifacts:
   - `final_test_confusion_matrix.png` (absoluta + normalizada por linha)
   - `final_test_roc_pr.png` (ROC OvR per-class + macro; PR per-class + macro)
   - `final_test_classification_report.txt`
   - `final_test_y_pred.csv` (y_true, y_pred, proba_0..2)
6. Loga MLflow run novo: tags.variante=`final_test`, tags.source=`local_final`,
   tags.base_run_id=<id do run CV original>, métricas + params + artifacts.
7. Imprime resumo + Δ vs CV (generalization gap).

Classes (de `src/data_loader.py`):
- 0 = Descartado (códigos 5)
- 1 = Comum (códigos 1, 10)
- 2 = Alerta/Grave (códigos 11, 12, 2, 3, 4) — minoritária, foco clínico

Uso:
    python scripts/final_evaluation.py                              # lightgbm v2_smote
    python scripts/final_evaluation.py --algo xgboost --variant v2_smote
    python scripts/final_evaluation.py --dry-run                    # plano sem rodar
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import UNLOCK_TOKEN, load_test, load_train  # noqa: E402
from src.experiment_runner import build_pipeline  # noqa: E402

# Reusa toda a lógica de reconstrução de pipeline
from scripts.recover_fold_scores import (  # noqa: E402
    _build_factory,
    _build_variant_steps,
)

DB_PATH = PROJECT_ROOT / "mlflow_dengue.db"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
EXPERIMENT_NAME = "triagem-dengue"
TRACKING_URI = f"sqlite:///{DB_PATH}"

CLASS_NAMES = ["Descartado", "Comum", "Alerta/Grave"]
CLASS_LABELS = [0, 1, 2]


# ---------- MLflow lookup ----------

def find_base_run(client: MlflowClient, exp_id: str, algo: str, variant: str):
    """Acha o run MLflow correspondente (algo + variant + source=apuana)."""
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string=(
            f"tags.algoritmo = '{algo}' AND "
            f"tags.variante = '{variant}' AND "
            f"tags.source = 'apuana'"
        ),
        max_results=5,
    )
    if not runs:
        raise RuntimeError(
            f"Nenhum run encontrado com algo={algo}, variante={variant}, "
            f"source=apuana. Confira `tags.source` (sync já rodou?)."
        )
    if len(runs) > 1:
        print(f"⚠ {len(runs)} runs candidatos — usando o mais recente.")
    # Mais recente
    return sorted(runs, key=lambda r: r.info.start_time, reverse=True)[0]


# ---------- Plots ----------

def plot_confusion_matrices(y_true, y_pred, out_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """2 subplots: absoluta + normalizada-por-linha (recall por classe)."""
    cm_abs = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    cm_norm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title, fmt, vmax in [
        (axes[0], cm_abs, "Absoluta", "d", None),
        (axes[1], cm_norm, "Normalizada por classe real (recall)", ".2f", 1.0),
    ]:
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, fontsize=9, rotation=15)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predita", fontsize=10)
        ax.set_ylabel("Real", fontsize=10)
        ax.set_title(title, fontsize=11)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                value = cm[i, j]
                text = f"{int(value)}" if fmt == "d" else f"{value:.2f}"
                color = "white" if (cm_norm[i, j] if vmax else cm_abs[i, j] / cm_abs.max()) > 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Matriz de confusão — modelo final no teste", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return cm_abs, cm_norm


def plot_roc_pr(y_true, y_proba, out_path: Path) -> dict:
    """1×2 subplots: ROC OvR + PR OvR, com macro average. Retorna AUCs."""
    y_bin = label_binarize(y_true, classes=CLASS_LABELS)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))

    # ROC per-class
    fpr_grid = np.linspace(0, 1, 200)
    tpr_interp = np.zeros_like(fpr_grid)
    roc_auc_per_class = {}
    for k, name in zip(CLASS_LABELS, CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, k], y_proba[:, k])
        auc_k = roc_auc_score(y_bin[:, k], y_proba[:, k])
        roc_auc_per_class[k] = auc_k
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={auc_k:.3f})", linewidth=1.5)
        tpr_interp += np.interp(fpr_grid, fpr, tpr)
    tpr_interp /= len(CLASS_LABELS)
    auc_macro = roc_auc_score(y_true, y_proba, multi_class="ovr",
                              average="macro", labels=CLASS_LABELS)
    ax_roc.plot(fpr_grid, tpr_interp, "k--",
                label=f"Macro (AUC={auc_macro:.3f})", linewidth=2)
    ax_roc.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=0.8)
    ax_roc.set_xlabel("FPR", fontsize=10)
    ax_roc.set_ylabel("TPR (recall)", fontsize=10)
    ax_roc.set_title("ROC — one-vs-rest", fontsize=11)
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(alpha=0.3)

    # PR per-class
    pr_auc_per_class = {}
    for k, name in zip(CLASS_LABELS, CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(y_bin[:, k], y_proba[:, k])
        ap_k = average_precision_score(y_bin[:, k], y_proba[:, k])
        pr_auc_per_class[k] = ap_k
        ax_pr.plot(recall, precision, label=f"{name} (AP={ap_k:.3f})", linewidth=1.5)
        # No-skill baseline = prevalência da classe
        prev = y_bin[:, k].mean()
        ax_pr.axhline(prev, linestyle=":", linewidth=0.6, alpha=0.5,
                      color=ax_pr.lines[-1].get_color())
    ap_macro = average_precision_score(y_bin, y_proba, average="macro")
    ax_pr.set_xlabel("Recall", fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.set_title(f"Precision-Recall — macro AP={ap_macro:.3f}", fontsize=11)
    ax_pr.legend(loc="lower left", fontsize=9)
    ax_pr.grid(alpha=0.3)

    fig.suptitle("Curvas ROC e PR — modelo final no teste", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return {
        "roc_auc_macro": float(auc_macro),
        "pr_auc_macro": float(ap_macro),
        "roc_auc_per_class": {int(k): float(v) for k, v in roc_auc_per_class.items()},
        "pr_auc_per_class": {int(k): float(v) for k, v in pr_auc_per_class.items()},
    }


# ---------- Main ----------

def run(algo: str, variant: str, dry_run: bool = False) -> int:
    print(f"=== Avaliação final no teste ===")
    print(f"Algoritmo: {algo}")
    print(f"Variante:  {variant}")
    print(f"Dry-run:   {dry_run}\n")

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print(f"❌ Experimento '{EXPERIMENT_NAME}' não existe no MLflow.")
        return 1
    exp_id = exp.experiment_id

    # 1. Acha run base
    base_run = find_base_run(client, exp_id, algo, variant)
    base_rid = base_run.info.run_id
    f1_cv = base_run.data.metrics.get("f1_macro_cv_search", float("nan"))
    print(f"Base run: {base_run.info.run_name} (run_id={base_rid[:8]})")
    print(f"F1-macro CV (busca):  {f1_cv:.4f}")

    if dry_run:
        print("\n[dry-run] viria fittar + load_test + predict + log. Encerrando.")
        return 0

    # 2. Carrega train + reconstrói pipeline
    print("\n[1/5] Carregando treino...")
    X_train, y_train = load_train()
    print(f"      train: {X_train.shape}, y dist: {pd.Series(y_train).value_counts().to_dict()}")

    print("[2/5] Reconstruindo pipeline...")
    model = _build_factory(algo, base_run.data.params)
    extra_steps, preprocessor_builder = _build_variant_steps(variant)
    pipe = build_pipeline(X_train, model, extra_steps=extra_steps,
                          preprocessor_builder=preprocessor_builder)

    # 3. Fit no train completo
    print("[3/5] Treinando no train completo (sem CV)...")
    t0 = time.time()
    pipe.fit(X_train, y_train)
    print(f"      ✓ fit em {time.time()-t0:.1f}s")

    # 4. SENTINELA: única chamada autorizada de load_test
    print("[4/5] >>> LIBERANDO SENTINEL — load_test() <<<")
    X_test, y_test = load_test(UNLOCK_TOKEN)
    print(f"      test: {X_test.shape}, y dist: {pd.Series(y_test).value_counts().to_dict()}")

    y_pred = pipe.predict(X_test)
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)
    else:
        raise RuntimeError(f"{algo} não tem predict_proba — não consegue gerar ROC/PR.")

    # 5. Métricas
    print("[5/5] Calculando métricas + gerando artifacts...")
    metrics = {
        "f1_macro_test": float(f1_score(y_test, y_pred, average="macro", labels=CLASS_LABELS)),
        "balanced_accuracy_test": float(balanced_accuracy_score(y_test, y_pred)),
        "accuracy_test": float(accuracy_score(y_test, y_pred)),
    }
    # Per-class
    p_per = precision_score(y_test, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    r_per = recall_score(y_test, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    f_per = f1_score(y_test, y_pred, labels=CLASS_LABELS, average=None, zero_division=0)
    for k in CLASS_LABELS:
        metrics[f"precision_class_{k}"] = float(p_per[k])
        metrics[f"recall_class_{k}"]    = float(r_per[k])
        metrics[f"f1_class_{k}"]        = float(f_per[k])

    # Artifacts
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cm_path  = FIG_DIR / "final_test_confusion_matrix.png"
    roc_path = FIG_DIR / "final_test_roc_pr.png"
    report_path = FIG_DIR / "final_test_classification_report.txt"
    csv_path = FIG_DIR / "final_test_y_pred.csv"
    cm_csv   = FIG_DIR / "final_test_confusion_matrix.csv"

    cm_abs, cm_norm = plot_confusion_matrices(y_test, y_pred, cm_path)
    roc_pr_metrics = plot_roc_pr(y_test, y_proba, roc_path)
    metrics.update({
        "roc_auc_macro_test": roc_pr_metrics["roc_auc_macro"],
        "pr_auc_macro_test":  roc_pr_metrics["pr_auc_macro"],
    })
    for k in CLASS_LABELS:
        metrics[f"roc_auc_class_{k}"] = roc_pr_metrics["roc_auc_per_class"][k]
        metrics[f"pr_auc_class_{k}"]  = roc_pr_metrics["pr_auc_per_class"][k]

    # Classification report (texto + CSV)
    report_text = classification_report(y_test, y_pred, labels=CLASS_LABELS,
                                         target_names=CLASS_NAMES, digits=4,
                                         zero_division=0)
    report_path.write_text(report_text, encoding="utf-8")

    # CM CSV (normalizada)
    pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(cm_csv)

    # y_pred CSV
    df_pred = pd.DataFrame({
        "y_true": y_test.values if hasattr(y_test, "values") else y_test,
        "y_pred": y_pred,
        "proba_0": y_proba[:, 0],
        "proba_1": y_proba[:, 1],
        "proba_2": y_proba[:, 2],
    })
    df_pred.to_csv(csv_path, index=False)

    # MLflow log
    with mlflow.start_run(experiment_id=exp_id, run_name=f"final_test__{algo}_{variant}") as final_run:
        # Tags
        mlflow.set_tags({
            "variante": "final_test",
            "source": "local_final",
            "algoritmo": algo,
            "base_variante": variant,
            "base_run_id": base_rid,
        })
        # Params (copiados do base)
        for k, v in base_run.data.params.items():
            mlflow.log_param(k, v)
        # Métricas + CV reference + gap
        mlflow.log_metric("f1_macro_cv_search_base", f1_cv)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        gap = metrics["f1_macro_test"] - f1_cv
        mlflow.log_metric("f1_generalization_gap", gap)
        # Artifacts
        for p in [cm_path, roc_path, report_path, csv_path, cm_csv]:
            mlflow.log_artifact(str(p))
        final_rid = final_run.info.run_id

    # ---------- Stdout summary ----------
    print("\n" + "=" * 70)
    print(f"  Modelo final: {algo} + {variant}")
    print(f"  Base run_id:  {base_rid[:8]}")
    print(f"  Final run_id: {final_rid[:8]}")
    print("=" * 70)
    print(f"  F1-macro CV (busca):      {f1_cv:.4f}")
    print(f"  F1-macro TEST:            {metrics['f1_macro_test']:.4f}")
    print(f"  Δ (gap test - CV):        {gap:+.4f}")
    print(f"  Balanced accuracy TEST:   {metrics['balanced_accuracy_test']:.4f}")
    print(f"  Accuracy TEST:            {metrics['accuracy_test']:.4f}")
    print(f"  ROC-AUC macro TEST:       {metrics['roc_auc_macro_test']:.4f}")
    print(f"  PR-AUC macro TEST:        {metrics['pr_auc_macro_test']:.4f}")
    print()
    print(f"  Per-class:")
    for k, name in zip(CLASS_LABELS, CLASS_NAMES):
        print(f"    [{k}] {name:<14} F1={metrics[f'f1_class_{k}']:.4f}  "
              f"P={metrics[f'precision_class_{k}']:.4f}  "
              f"R={metrics[f'recall_class_{k}']:.4f}")
    print()
    print("  Classification report:")
    print(report_text)
    print(f"  Artifacts em: {FIG_DIR}")
    print(f"    - {cm_path.name}")
    print(f"    - {roc_path.name}")
    print(f"    - {report_path.name}")
    print(f"    - {csv_path.name}")
    print(f"    - {cm_csv.name}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", default="lightgbm",
                   help="Algoritmo final (default: lightgbm — vencedor empírico)")
    p.add_argument("--variant", default="v2_smote",
                   help="Variante final (default: v2_smote — única com efeitos positivos consistentes)")
    p.add_argument("--dry-run", action="store_true",
                   help="Mostra plano sem rodar (não libera sentinel)")
    args = p.parse_args(argv)
    return run(args.algo, args.variant, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
