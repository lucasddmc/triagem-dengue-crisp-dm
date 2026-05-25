"""
Grid 2×5 de matrizes de confusão NORMALIZADAS (linhas = real) pros 10 baselines.

Parser dos logs `.out` do Apuana que printam a matriz via
`pd.DataFrame(cm, index=["0_real",...], columns=["0_pred",...])` em
`src/experiment_runner.py:298-300`.

Pré-requisito: rsync dos logs (ver docstring abaixo).

Saídas:
    reports/figures/confusion_matrices_v1_baseline.png  (grid 2×5)
    reports/figures/confusion_matrices_v1_baseline.csv  (formato longo, 90 linhas)

Uso:
    # 1) rsync dos logs do Apuana (uma vez):
    rsync -avz -e "ssh -i ~/.ssh/id_ed25519_apuana" \\
      'ldmc@slurm-client1.cin.ufpe.br:~/triagem-dengue/logs/triagem_*.out' \\
      ./mlflow_apuana/logs/

    # 2) Rodar:
    python scripts/build_confusion_matrices.py [--variant v1_baseline]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "mlflow_dengue.db"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
LOGS_DIR = PROJECT_ROOT / "mlflow_apuana" / "logs"
EXPERIMENT_NAME = "triagem-dengue"

# Mapeamento canônico — qual `.out` corresponde a cada algoritmo
ALGO_TO_LOG = {
    "decision_tree":  "triagem_664_0.out",
    "random_forest":  "triagem_664_2.out",
    "xgboost":        "triagem_664_3.out",
    "knn":            "triagem_664_5.out",
    "stacking":       "triagem_664_6.out",
    "rna_committee":  "triagem_664_7.out",
    "mlp":            "triagem_664_8.out",
    "lvq":            "triagem_680_4.out",
    "lightgbm":       "triagem_875_1.out",
    # SVM original (job 742) perdeu o log; matriz re-derivada localmente
    # via scripts/recover_svm_confusion_matrix.py (LinearSVC + best_params do MLflow,
    # F1 confere com 0.3921 do run original).
    "svm":            "triagem_svm_recovered.out",
}

# Ordem do grid 2×5: top row = pareto-ish (mais F1), bottom = restantes
GRID_ORDER = [
    "lightgbm", "xgboost", "decision_tree", "mlp", "stacking",
    "rna_committee", "random_forest", "lvq", "knn", "svm",
]

EXPECTED_TOTAL = 52213  # n_train pós-cleanup (sanity check)


def _setup_mlflow() -> MlflowClient:
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    return MlflowClient()


def parse_confusion_matrix(log_text: str) -> np.ndarray | None:
    """Extrai a matriz 3×3 do output `pd.DataFrame(cm, index=['0_real'...], columns=['0_pred'...])`.

    Formato esperado:
        Matriz de confusão:
                0_pred  1_pred  2_pred
        0_real   11307   11751      50
        1_real    7556   21136      92
        2_real     110     200      11

    Pega a ÚLTIMA ocorrência caso o log tenha rerun parcial.
    Retorna None se não achar.
    """
    pattern = re.compile(
        r"0_real\s+(\d+)\s+(\d+)\s+(\d+)\s*\n"
        r"\s*1_real\s+(\d+)\s+(\d+)\s+(\d+)\s*\n"
        r"\s*2_real\s+(\d+)\s+(\d+)\s+(\d+)"
    )
    matches = pattern.findall(log_text)
    if not matches:
        return None
    last = matches[-1]
    cm = np.array([int(x) for x in last], dtype=int).reshape(3, 3)
    return cm


def load_matrices() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    missing = []
    for algo, fname in ALGO_TO_LOG.items():
        path = LOGS_DIR / fname
        if not path.exists():
            print(f"  [warn] log não encontrado pra {algo}: {path}")
            missing.append(algo)
            continue
        cm = parse_confusion_matrix(path.read_text())
        if cm is None:
            print(f"  [warn] matriz não parseada pra {algo} em {fname}")
            missing.append(algo)
            continue
        total = int(cm.sum())
        if abs(total - EXPECTED_TOTAL) > 50:
            print(f"  [warn] {algo}: soma={total} (esperado ~{EXPECTED_TOTAL})")
        out[algo] = cm
        print(f"  ✓ {algo}: cm.sum()={total}")
    if missing:
        print(f"\n  Algos sem matriz ({len(missing)}): {missing}")
    return out


def fetch_f1_per_algo(variant: str, source: str = "apuana") -> dict[str, float]:
    client = _setup_mlflow()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return {}
    filter_str = f"tags.variante = '{variant}' AND tags.source = '{source}'"
    runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string=filter_str)
    return {r.data.tags.get("algoritmo", ""): r.data.metrics.get("f1_macro_cv_predict", float("nan"))
            for r in runs}


def _draw_one(ax: plt.Axes, algo: str, cm: np.ndarray | None, f1: float | None) -> None:
    if cm is None:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{algo}\n(log perdido)",
                ha="center", va="center", fontsize=11, color="#999")
        return

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    classes = ["Descart.", "Comum", "Alerta/Grave"]
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predito", fontsize=9)
    ax.set_ylabel("Real", fontsize=9)

    # Anotações: fração + count
    for i in range(3):
        for j in range(3):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if val > 0.5 else "#222"
            ax.text(j, i, f"{val:.2f}\n({count})",
                    ha="center", va="center", fontsize=8, color=color)

    f1_str = f"F1={f1:.4f}" if f1 is not None and not np.isnan(f1) else ""
    ax.set_title(f"{algo}  {f1_str}", fontsize=10, fontweight="bold")
    return im


def plot_grid(cms: dict[str, np.ndarray], f1s: dict[str, float]) -> plt.Figure:
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    axes_flat = axes.flatten()

    last_im = None
    for ax, algo in zip(axes_flat, GRID_ORDER):
        cm = cms.get(algo)
        f1 = f1s.get(algo)
        result = _draw_one(ax, algo, cm, f1)
        if result is not None:
            last_im = result

    fig.suptitle("Matrizes de confusão normalizadas (linhas = real) — v1 baseline\n"
                 "Anotação: fração da linha (count absoluto)",
                 fontsize=13, fontweight="bold")

    # Colorbar único compartilhado
    if last_im is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.70])
        fig.colorbar(last_im, cax=cbar_ax, label="fração da classe real")

    fig.tight_layout(rect=[0, 0, 0.92, 0.94])
    return fig


def build_long_csv(cms: dict[str, np.ndarray]) -> pd.DataFrame:
    classes = ["Descartado", "Comum", "Alerta/Grave"]
    rows = []
    for algo, cm in cms.items():
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        for i, real in enumerate(classes):
            for j, pred in enumerate(classes):
                rows.append({
                    "algoritmo": algo,
                    "real_class": real,
                    "pred_class": pred,
                    "count": int(cm[i, j]),
                    "count_normalized": float(cm_norm[i, j]),
                })
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="v1_baseline")
    args = p.parse_args(argv)

    if not LOGS_DIR.exists() or not list(LOGS_DIR.glob("triagem_*.out")):
        print(f"❌ Nenhum log em {LOGS_DIR}")
        print("   Rode o rsync (ver docstring do script).")
        return 1

    print(f"Parseando matrizes em {LOGS_DIR.relative_to(PROJECT_ROOT)}...")
    cms = load_matrices()
    if not cms:
        print("Nenhuma matriz parseada — abortando.")
        return 1
    print(f"\n  ✓ {len(cms)} matrizes carregadas: {sorted(cms.keys())}\n")

    f1s = fetch_f1_per_algo(args.variant)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_grid(cms, f1s)
    png_path = FIG_DIR / f"confusion_matrices_{args.variant}.png"
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura → {png_path.relative_to(PROJECT_ROOT)}")

    long_df = build_long_csv(cms)
    csv_path = FIG_DIR / f"confusion_matrices_{args.variant}.csv"
    long_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  ✓ CSV    → {csv_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
