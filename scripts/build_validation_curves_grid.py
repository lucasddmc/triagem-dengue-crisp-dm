"""
Grid 3×3 das validation curves (treino vs validação) dos 9 algoritmos com HP
escalar único. Stacking fica fora (sem `main_hp_for_curve`).

Lê os artifacts `*_validation_curve.csv` já gravados pelo `experiment_runner`
de cada run no MLflow local (sincronizado do Apuana via
`sync_mlflow_from_apuana.py`). Adicionalmente computa o **gap train-val no best
param** pra rastrear overfit comparável entre modelos.

Saídas:
    reports/figures/validation_curves_grid_v1_baseline.png
    reports/figures/overfit_gap_v1_baseline.csv

Uso:
    python scripts/build_validation_curves_grid.py [--variant v1_baseline] [--source apuana]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "mlflow_dengue.db"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
APUANA_MLRUNS = PROJECT_ROOT / "mlflow_apuana" / "mlruns"
EXPERIMENT_NAME = "triagem-dengue"

# Espelha `main_hp_for_curve` dos `experiments/*.py`. Stacking não tem (None).
ALGO_TO_HP = {
    "decision_tree": "max_depth",
    "knn": "n_neighbors",
    "lvq": "prototype_n_per_class",
    "lightgbm": "num_leaves",
    "mlp": "alpha",
    "random_forest": "n_estimators",
    "rna_committee": "n_estimators",
    "svm": "C",
    "xgboost": "max_depth",
}

# Ordem do grid 3×3 (linha-a-linha, esquerda pra direita)
GRID_ORDER = [
    "decision_tree", "random_forest", "knn",
    "xgboost", "lightgbm", "mlp",
    "rna_committee", "lvq", "svm",
]


def _setup_tracking() -> MlflowClient:
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    return MlflowClient()


def _find_curve_csv_local(algo: str, variant: str) -> Path | None:
    """Fallback se download_artifacts() falhar — busca direto em mlflow_apuana/."""
    matches = list(APUANA_MLRUNS.glob(f"*/*/artifacts/{algo}_{variant}_validation_curve.csv"))
    return matches[0] if matches else None


def load_curve_artifacts(variant: str, source: str | None = "apuana") -> dict[str, dict]:
    """Retorna dict {algo: {"df": pd.DataFrame, "best_param": value, "run_id": str}}.

    Tenta MlflowClient.download_artifacts primeiro; em caso de erro
    (paths absolutos do Apuana no SQLite), fallback pra glob local.
    """
    client = _setup_tracking()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"Experiment '{EXPERIMENT_NAME}' não existe em {DB_PATH}")

    filter_parts = [f"tags.variante = '{variant}'"]
    if source is not None:
        filter_parts.append(f"tags.source = '{source}'")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=" AND ".join(filter_parts),
        max_results=10_000,
    )

    out: dict[str, dict] = {}
    for r in runs:
        algo = r.data.tags.get("algoritmo", "")
        if algo not in ALGO_TO_HP:
            continue  # stacking ou outro sem main_hp
        hp_name = ALGO_TO_HP[algo]
        best_param_raw = r.data.params.get(hp_name, None)

        artifact_name = f"{algo}_{variant}_validation_curve.csv"
        # Tenta path local (que MLflow apontou)
        csv_path: Path | None = None
        try:
            local = client.download_artifacts(r.info.run_id, artifact_name)
            csv_path = Path(local)
            if not csv_path.exists():
                csv_path = None
        except Exception:
            csv_path = None
        if csv_path is None:
            csv_path = _find_curve_csv_local(algo, variant)
        if csv_path is None or not csv_path.exists():
            print(f"  [warn] CSV de validation curve não achado pra {algo} (run {r.info.run_id[:8]})")
            continue

        df = pd.read_csv(csv_path)
        # cast best_param tentativo float; senão string
        best_param: float | str | None
        if best_param_raw is None:
            best_param = None
        else:
            try:
                best_param = float(best_param_raw)
                if best_param.is_integer():
                    best_param = int(best_param)
            except ValueError:
                best_param = str(best_param_raw)

        out[algo] = {"df": df, "best_param": best_param, "run_id": r.info.run_id, "hp_name": hp_name}

    missing = [a for a in ALGO_TO_HP if a not in out]
    if missing:
        print(f"  [warn] {len(missing)} algos sem curve: {missing}")
    return out


def _plot_one(ax: plt.Axes, algo: str, info: dict) -> None:
    df = info["df"]
    hp = info["hp_name"]
    best = info["best_param"]

    # X numérico ou indexado
    try:
        xs = [float(v) for v in df["param_range"]]
        xticks = None
    except (TypeError, ValueError):
        xs = list(range(len(df)))
        xticks = [str(v) for v in df["param_range"]]

    train_m = df["train_mean"].to_numpy()
    train_s = df["train_std"].to_numpy()
    val_m = df["val_mean"].to_numpy()
    val_s = df["val_std"].to_numpy()

    ax.plot(xs, train_m, "o-", label="treino", color="C0", linewidth=1.5, markersize=4)
    ax.fill_between(xs, train_m - train_s, train_m + train_s, alpha=0.15, color="C0")
    ax.plot(xs, val_m, "o-", label="validação", color="C1", linewidth=1.5, markersize=4)
    ax.fill_between(xs, val_m - val_s, val_m + val_s, alpha=0.15, color="C1")

    # Marca o best — escolhe o ponto mais próximo
    if best is not None:
        try:
            best_f = float(best)
            distances = [abs(float(v) - best_f) for v in df["param_range"]]
            best_idx = int(np.argmin(distances))
            ax.axvline(xs[best_idx], color="#444", linestyle=":", linewidth=1, alpha=0.8)
        except (TypeError, ValueError):
            pass

    ax.set_xlabel(hp, fontsize=9)
    ax.set_ylabel("F1-macro", fontsize=9)
    ax.set_title(f"{algo}  (best {hp}={best})", fontsize=10, fontweight="bold")
    if xticks is not None:
        ax.set_xticks(xs)
        ax.set_xticklabels(xticks, rotation=45 if max(len(s) for s in xticks) > 6 else 0,
                           fontsize=8)
    else:
        ax.tick_params(axis="both", labelsize=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")


def plot_grid(curves: dict[str, dict]) -> plt.Figure:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5))
    axes = axes.flatten()

    for ax, algo in zip(axes, GRID_ORDER):
        if algo in curves:
            _plot_one(ax, algo, curves[algo])
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{algo}\n(sem dados)", ha="center", va="center",
                    fontsize=10, color="#999")

    fig.suptitle("Validation curves treino vs validação — v1 baseline\n"
                 "(Stacking omitido — sem HP escalar único)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def compute_overfit_gap(curves: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for algo, info in curves.items():
        df = info["df"]
        best = info["best_param"]
        hp = info["hp_name"]

        # Localiza o ponto mais próximo do best param no curve_range
        best_in_range = False
        idx = None
        if best is not None:
            try:
                best_f = float(best)
                distances = [abs(float(v) - best_f) for v in df["param_range"]]
                idx = int(np.argmin(distances))
                # "in_range" = best param tá entre os valores plotados (tolerância numérica)
                best_in_range = distances[idx] < 1e-6
            except (TypeError, ValueError):
                idx = None

        if idx is None:
            rows.append({
                "algoritmo": algo, "hp_name": hp, "best_value": best,
                "train_at_best": np.nan, "val_at_best": np.nan,
                "gap_train_val": np.nan, "best_in_curve_range": False,
            })
            continue

        train_at = float(df["train_mean"].iloc[idx])
        val_at = float(df["val_mean"].iloc[idx])
        rows.append({
            "algoritmo": algo, "hp_name": hp, "best_value": best,
            "train_at_best": train_at, "val_at_best": val_at,
            "gap_train_val": train_at - val_at,
            "best_in_curve_range": best_in_range,
        })

    df_gap = pd.DataFrame(rows).sort_values("gap_train_val", ascending=False).reset_index(drop=True)
    return df_gap


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="v1_baseline")
    p.add_argument("--source", default="apuana", help="Use 'all' pra não filtrar.")
    args = p.parse_args(argv)

    source = None if args.source == "all" else args.source
    print(f"Carregando curves para variante='{args.variant}', source='{source}'...")
    curves = load_curve_artifacts(args.variant, source=source)
    print(f"  ✓ {len(curves)} curves carregadas: {sorted(curves.keys())}\n")

    if not curves:
        print("Nenhuma curva encontrada — abortando.")
        return 1

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Figura
    fig = plot_grid(curves)
    png_path = FIG_DIR / f"validation_curves_grid_{args.variant}.png"
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figura → {png_path.relative_to(PROJECT_ROOT)}")

    # CSV gap
    gap_df = compute_overfit_gap(curves)
    csv_path = FIG_DIR / f"overfit_gap_{args.variant}.csv"
    gap_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  ✓ CSV    → {csv_path.relative_to(PROJECT_ROOT)}")
    print("\nGap train-val no best param (decrescente):")
    print(gap_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
