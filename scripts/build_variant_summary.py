"""
Constrói tabela CSV + figura PNG comparativa pra uma variante (v1, v2, v3, v4).

Lê os runs do MLflow local filtrando por `tags.variante = <variant>` e
`tags.source = 'apuana'`, ordena por F1-macro CV (via cross_val_predict),
exporta:
  - reports/figures/tabela_{variant}_baselines.csv
  - reports/figures/barplot_{variant}_baselines.png

Uso:
    python scripts/build_variant_summary.py --variant v1_baseline
    python scripts/build_variant_summary.py --variant v2_class_weight  # depois
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "mlflow_dengue.db"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
EXPERIMENT_NAME = "triagem-dengue"


def load_variant_runs(variant_tag: str, source: str | None = "apuana") -> pd.DataFrame:
    """Retorna DataFrame com 1 linha por algoritmo da variante.

    Por default filtra também por `tags.source = 'apuana'` pra excluir runs
    locais antigos (smoke tests, validações iniciais). Passe `source=None`
    pra trazer tudo independente da origem.
    """
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"Experiment '{EXPERIMENT_NAME}' não existe em {DB_PATH}")

    filter_parts = [f"tags.variante = '{variant_tag}'"]
    if source is not None:
        filter_parts.append(f"tags.source = '{source}'")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=" AND ".join(filter_parts),
        max_results=10_000,
    )
    if not runs:
        raise SystemExit(f"Nenhum run com tags.variante = '{variant_tag}' no experimento.")

    rows = []
    for r in runs:
        rows.append({
            "algoritmo": r.data.tags.get("algoritmo", r.info.run_name),
            "run_name": r.info.run_name,
            "f1_macro_cv": r.data.metrics.get("f1_macro_cv_predict", float("nan")),
            "f1_macro_cv_search": r.data.metrics.get("f1_macro_cv_search", float("nan")),
            "f1_macro_cv_search_std": r.data.metrics.get("f1_macro_cv_search_std", float("nan")),
            "recall_descartado": r.data.metrics.get("recall_descartado", float("nan")),
            "recall_comum": r.data.metrics.get("recall_comum", float("nan")),
            "recall_alerta_grave": r.data.metrics.get("recall_alerta_grave", float("nan")),
            "search_method": r.data.tags.get("search_method", ""),
        })

    df = pd.DataFrame(rows).sort_values("f1_macro_cv", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def save_csv(df: pd.DataFrame, variant_tag: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"tabela_{variant_tag}.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    return out


def save_barplot(df: pd.DataFrame, variant_tag: str) -> Path:
    """Painel duplo: F1-macro (esquerda) + Recall Alerta/Grave (direita)."""
    # Ordem comum: maior F1 em cima
    df_plot = df.sort_values("f1_macro_cv", ascending=True).reset_index(drop=True)
    algos = df_plot["algoritmo"].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    # Painel 1 — F1-macro CV
    bars1 = ax1.barh(algos, df_plot["f1_macro_cv"], color="#4C72B0", edgecolor="white")
    ax1.set_xlabel("F1-macro (CV via cross_val_predict)")
    ax1.set_title("F1-macro por algoritmo", fontsize=11)
    ax1.axvline(df_plot["f1_macro_cv"].mean(), color="#888", linestyle="--", linewidth=1, alpha=0.7,
                label=f"média = {df_plot['f1_macro_cv'].mean():.4f}")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars1, df_plot["f1_macro_cv"]):
        ax1.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=8)
    ax1.set_xlim(0, max(df_plot["f1_macro_cv"]) * 1.15)

    # Painel 2 — Recall Alerta/Grave
    # Cor por threshold: vermelho se = 0, laranja se < 0.02, verde se >= 0.02
    colors = []
    for v in df_plot["recall_alerta_grave"]:
        if v <= 0.001:
            colors.append("#C44E52")  # vermelho — não detecta nada
        elif v < 0.02:
            colors.append("#DD8452")  # laranja — quase nada
        else:
            colors.append("#55A868")  # verde — algum sinal

    bars2 = ax2.barh(algos, df_plot["recall_alerta_grave"], color=colors, edgecolor="white")
    ax2.set_xlabel("Recall — classe Alerta/Grave (0.6% da base)")
    ax2.set_title("Recall na classe minoritária", fontsize=11)
    ax2.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars2, df_plot["recall_alerta_grave"]):
        ax2.text(val + max(df_plot["recall_alerta_grave"]) * 0.02 + 0.0003,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=8)
    upper = max(df_plot["recall_alerta_grave"].max() * 1.30, 0.05)
    ax2.set_xlim(0, upper)

    fig.suptitle(f"Baselines {variant_tag} — 10 algoritmos (ordenados por F1-macro)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = FIG_DIR / f"barplot_{variant_tag}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="v1_baseline",
                   help="Tag da variante (ex: v1_baseline, v2_class_weight). Default: v1_baseline.")
    p.add_argument("--source", default="apuana",
                   help="Filtra por tags.source (default: 'apuana'). Use 'all' pra trazer tudo.")
    args = p.parse_args()

    source = None if args.source == "all" else args.source
    df = load_variant_runs(args.variant, source=source)
    print(f"\n{len(df)} runs encontrados pra variante '{args.variant}':\n")
    print(df.drop(columns=["run_name", "search_method"]).to_string(index=False))
    print()

    csv_path = save_csv(df, args.variant)
    print(f"  ✓ CSV    → {csv_path.relative_to(PROJECT_ROOT)}")
    fig_path = save_barplot(df, args.variant)
    print(f"  ✓ Figura → {fig_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
