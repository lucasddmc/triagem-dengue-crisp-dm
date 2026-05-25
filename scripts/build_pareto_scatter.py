"""
Scatter F1-macro × Recall Alerta/Grave + fronteira de Pareto pros 10 baselines.

Visualiza o trade-off central observado em v1: modelos não conseguem alta F1-macro
E alto recall na classe minoritária simultaneamente. Identifica os pontos
não-dominados (fronteira de Pareto) — motivação visual pra v2 (balanceamento).

Lê `reports/figures/tabela_v1_baseline.csv` (gerado por `build_variant_summary.py`).

Saídas:
    reports/figures/scatter_pareto_v1_baseline.png
    reports/figures/scatter_pareto_v1_baseline.csv

Uso:
    python scripts/build_pareto_scatter.py [--variant v1_baseline]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# Offsets manuais pros labels (10 pontos, evita sobreposição visual)
# Ajustar caso valores mudem muito.
LABEL_OFFSETS = {
    "lightgbm":      (+0.0015, +0.0015),
    "xgboost":       (+0.0015, -0.0020),
    "decision_tree": (-0.0040, +0.0020),
    "mlp":           (+0.0015, -0.0020),
    "stacking":      (-0.0040, -0.0020),
    "rna_committee": (-0.0050, +0.0010),
    "random_forest": (+0.0015, +0.0010),
    "lvq":           (-0.0020, +0.0020),
    "knn":           (+0.0015, +0.0010),
    "svm":           (+0.0015, +0.0010),
}


def compute_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    """Retorna boolean Series indicando pontos Pareto-ótimos (maximização nos 2 eixos).

    Ponto `i` é Pareto se NÃO existe `j != i` com:
        df[x][j] >= df[x][i] AND df[y][j] >= df[y][i] AND (df[x][j] > df[x][i] OR df[y][j] > df[y][i])
    """
    n = len(df)
    xs = df[x_col].to_numpy()
    ys = df[y_col].to_numpy()
    pareto = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if (xs[j] >= xs[i] and ys[j] >= ys[i]
                    and (xs[j] > xs[i] or ys[j] > ys[i])):
                dominated = True
                break
        pareto.append(not dominated)
    return pd.Series(pareto, index=df.index)


def plot_pareto_scatter(df: pd.DataFrame, variant: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 7))

    # Região "aceitável" (recall_grave >= 0.02) sombreada
    ymax = max(df["recall_alerta_grave"].max() * 1.4, 0.05)
    ax.axhspan(0.02, ymax, alpha=0.07, color="#55A868", zorder=0)
    ax.axhline(0.02, color="#55A868", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(df["f1_macro_cv"].min() + 0.001, 0.021, "região aceitável (recall_grave ≥ 0.02)",
            fontsize=8, color="#55A868", style="italic")

    # Dominados (cinza, marker menor)
    dom = df[~df["pareto_optimal"]]
    ax.scatter(dom["f1_macro_cv"], dom["recall_alerta_grave"],
               color="#888888", s=80, marker="o", edgecolor="white", linewidth=1.2,
               zorder=2, label="dominado")

    # Pareto-ótimos (verde, marker maior)
    par = df[df["pareto_optimal"]].sort_values("f1_macro_cv")
    ax.scatter(par["f1_macro_cv"], par["recall_alerta_grave"],
               color="#55A868", s=180, marker="o", edgecolor="white", linewidth=1.5,
               zorder=3, label="Pareto-ótimo")

    # Linha conectando Pareto-ótimos (step plot — convenção pra fronteira discreta)
    if len(par) >= 2:
        ax.plot(par["f1_macro_cv"], par["recall_alerta_grave"],
                drawstyle="steps-post", color="#55A868", linestyle="--",
                linewidth=1.5, alpha=0.7, zorder=1)

    # Anotações com offsets manuais
    for _, row in df.iterrows():
        algo = row["algoritmo"]
        dx, dy = LABEL_OFFSETS.get(algo, (+0.0015, +0.0015))
        ax.annotate(algo,
                    xy=(row["f1_macro_cv"], row["recall_alerta_grave"]),
                    xytext=(row["f1_macro_cv"] + dx, row["recall_alerta_grave"] + dy),
                    fontsize=9, fontweight="bold" if row["pareto_optimal"] else "normal",
                    color="#1d6e3a" if row["pareto_optimal"] else "#333",
                    zorder=4)

    # Eixos
    xmin = df["f1_macro_cv"].min() - 0.005
    xmax = df["f1_macro_cv"].max() + 0.010
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.003, ymax)
    ax.set_xlabel("F1-macro CV (cross_val_predict)", fontsize=11)
    ax.set_ylabel("Recall — classe Alerta/Grave (0.6% da base)", fontsize=11)
    ax.set_title(f"Trade-off F1-macro × Recall Alerta/Grave — {variant}\n"
                 f"Pareto-ótimos: {', '.join(par['algoritmo'].tolist())}",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", default="v1_baseline")
    args = p.parse_args(argv)

    csv_in = FIG_DIR / f"tabela_{args.variant}.csv"
    if not csv_in.exists():
        print(f"❌ Arquivo {csv_in} não existe.")
        print(f"   Rode primeiro: python scripts/build_variant_summary.py --variant {args.variant}")
        return 1

    df = pd.read_csv(csv_in)
    df["pareto_optimal"] = compute_pareto_frontier(df, "f1_macro_cv", "recall_alerta_grave")

    print(f"\n{len(df)} algoritmos analisados:")
    print(df[["algoritmo", "f1_macro_cv", "recall_alerta_grave", "pareto_optimal"]]
          .to_string(index=False))

    pareto_algos = df.loc[df["pareto_optimal"], "algoritmo"].tolist()
    print(f"\nPareto-ótimos ({len(pareto_algos)}): {pareto_algos}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_pareto_scatter(df, args.variant)
    png_path = FIG_DIR / f"scatter_pareto_{args.variant}.png"
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    csv_out = FIG_DIR / f"scatter_pareto_{args.variant}.csv"
    df[["algoritmo", "f1_macro_cv", "recall_alerta_grave", "pareto_optimal"]] \
        .to_csv(csv_out, index=False, float_format="%.4f")

    print(f"\n  ✓ Figura → {png_path.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ CSV    → {csv_out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
