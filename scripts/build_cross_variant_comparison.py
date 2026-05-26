"""
Comparação cross-variante — barras agrupadas baseline (v1) + variantes (v2/v3/v4).

Lê `reports/figures/tabela_{variant}.csv` (gerados por `build_variant_summary.py`)
pras 4 variantes e produz figura única com 2 subplots:
1. F1-macro CV por algoritmo × variante (slide 12 do template Beamer).
2. Recall Alerta/Grave (classe minoritária, foco clínico) por algoritmo × variante.

Saída:
    reports/figures/cross_variant_comparison.png
    reports/figures/cross_variant_comparison.csv  (long-format pra inspeção)

Uso:
    python scripts/build_cross_variant_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

VARIANTS = ["v1_baseline", "v2_smote", "v3_target_enc", "v4_selectk"]
VARIANT_LABELS = {
    "v1_baseline":   "v1 — baseline",
    "v2_smote":      "v2 — SMOTE",
    "v3_target_enc": "v3 — TargetEncoder",
    "v4_selectk":    "v4 — SelectKBest",
}
# Cores discretas, alta legibilidade em projetor
COLORS = {
    "v1_baseline":   "#4C72B0",  # azul
    "v2_smote":      "#DD8452",  # laranja
    "v3_target_enc": "#55A868",  # verde
    "v4_selectk":    "#C44E52",  # vermelho
}


def main():
    # Carrega 4 CSVs e empilha em long-format
    rows = []
    for v in VARIANTS:
        path = FIG_DIR / f"tabela_{v}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Faltou rodar build_variant_summary.py --variant {v}")
        df = pd.read_csv(path)
        df["variant"] = v
        rows.append(df)
    full = pd.concat(rows, ignore_index=True)

    # Long-format inspeção
    out_csv = FIG_DIR / "cross_variant_comparison.csv"
    cols_keep = ["variant", "algoritmo", "f1_macro_cv", "f1_macro_cv_search",
                 "recall_alerta_grave", "recall_descartado", "recall_comum"]
    full[cols_keep].to_csv(out_csv, index=False)

    # Pivot pra plot
    f1_pivot = full.pivot(index="algoritmo", columns="variant", values="f1_macro_cv")
    rec_pivot = full.pivot(index="algoritmo", columns="variant", values="recall_alerta_grave")

    # Ordena algoritmos por F1 do baseline (descendente) pra leitura fácil
    algo_order = f1_pivot["v1_baseline"].sort_values(ascending=False).index.tolist()
    f1_pivot = f1_pivot.reindex(algo_order)[VARIANTS]
    rec_pivot = rec_pivot.reindex(algo_order)[VARIANTS]

    # Plot
    fig, (ax_f1, ax_rec) = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(algo_order))
    bar_width = 0.20

    for i, v in enumerate(VARIANTS):
        offset = (i - 1.5) * bar_width  # centraliza grupo
        ax_f1.bar(x + offset, f1_pivot[v].values, bar_width,
                  label=VARIANT_LABELS[v], color=COLORS[v], edgecolor="white", linewidth=0.5)
        ax_rec.bar(x + offset, rec_pivot[v].values, bar_width,
                   label=VARIANT_LABELS[v], color=COLORS[v], edgecolor="white", linewidth=0.5)

    # F1-macro panel
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels(algo_order, rotation=30, ha="right", fontsize=9)
    ax_f1.set_ylabel("F1-macro CV (média 5-fold)", fontsize=10)
    ax_f1.set_title("F1-macro — baseline vs variantes", fontsize=11)
    ax_f1.set_ylim(0.35, max(f1_pivot.values.max() + 0.02, 0.46))
    ax_f1.grid(axis="y", alpha=0.3)
    ax_f1.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Linha do melhor F1 absoluto = lightgbm_v2_smote
    best_f1 = f1_pivot.values.max()
    ax_f1.axhline(best_f1, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
    ax_f1.text(len(algo_order) - 0.5, best_f1 + 0.001,
               f"melhor: {best_f1:.4f}", fontsize=7, ha="right", va="bottom", style="italic")

    # Recall Alerta/Grave panel
    ax_rec.set_xticks(x)
    ax_rec.set_xticklabels(algo_order, rotation=30, ha="right", fontsize=9)
    ax_rec.set_ylabel("Recall — classe Alerta/Grave (0.6% da base)", fontsize=10)
    ax_rec.set_title("Recall na classe minoritária — efeito do SMOTE", fontsize=11)
    ax_rec.set_ylim(0, max(rec_pivot.values.max() + 0.02, 0.15))
    ax_rec.grid(axis="y", alpha=0.3)
    ax_rec.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Comparação cross-variante (40 runs, n=10 algoritmos × 4 variantes)\n"
        "F1-macro CV à esquerda; Recall na classe-alvo (Alerta/Grave) à direita",
        fontsize=12, y=1.01
    )
    plt.tight_layout()

    out_png = FIG_DIR / "cross_variant_comparison.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ CSV    → {out_csv.relative_to(PROJECT_ROOT)}")
    print(f"✓ Figura → {out_png.relative_to(PROJECT_ROOT)}")

    # Resumo no stdout
    print("\n=== F1-macro CV: top 3 por variante ===")
    for v in VARIANTS:
        top3 = full[full["variant"] == v].nlargest(3, "f1_macro_cv")[["algoritmo", "f1_macro_cv"]]
        print(f"  {VARIANT_LABELS[v]}:")
        for _, r in top3.iterrows():
            print(f"     {r['algoritmo']:<15} {r['f1_macro_cv']:.4f}")

    print("\n=== Recall Alerta/Grave: top 3 por variante ===")
    for v in VARIANTS:
        top3 = full[full["variant"] == v].nlargest(3, "recall_alerta_grave")[["algoritmo", "recall_alerta_grave"]]
        print(f"  {VARIANT_LABELS[v]}:")
        for _, r in top3.iterrows():
            print(f"     {r['algoritmo']:<15} {r['recall_alerta_grave']:.4f}")


if __name__ == "__main__":
    main()
