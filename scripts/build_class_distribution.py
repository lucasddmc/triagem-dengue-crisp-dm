"""Gera reports/figures/class_distribution.png pro slide 4 da apresentação.

Não toca o sentinel: lê y_train via load_train() (sem restrição) e y_test do
artifact já gerado em reports/figures/final_test_y_pred.csv (resultado da única
liberação autorizada do sentinel em scripts/final_evaluation.py).

Sanity check: razão treino:teste por classe ≈ 4.0 confirma split estratificado 80/20.

Uso:
    python scripts/build_class_distribution.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_train

# Treino: chamada livre (não toca o sentinel)
_, y_train = load_train()
counts_train = pd.Series(y_train).value_counts().sort_index()

# Teste: lê do artifact já publicado em final_test_y_pred.csv (NÃO chama
# load_test de novo — mantém o sentinel "uma única vez")
df_pred = pd.read_csv(PROJECT_ROOT / "reports" / "figures" / "final_test_y_pred.csv")
counts_test = df_pred["y_true"].value_counts().sort_index()

names = ["Descartado", "Comum", "Alerta/Grave"]
colors = ("#4C72B0", "#DD8452")  # treino azul, teste laranja (seaborn deep)

fig, ax = plt.subplots(figsize=(9, 4.8))
x = list(range(3))
bw = 0.38

bars_train = ax.bar([i - bw/2 for i in x], counts_train.values, bw,
                     label=f"Treino (n={counts_train.sum():,})",
                     color=colors[0], edgecolor="white", linewidth=0.6)
bars_test = ax.bar([i + bw/2 for i in x], counts_test.values, bw,
                    label=f"Teste (n={counts_test.sum():,})",
                    color=colors[1], edgecolor="white", linewidth=0.6)

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("Número de notificações (escala log)", fontsize=10)
ax.set_title("Distribuição das classes — split estratificado treino/teste (80/20, random_state=42)",
             fontsize=11)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(axis="y", alpha=0.3, which="both", linestyle="--", linewidth=0.4)

# Anotações: valor absoluto + percentual em cada barra
for i, (n_tr, n_te) in enumerate(zip(counts_train.values, counts_test.values)):
    pct_tr = 100 * n_tr / counts_train.sum()
    pct_te = 100 * n_te / counts_test.sum()
    ax.text(i - bw/2, n_tr, f"{n_tr:,}\n({pct_tr:.2f}%)",
            ha="center", va="bottom", fontsize=8.5)
    ax.text(i + bw/2, n_te, f"{n_te:,}\n({pct_te:.2f}%)",
            ha="center", va="bottom", fontsize=8.5)

# Headroom pro topo (log scale exige margem maior)
ymax = max(counts_train.max(), counts_test.max())
ax.set_ylim(top=ymax * 4.5)

fig.text(0.5, -0.02,
         "Classe Alerta/Grave é 0.6% da base — desbalanceamento extremo motivou v2 (SMOTE) e PR-AUC como métrica clínica preferida.",
         ha="center", fontsize=8.5, style="italic", color="#444")

plt.tight_layout()
out = PROJECT_ROOT / "reports" / "figures" / "class_distribution.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
plt.close(fig)

print(f"✓ Salvo: {out}")
print(f"  Treino:  {counts_train.to_dict()} (total {counts_train.sum():,})")
print(f"  Teste:   {counts_test.to_dict()} (total {counts_test.sum():,})")
print(f"  Razão treino:teste por classe:")
for cls in [0, 1, 2]:
    ratio = counts_train[cls] / counts_test[cls]
    print(f"    {cls} ({names[cls]:<14}): {ratio:.2f}× (esperado ~4.0 com 80/20)")
