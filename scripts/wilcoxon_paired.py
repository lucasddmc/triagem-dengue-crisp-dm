"""
scripts/wilcoxon_paired.py

Teste Wilcoxon signed-rank pareado entre v1_baseline (controle) e cada variante
{v2_smote, v3_target_enc, v4_selectk} (tratamentos), por algoritmo. Métrica:
F1-macro por fold (5 folds pareados — mesmo StratifiedKFold, random_state=42).

Para cada par (algoritmo, variante):
    x = [f1_macro_fold_i (v1)]_{i=0..4}
    y = [f1_macro_fold_i (variante)]_{i=0..4}
    H_0: x e y vêm da mesma distribuição.
    Teste: wilcoxon two-sided, method='exact', zero_method='wilcox'.

Correção: Holm-Bonferroni *dentro de cada variante* (10 testes/família —
controle de FWER no nível α=0.05 por família, alinhado ao framework do Demšar
2006 em que cada variante vs baseline é uma família independente de hipóteses).

Effect sizes reportados (essenciais dado N=5):
    - rank-biserial correlation:  r = (W+ − W−) / (W+ + W−),  range [−1, +1]
    - Δ médio = mean(variante) − mean(v1)

REFERÊNCIAS
-----------
- Demšar, J. (2006). "Statistical Comparisons of Classifiers over Multiple
  Data Sets". JMLR 7:1–30. https://jmlr.csail.mit.edu/papers/v7/demsar06a
  → recomenda Wilcoxon signed-rank como teste não-paramétrico padrão pra
    comparação pareada de dois classificadores.
- Holm, S. (1979). "A simple sequentially rejective multiple test procedure".
  Scand. J. Statistics 6:65–70.
- Wright, S. P. (1992). "Adjusted P-values for simultaneous inference".
  Biometrics 48:1005–13. → fórmula step-down do p_adj.
- Nadeau, C. & Bengio, Y. (2003). "Inference for the Generalization Error".
  Machine Learning 52:239–281. → citado no caveat L2 abaixo.
- scipy.stats.wilcoxon — `method='exact'` recomendado pra N≤25.

LIMITAÇÕES METODOLÓGICAS (registrar no relatório)
-------------------------------------------------
L1. Demšar (2006) propõe Wilcoxon pra COMPARAÇÃO DE CLASSIFICADORES SOBRE
    MÚLTIPLOS DATASETS — cada dataset funciona como um "sujeito" pareado. Aqui
    aplicamos sobre os 5 folds de um único dataset, o que viola levemente a
    independência entre observações (folds compartilham amostras). O teste
    teoricamente mais correto pra essa configuração seria o *corrected
    resampled t-test* (Nadeau & Bengio 2003), que ajusta a variância pela
    sobreposição entre folds. Mantemos Wilcoxon porque: (a) a documentação do
    projeto IF1014 pede explicitamente "Wilcoxon pareado ou paired t-test", e
    (b) ainda fornece evidência útil mesmo com a violação branda.

L2. Com N=5 folds, o p-value exato two-sided mínimo possível é
    2 × (1/2^5) = 0.0625. Logo mesmo com TODAS as 5 dobras favorecendo o mesmo
    sentido, p ≥ 0.06 — significância "estrita" α=0.05 é fora do alcance
    two-sided. Por isso reportamos rank-biserial e Δ médio: **p > 0.05 com
    N=5 NÃO equivale a "sem efeito"**, pode ser apenas falta de poder
    estatístico. Effect size + Δ orientam a interpretação prática.

USO
---
    python scripts/wilcoxon_paired.py
    python scripts/wilcoxon_paired.py --alpha 0.10
    python scripts/wilcoxon_paired.py --output reports/wilcoxon_paired.csv

OUTPUTS
-------
    reports/wilcoxon_paired.csv          — long-form (1 linha por algo×variante)
    reports/wilcoxon_paired_summary.md   — tabela detalhada + pivot pra slides
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "mlflow_dengue.db"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENT_NAME = "triagem-dengue"

BASELINE = "v1_baseline"
VARIANTS = ("v2_smote", "v3_target_enc", "v4_selectk")
N_FOLDS = 5


# ----------------------------------------------------------------------------
# Carga de dados do MLflow
# ----------------------------------------------------------------------------
def load_fold_scores(
    client: MlflowClient, exp_id: str, variant: str
) -> dict[str, np.ndarray]:
    """Retorna {algoritmo: array de 5 fold scores} pra uma variante.

    Filtra por `tags.variante = <variant> AND tags.source = 'apuana'`. Runs com
    fold scores incompletos são silenciosamente ignorados.
    """
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string=(
            f"tags.variante = '{variant}' AND tags.source = 'apuana'"
        ),
        max_results=200,
    )
    out: dict[str, np.ndarray] = {}
    for r in runs:
        algo = r.data.tags.get("algoritmo", r.info.run_name)
        folds = [r.data.metrics.get(f"f1_macro_fold_{i}") for i in range(N_FOLDS)]
        if any(f is None for f in folds):
            continue
        out[algo] = np.asarray(folds, dtype=float)
    return out


# ----------------------------------------------------------------------------
# Estatística
# ----------------------------------------------------------------------------
def rank_biserial(deltas: np.ndarray) -> float:
    """Rank-biserial correlation pra Wilcoxon signed-rank.

    Definição: r = (W+ − W−) / (W+ + W−), onde W± é a soma dos ranks dos
    |diffs| positivos/negativos. Range [−1, +1]. Convenção: positivo = y > x
    (variante > v1). Considera apenas diffs != 0 (zero_method='wilcox').

    Interpretação aproximada (Cohen-style):
        |r| ≈ 0.1 → pequeno
        |r| ≈ 0.3 → médio
        |r| ≈ 0.5 → grande
    """
    d = deltas[deltas != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    sum_all = ranks.sum()
    sum_pos = ranks[d > 0].sum()
    sum_neg = ranks[d < 0].sum()
    return float((sum_pos - sum_neg) / sum_all)


def paired_test(v1: np.ndarray, var: np.ndarray) -> dict:
    """Wilcoxon signed-rank two-sided exato + estatísticas auxiliares.

    Convenção do sinal: `delta = var − v1`, positivo => variante > baseline.
    """
    delta = var - v1
    if np.all(delta == 0):
        return dict(
            stat=0.0,
            p_value=1.0,
            rank_biserial=0.0,
            mean_delta=0.0,
            n_pairs=len(v1),
            n_nonzero=0,
        )
    # scipy>=1.13 usa kwarg `method`; mantemos try/except defensivo
    try:
        result = stats.wilcoxon(
            var, v1, alternative="two-sided", method="exact", zero_method="wilcox"
        )
    except TypeError:  # pragma: no cover (scipy antigo usa `mode`)
        result = stats.wilcoxon(
            var, v1, alternative="two-sided", mode="exact", zero_method="wilcox"
        )
    return dict(
        stat=float(result.statistic),
        p_value=float(result.pvalue),
        rank_biserial=rank_biserial(delta),
        mean_delta=float(delta.mean()),
        n_pairs=len(v1),
        n_nonzero=int(np.count_nonzero(delta)),
    )


def holm_bonferroni(p_values: Iterable[float]) -> np.ndarray:
    """Step-down Holm-Bonferroni adjusted p-values (Wright 1992).

    Algoritmo:
        1. Ordena p ascendente:  p_(1) ≤ p_(2) ≤ … ≤ p_(m).
        2. Para k=1..m, define p_adj_(k) = max(p_adj_(k-1), (m−k+1) · p_(k)),
           capped em 1.
        3. Restaura a ordem original.
    Decisão: rejeita H_k se p_adj_k < α (controla FWER ≤ α).

    Equivalente a `statsmodels.stats.multitest.multipletests(method='holm')`,
    implementado à mão pra evitar a dependência.
    """
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    p_sorted = p[order]
    p_adj_sorted = np.empty(m)
    running_max = 0.0
    for k in range(m):
        candidate = (m - k) * p_sorted[k]
        running_max = max(running_max, candidate)
        p_adj_sorted[k] = min(running_max, 1.0)
    p_adj = np.empty(m)
    p_adj[order] = p_adj_sorted
    return p_adj


# ----------------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------------
def build_results(
    fold_scores_v1: dict[str, np.ndarray],
    fold_scores_variants: dict[str, dict[str, np.ndarray]],
    alpha: float,
) -> pd.DataFrame:
    """Roda Wilcoxon pra todos os pares (algo, variante) e aplica Holm-B."""
    rows: list[dict] = []
    for variant in VARIANTS:  # ordem estável
        if variant not in fold_scores_variants:
            continue
        scores = fold_scores_variants[variant]
        variant_rows: list[dict] = []
        for algo in sorted(scores.keys()):
            if algo not in fold_scores_v1:
                print(f"  ⚠ '{algo}' ausente em v1_baseline — pulando ({variant}).")
                continue
            v1 = fold_scores_v1[algo]
            var = scores[algo]
            test = paired_test(v1, var)
            variant_rows.append(
                dict(
                    variante=variant,
                    algoritmo=algo,
                    n_folds=test["n_pairs"],
                    v1_mean=float(v1.mean()),
                    var_mean=float(var.mean()),
                    delta=test["mean_delta"],
                    wilcoxon_stat=test["stat"],
                    p_value=test["p_value"],
                    rank_biserial=test["rank_biserial"],
                    n_nonzero=test["n_nonzero"],
                )
            )
        if not variant_rows:
            continue
        p_adj = holm_bonferroni([r["p_value"] for r in variant_rows])
        for r, pa in zip(variant_rows, p_adj):
            r["p_adj_holm"] = float(pa)
            r["significant_005"] = bool(pa < alpha)
        rows.extend(variant_rows)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Renderização
# ----------------------------------------------------------------------------
def _df_to_markdown(df: pd.DataFrame) -> str:
    """Markdown table renderer sem depender de tabulate."""
    cols = df.columns.tolist()
    sep = "|"
    head = sep + sep.join(f" {c} " for c in cols) + sep
    rule = sep + sep.join(" --- " for _ in cols) + sep
    body = []
    for _, row in df.iterrows():
        cells = [str(row[c]) if row[c] is not None else "" for c in cols]
        body.append(sep + sep.join(f" {c} " for c in cells) + sep)
    return "\n".join([head, rule, *body])


def render_markdown(df: pd.DataFrame, alpha: float) -> str:
    if df.empty:
        return "Nenhum resultado.\n"
    out: list[str] = []
    out.append("# Wilcoxon pareado — v1_baseline × variantes\n")
    out.append("- **Métrica**: F1-macro por fold (N=5, StratifiedKFold random_state=42).")
    out.append("- **Teste**: scipy.stats.wilcoxon two-sided, method='exact', zero_method='wilcox'.")
    out.append("- **Correção**: Holm-Bonferroni dentro de cada variante (10 testes/família).")
    out.append(f"- **α** = {alpha}.")
    out.append("- **Δ** = média(variante) − média(v1); positivo = variante > v1.")
    out.append(
        "- ⚠ N=5 limita o p-value two-sided mínimo a ~0.0625. "
        "Sempre inspecione rank_biserial e Δ — `p > 0.05` ≠ \"sem efeito\".\n"
    )

    # Tabela detalhada
    out.append("## Tabela detalhada\n")
    detail = df.copy()
    detail["v1_mean"] = detail["v1_mean"].map(lambda x: f"{x:.4f}")
    detail["var_mean"] = detail["var_mean"].map(lambda x: f"{x:.4f}")
    detail["delta"] = detail["delta"].map(lambda x: f"{x:+.4f}")
    detail["wilcoxon_stat"] = detail["wilcoxon_stat"].map(lambda x: f"{x:.1f}")
    detail["p_value"] = detail["p_value"].map(lambda x: f"{x:.4f}")
    detail["p_adj_holm"] = detail["p_adj_holm"].map(lambda x: f"{x:.4f}")
    detail["rank_biserial"] = detail["rank_biserial"].map(lambda x: f"{x:+.3f}")
    detail["significant_005"] = detail["significant_005"].map(lambda x: "✓" if x else "")
    detail = detail[
        [
            "variante", "algoritmo", "v1_mean", "var_mean", "delta",
            "wilcoxon_stat", "p_value", "p_adj_holm", "rank_biserial",
            "significant_005",
        ]
    ]
    out.append(_df_to_markdown(detail))

    # Tabela pivotada Δ (p_adj)
    out.append("\n\n## Pivot — Δ F1 (p_adj entre parênteses)\n")
    df_pivot = df.copy()
    df_pivot["cell"] = df_pivot.apply(
        lambda r: (
            f"{r['delta']:+.4f} ({r['p_adj_holm']:.3f})"
            + ("*" if r["significant_005"] else "")
        ),
        axis=1,
    )
    pivot = df_pivot.pivot(
        index="algoritmo", columns="variante", values="cell"
    ).fillna("—")
    pivot = pivot.reset_index()
    out.append(_df_to_markdown(pivot))
    out.append("\n`*` = significativo (p_adj < α).\n")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Wilcoxon signed-rank pareado v1_baseline × variantes, com "
            "Holm-Bonferroni dentro de cada variante."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "wilcoxon_paired.csv",
        help="Caminho do CSV long-form.",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=REPORTS_DIR / "wilcoxon_paired_summary.md",
        help="Caminho do resumo Markdown.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(
            f"Experimento '{EXPERIMENT_NAME}' não existe em {DB_PATH}."
        )

    v1_scores = load_fold_scores(client, exp.experiment_id, BASELINE)
    if not v1_scores:
        raise SystemExit(
            f"Nenhum run de '{BASELINE}' com fold scores. "
            f"Roda `scripts/recover_fold_scores.py --variant v1_baseline` antes."
        )
    print(f"v1_baseline: {len(v1_scores)} algoritmos com fold scores.")

    variant_scores: dict[str, dict[str, np.ndarray]] = {}
    for v in VARIANTS:
        s = load_fold_scores(client, exp.experiment_id, v)
        print(f"  {v:<18} → {len(s)} algoritmos com fold scores.")
        if s:
            variant_scores[v] = s

    if not variant_scores:
        raise SystemExit(
            "Nenhuma variante tem fold scores ainda. "
            "Submete v3/v4 ou roda recover_fold_scores.py."
        )

    df = build_results(v1_scores, variant_scores, alpha=args.alpha)
    if df.empty:
        raise SystemExit("Nenhum par v1 × variante alinhado.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, float_format="%.6f")
    print(f"\nCSV escrito: {args.output}")

    md = render_markdown(df, args.alpha)
    args.md_output.write_text(md, encoding="utf-8")
    print(f"Markdown escrito: {args.md_output}")

    sig = df[df["significant_005"]]
    print(
        f"\nResumo: {len(df)} testes, "
        f"{len(sig)} significativos (p_adj < {args.alpha})."
    )
    if not sig.empty:
        for _, row in sig.iterrows():
            print(
                f"  ✓ {row['algoritmo']:<15} vs {row['variante']:<15} "
                f"Δ={row['delta']:+.4f}  p_adj={row['p_adj_holm']:.4f}  "
                f"r_b={row['rank_biserial']:+.3f}"
            )
    else:
        print("  (nenhum atinge α — consulte effect sizes em rank_biserial e Δ).")


if __name__ == "__main__":
    main()
