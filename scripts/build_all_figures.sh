#!/usr/bin/env bash
# Gera todas as 22 figuras + análises do projeto a partir do MLflow local.
#
# Pré-requisito: 40 runs sincronizados em mlflow_dengue.db (vide
# scripts/run_all_experiments.sh OU scripts/sync_mlflow_from_apuana.py).
#
# Tempo total: ~3-5 min.

set -euo pipefail
cd "$(dirname "$0")/.."

VARIANTS=("v1_baseline" "v2_smote" "v3_target_enc" "v4_selectk")

echo "=== [1/5] Validation curves grids (4 figuras) ==="
for v in "${VARIANTS[@]}"; do
    python scripts/build_validation_curves_grid.py --variant "$v"
done

echo ""
echo "=== [2/5] Confusion matrices grids (4 figuras) ==="
for v in "${VARIANTS[@]}"; do
    python scripts/build_confusion_matrices.py --variant "$v"
done

echo ""
echo "=== [3/5] Variant summary tables + barplots (4 pares) ==="
for v in "${VARIANTS[@]}"; do
    python scripts/build_variant_summary.py --variant "$v"
done

echo ""
echo "=== [4/5] Pareto scatters (4 figuras) ==="
for v in "${VARIANTS[@]}"; do
    python scripts/build_pareto_scatter.py --variant "$v"
done

echo ""
echo "=== [5/6] Cross-variant comparison + Wilcoxon + Final test ==="
python scripts/build_cross_variant_comparison.py
python scripts/wilcoxon_paired.py
python scripts/final_evaluation.py

echo ""
echo "=== [6/6] Distribuição das classes (slide 4 da apresentação) ==="
python scripts/build_class_distribution.py

echo ""
echo "✓ Todas as figuras geradas em reports/figures/."
echo "  Resumo Wilcoxon: reports/wilcoxon_paired_summary.md"
echo "  Avaliação final: reports/figures/final_test_*"
