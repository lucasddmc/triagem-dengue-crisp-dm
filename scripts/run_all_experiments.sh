#!/usr/bin/env bash
# Reproduz todos os 40 runs experimentais (10 algoritmos × 4 variantes) localmente.
#
# IDEMPOTÊNCIA: cada experimento checa MLflow antes de submeter; se já tem run
# com mesmo nome no experimento `triagem-dengue`, ele cria run novo (não pula).
# Portanto: 2 execuções = 80 runs. Pra evitar duplicatas, apague mlflow_dengue.db
# antes ou use SLURM array no Apuana (vide scripts/apuana_run_variant.sh).
#
# TEMPO ESTIMADO LOCAL (Mac M1/M2, n_jobs=-1):
#   - v1 baseline:    ~2-3h
#   - v2 SMOTE:       ~2-3h
#   - v3 TargetEncoder: ~2-3h
#   - v4 SelectKBest: ~1-2h (mais rápido por dimensionalidade reduzida)
#   - Total wall:     ~8-11h
#
# Pra rodar mais rápido use SLURM array no Apuana (~2h paralelo) — vide
# scripts/apuana_run_variant.sh.
#
# Uso:
#     bash scripts/run_all_experiments.sh                # roda tudo em sequência
#     bash scripts/run_all_experiments.sh --variant v2   # roda só uma variante
#     bash scripts/run_all_experiments.sh --algo svm     # roda só um algoritmo nas 4 variantes

set -euo pipefail
cd "$(dirname "$0")/.."

# Algoritmos com naming do projeto (run_lightgbm e xgboost_baseline têm prefixo
# por colisão com nome de biblioteca).
declare -A V1_FILES=(
    [decision_tree]="experiments/decision_tree.py"
    [knn]="experiments/knn.py"
    [lightgbm]="experiments/run_lightgbm.py"
    [lvq]="experiments/lvq.py"
    [mlp]="experiments/mlp.py"
    [random_forest]="experiments/random_forest.py"
    [rna_committee]="experiments/rna_committee.py"
    [stacking]="experiments/stacking.py"
    [svm]="experiments/svm.py"
    [xgboost]="experiments/xgboost_baseline.py"
)

declare -A V2_FILES=(
    [decision_tree]="experiments/decision_tree_v2.py"
    [knn]="experiments/knn_v2.py"
    [lightgbm]="experiments/run_lightgbm_v2.py"
    [lvq]="experiments/lvq_v2.py"
    [mlp]="experiments/mlp_v2.py"
    [random_forest]="experiments/random_forest_v2.py"
    [rna_committee]="experiments/rna_committee_v2.py"
    [stacking]="experiments/stacking_v2.py"
    [svm]="experiments/svm_v2.py"
    [xgboost]="experiments/xgboost_v2.py"
)

declare -A V3_FILES=(
    [decision_tree]="experiments/decision_tree_v3.py"
    [knn]="experiments/knn_v3.py"
    [lightgbm]="experiments/run_lightgbm_v3.py"
    [lvq]="experiments/lvq_v3.py"
    [mlp]="experiments/mlp_v3.py"
    [random_forest]="experiments/random_forest_v3.py"
    [rna_committee]="experiments/rna_committee_v3.py"
    [stacking]="experiments/stacking_v3.py"
    [svm]="experiments/svm_v3.py"
    [xgboost]="experiments/xgboost_v3.py"
)

declare -A V4_FILES=(
    [decision_tree]="experiments/decision_tree_v4.py"
    [knn]="experiments/knn_v4.py"
    [lightgbm]="experiments/run_lightgbm_v4.py"
    [lvq]="experiments/lvq_v4.py"
    [mlp]="experiments/mlp_v4.py"
    [random_forest]="experiments/random_forest_v4.py"
    [rna_committee]="experiments/rna_committee_v4.py"
    [stacking]="experiments/stacking_v4.py"
    [svm]="experiments/svm_v4.py"
    [xgboost]="experiments/xgboost_v4.py"
)

VARIANT_FILTER=""
ALGO_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) VARIANT_FILTER="$2"; shift 2 ;;
        --algo)    ALGO_FILTER="$2";    shift 2 ;;
        *) echo "Argumento desconhecido: $1"; exit 1 ;;
    esac
done

run_variant() {
    local variant_name="$1"
    local -n files_ref=$2

    if [[ -n "$VARIANT_FILTER" && "$VARIANT_FILTER" != "$variant_name" ]]; then
        return
    fi

    echo ""
    echo "==============================================="
    echo " Variante: $variant_name"
    echo "==============================================="

    for algo in "${!files_ref[@]}"; do
        if [[ -n "$ALGO_FILTER" && "$ALGO_FILTER" != "$algo" ]]; then
            continue
        fi
        local script="${files_ref[$algo]}"
        echo ""
        echo "  → $algo ($variant_name) — $script"
        echo "  [$(date)]"
        python "$script"
        echo "  ✓ $algo $variant_name finalizado"
    done
}

run_variant "v1" V1_FILES
run_variant "v2" V2_FILES
run_variant "v3" V3_FILES
run_variant "v4" V4_FILES

echo ""
echo "✓ Todos os experimentos completos. Inspecione com:"
echo "  mlflow ui --backend-store-uri sqlite:///mlflow_dengue.db"
