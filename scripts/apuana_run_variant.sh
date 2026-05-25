#!/bin/bash
#SBATCH --job-name=triagem_run
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-9%2
#SBATCH --output=logs/triagem_%x_%A_%a.out
#SBATCH --error=logs/triagem_%x_%A_%a.err
#
# Submete os 10 algoritmos de uma variante como array job (paralelismo %2 — limite
# QoS `simple` no Apuana CIn-UFPE). Substitui scripts/apuana_run_v{2,3,4}.sh.
#
# Uso:
#     sbatch --job-name=triagem_v3 --export=ALL,VARIANT=v3 scripts/apuana_run_variant.sh
#     sbatch --job-name=triagem_v4 --export=ALL,VARIANT=v4 scripts/apuana_run_variant.sh
#
# Outputs em logs/triagem_<job-name>_<job-id>_<array-id>.out (via %x = job-name).
# Time 06:00:00 acomoda v1/v2 pesados em rerun; v3/v4 terminam em ~2-3h.
set -euo pipefail
: "${VARIANT:?VARIANT env var required (e.g. v2, v3, v4) — passar via 'sbatch --export=ALL,VARIANT=v3 ...'}"

ALGORITHMS=(
    "decision_tree"
    "run_lightgbm"
    "random_forest"
    "xgboost"
    "lvq"
    "knn"
    "stacking"
    "rna_committee"
    "mlp"
    "svm"
)
ALGO="${ALGORITHMS[$SLURM_ARRAY_TASK_ID]}_${VARIANT}"

REPO_DIR="$HOME/triagem-dengue"
VENV_DIR="$HOME/triagem-dengue-venv"

cd "$REPO_DIR"
mkdir -p logs reports/figures
find . -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "[$(date)] Iniciando $ALGO (task $SLURM_ARRAY_TASK_ID, job $SLURM_JOB_ID)"
source "$VENV_DIR/bin/activate"

python3 "experiments/${ALGO}.py"

echo "[$(date)] ✓ $ALGO finalizado"
