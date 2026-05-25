#!/bin/bash
#SBATCH --job-name=triagem_v1
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-8%4
#SBATCH --output=logs/triagem_%A_%a.out
#SBATCH --error=logs/triagem_%A_%a.err
#
# Submete os 9 algoritmos v1_baseline como array job.
# Index do array → algoritmo:
#   0: decision_tree     (rápido, ~5min)
#   1: lightgbm          (rápido, ~10min)
#   2: random_forest     (médio, ~15min)
#   3: xgboost_baseline  (médio, ~15min)
#   4: lvq               (médio, ~20min — se sklvq instalou)
#   5: knn               (pesado, ~30-60min)
#   6: stacking          (pesado, ~30min)
#   7: rna_committee     (pesado, ~60min)
#   8: mlp               (pesado, ~60min)
#   (svm separado — script svm_only.sh com time maior, pode demorar horas)
#
# Uso: sbatch scripts/apuana_run.sh
set -euo pipefail

ALGORITHMS=(
    "decision_tree"
    "run_lightgbm"
    "random_forest"
    "xgboost_baseline"
    "lvq"
    "knn"
    "stacking"
    "rna_committee"
    "mlp"
)
ALGO="${ALGORITHMS[$SLURM_ARRAY_TASK_ID]}"

REPO_DIR="$HOME/triagem-dengue"
VENV_DIR="$HOME/triagem-dengue-venv"

cd "$REPO_DIR"
mkdir -p logs reports/figures

echo "[$(date)] Iniciando $ALGO (task $SLURM_ARRAY_TASK_ID, job $SLURM_JOB_ID)"
source "$VENV_DIR/bin/activate"

python3 "experiments/${ALGO}.py"

echo "[$(date)] ✓ $ALGO finalizado"
