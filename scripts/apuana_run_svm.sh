#!/bin/bash
#SBATCH --job-name=triagem_svm
#SBATCH --partition=long-simple
#SBATCH --qos=simple
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/triagem_svm_%j.out
#SBATCH --error=logs/triagem_svm_%j.err
#
# SVM em job separado — pode demorar horas em 52k linhas.
# Recursos máximos da QoS simple (16 cpus / 64GB / 24h).
#
# Uso: sbatch scripts/apuana_run_svm.sh
set -euo pipefail

REPO_DIR="$HOME/triagem-dengue"
VENV_DIR="$HOME/triagem-dengue-venv"

cd "$REPO_DIR"
mkdir -p logs reports/figures

echo "[$(date)] Iniciando SVM (job $SLURM_JOB_ID)"
source "$VENV_DIR/bin/activate"

python3 experiments/svm.py

echo "[$(date)] ✓ SVM finalizado"
