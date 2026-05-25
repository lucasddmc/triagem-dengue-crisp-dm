#!/bin/bash
# Setup do ambiente no Apuana — roda 1 vez no nó de login.
# Cria venv em ~/triagem-dengue-venv e instala deps.
#
# Uso: bash scripts/apuana_setup.sh
set -euo pipefail

REPO_DIR="$HOME/triagem-dengue"
VENV_DIR="$HOME/triagem-dengue-venv"

cd "$REPO_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "[setup] Criando venv em $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "[setup] Ativando venv e atualizando pip..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

echo "[setup] Instalando dependências..."
pip install -r requirements.txt

echo "[setup] Baixando dataset do HuggingFace..."
mkdir -p data
python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download('lucasddmc/recife-dengue-harmonizado', repo_type='dataset', filename='data/dataset_harmonizado.parquet')
shutil.copy(path, 'data/dataset_harmonizado.parquet')
print('Dataset baixado:', path)
"

echo "[setup] Validando imports..."
python3 -c "
from src.data_loader import load_train
X, y = load_train()
print(f'OK: X_train={X.shape}, y_train={y.shape}')
"

echo ""
echo "[setup] ✓ Setup completo. Pra submeter job:"
echo "  sbatch scripts/apuana_run.sh"
