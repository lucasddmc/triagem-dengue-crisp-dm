#!/bin/bash
# ============================================================================
# Script de submissão dos últimos passos no Apuana — v3 + v4
# ============================================================================
# Pré-requisitos: VPN do CIn ligada; SSH key em ~/.ssh/id_ed25519_apuana.
#
# Uso recomendado (NÃO rodar tudo de uma vez — copy-paste por step):
#   bash scripts/submit_remaining.sh   # só lista os comandos
#
# OU executar steps manuais via SSH (mais seguro pra acompanhar):
#   ssh -i ~/.ssh/id_ed25519_apuana ldmc@slurm-client1.cin.ufpe.br
# ============================================================================
set -e

APUANA_SSH="ssh -i ~/.ssh/id_ed25519_apuana ldmc@slurm-client1.cin.ufpe.br"
PROJECT_LOCAL="/Users/ldmc/Desktop/faculdade/2026.1/mineração/triagem-dengue"

cat <<'EOF'
============================================================================
PASSOS RESTANTES PRA APRESENTAÇÃO 27/05 — copy/paste em ordem
============================================================================

# --------------------------------------------------------------------------
# STEP 1 — Submeter v3 (TargetEncoder) — 10 algoritmos × %2 paralelo
# Wall-clock estimado: ~2h
# --------------------------------------------------------------------------
ssh -i ~/.ssh/id_ed25519_apuana ldmc@slurm-client1.cin.ufpe.br

cd ~/triagem-dengue
# Limpa cache pra evitar bytecode stale (lição aprendida do v1 oversubscription)
find . -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Submete v3
sbatch --job-name=triagem_v3 --export=ALL,VARIANT=v3 scripts/apuana_run_variant.sh

# Acompanhar
squeue -u ldmc
# Esperar ~2h até squeue ficar vazio

# --------------------------------------------------------------------------
# STEP 2 — Quando v3 acabar: submeter v4 (SelectKBest)
# Wall-clock estimado: ~2h
# --------------------------------------------------------------------------
sbatch --job-name=triagem_v4 --export=ALL,VARIANT=v4 scripts/apuana_run_variant.sh
squeue -u ldmc

# --------------------------------------------------------------------------
# STEP 3 — Quando v4 acabar: sync local
# --------------------------------------------------------------------------
# Voltar pra máquina local (sair do ssh)
exit

# Local:
cd /Users/ldmc/Desktop/faculdade/2026.1/mineração/triagem-dengue

# rsync mlruns do Apuana (~5min, exclui .pkl pesados)
rsync -avz --exclude='*.pkl' --exclude='*.cloudpickle' \
  --exclude='conda.yaml' --exclude='python_env.yaml' \
  --exclude='requirements.txt' --exclude='*.tar.gz' \
  -e "ssh -i ~/.ssh/id_ed25519_apuana" \
  ldmc@slurm-client1.cin.ufpe.br:~/triagem-dengue/mlruns/ \
  ./mlflow_apuana/mlruns/

# Sync v3 e v4 pro SQLite local
python3 scripts/sync_mlflow_from_apuana.py \
  --source-uri "file://$PWD/mlflow_apuana/mlruns" \
  --target-uri "sqlite:///$PWD/mlflow_dengue.db" \
  --experiment-name triagem-dengue

# --------------------------------------------------------------------------
# STEP 4 — Gerar análises pras 4 variantes
# --------------------------------------------------------------------------
for v in v1_baseline v2_smote v3_target_enc v4_selectk; do
  echo "===== $v ====="
  python3 scripts/build_variant_summary.py --variant $v
  python3 scripts/build_validation_curves_grid.py --variant $v
  python3 scripts/build_confusion_matrices.py --variant $v
  python3 scripts/build_pareto_scatter.py --variant $v
done

# Resultado: ~16 PNGs + 16 CSVs em reports/figures/

# --------------------------------------------------------------------------
# STEP 5 — Wilcoxon pareado v1 vs cada variante (script ainda não criado!)
# Esboço de implementação:
# --------------------------------------------------------------------------
# Criar scripts/wilcoxon_paired.py com:
#   from scipy.stats import wilcoxon
#   from statsmodels.stats.multitest import multipletests
#   import mlflow
#
#   client = mlflow.tracking.MlflowClient(tracking_uri='sqlite:///mlflow_dengue.db')
#   exp = client.get_experiment_by_name('triagem-dengue')
#
#   # Pra cada variante VAR in [v2_smote, v3_target_enc, v4_selectk]:
#   #   Pra cada algoritmo:
#   #     v1_folds = [run_v1.data.metrics[f'f1_macro_fold_{i}'] for i in range(5)]
#   #     var_folds = [run_var.data.metrics[f'f1_macro_fold_{i}'] for i in range(5)]
#   #     stat, p = wilcoxon(v1_folds, var_folds)
#   #
#   #   Holm-Bonferroni nos 10 testes da variante
#   #   reject, p_adj = multipletests(p_values, method='holm')
#
#   # Exporta tabela CSV: variante × algoritmo × p_value × p_adj × significativo

# --------------------------------------------------------------------------
# STEP 6 — Slides Beamer
# --------------------------------------------------------------------------
# Template: wiki/faculdade/mineracao-de-dados/src-slides-template.md
# Conteúdo já pronto pra reuso (popular slides direto):
#   - Briefing → wiki/.../triagem-dengue.md seção "Briefing"
#   - EDA → 14 achados em wiki/.../triagem-dengue.md
#   - Resultados → reports/figures/ (4 variantes × 4 figuras = 16 PNGs)
#   - Limitações → wiki/.../triagem-dengue.md seção "Decisões de execução"
#
# Entregar PDF noite 26/05 ao prof.

EOF
