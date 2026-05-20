# Triagem de Dengue — Recife

**Disciplina**: IF1014 — Mineração de Dados Aplicada com CRISP-DM (2026.1)  
**Professor**: Leandro Maciel Almeida  
**Grupo**: ldmc, jvlm2

Classificação de casos de dengue do Recife (2016–2025) em três categorias de gravidade, seguindo a metodologia CRISP-DM.

---

## Objetivo

Prever a classificação clínica de uma notificação de dengue:

| Classe | Descrição | Códigos SINAN |
|--------|-----------|---------------|
| `0` — Descartado | Caso não confirmado | 5 |
| `1` — Comum | Dengue confirmada sem complicações | 1, 10 |
| `2` — Alerta/Grave | Dengue com sinais de alarme ou grave | 2, 3, 4, 11, 12 |

---

## Dataset

- **Fonte**: [Portal de Dados Abertos — Prefeitura do Recife](https://dados.recife.pe.gov.br/es/dataset/casos-de-dengue-zika-e-chikungunya/resource/705f7da8-6d85-4fd6-9005-d5f168293d0b)
- **HuggingFace**: [`lucasddmc/recife-dengue-harmonizado`](https://huggingface.co/datasets/lucasddmc/recife-dengue-harmonizado)
- **Instâncias**: 68.140 casos (9 tabelas concatenadas, 2016–2025)
- **Features**: 198 colunas pré-harmonização; ~30 após limpeza

```bash
# Baixar o dataset manualmente
python -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download('lucasddmc/recife-dengue-harmonizado', repo_type='dataset', filename='data/dataset_harmonizado.parquet')
shutil.copy(path, 'data/dataset_harmonizado.parquet')
"
```

---

## Estrutura

```
triagem-dengue/
├── data/
│   └── dataset_harmonizado.parquet     # baixado do HuggingFace (gitignored)
├── notebooks/
│   └── recife_dengue_ldmc_jvlm2.ipynb  # notebook principal (CRISP-DM completo)
├── src/                                # módulos Python reutilizáveis (data_loader.py, hp_search.py, evaluation.py)
├── models/                             # modelos exportados (gitignored)
├── reports/
│   ├── figures/                        # figuras curadas pro relatório
│   └── relatorio/                      # fonte LaTeX (relatorio.tex)
├── slides/                             # fonte Beamer (slides.tex)
├── mlruns/                             # artifacts MLflow (gitignored, regerável)
├── mlflow_dengue.db                    # banco MLflow tracking (gitignored)
├── .gitignore
├── LICENSE
└── README.md
```

> **Nota**: `mlruns/`, `mlflow_dengue.db` e `data/*.parquet` são regeráveis (a partir do notebook + HuggingFace) e não entram no Git. Pra visualizar os experimentos rastreados, rode `mlflow ui --backend-store-uri sqlite:///mlflow_dengue.db` da raiz do projeto.

---

## Pipeline (CRISP-DM)

1. **Entendimento dos dados** — EDA, distribuição de classes, correlações
2. **Preparação** — harmonização de colunas, cálculo de idade com fallback, drop de colunas >90% nulas, binarização, encoding cíclico de semana
3. **Modelagem** — 8 experimentos comparativos rastreados com MLflow:
   - Baseline LightGBM
   - SMOTE
   - Class Weights
   - Hybrid Sampling (Under + Over)
   - Random Forest + Hybrid Sampling
   - Abordagem Hierárquica (2 modelos em cascata)
   - Isolation Forest
   - XGBoost + Sample Weights
4. **Avaliação** — F1-score e Recall na classe `Alerta/Grave` como métricas principais (dataset desbalanceado)

---

## Dependências

```bash
pip install mlflow lightgbm xgboost scikit-learn imbalanced-learn \
            pandas numpy matplotlib seaborn huggingface_hub pyarrow
```
