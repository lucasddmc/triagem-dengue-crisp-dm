# Triagem de Dengue — Recife (CRISP-DM, IF1014)

**Disciplina**: IF1014 — Mineração de Dados Aplicada com CRISP-DM (2026.1)
**Centro**: CIn-UFPE
**Professor**: Leandro Maciel Almeida ([lma3@cin.ufpe.br](mailto:lma3@cin.ufpe.br))
**Grupo**: `ldmc`, `jvlm2`

Classificação multiclasse de notificações de dengue do Recife (2016-2025) em três categorias de gravidade, seguindo o workflow CRISP-DM exigido pelo edital. **40 experimentos** (10 algoritmos × 4 variantes do treino) com comparação pareada Wilcoxon + Holm-Bonferroni e avaliação final única no teste.

---

## Briefing do cliente (fictício)

> O cliente **Secretaria Executiva de Vigilância em Saúde da Secretaria de Saúde do Recife (SEVS/SESAU-Recife)** atua em **vigilância epidemiológica municipal** e precisa de um modelo capaz de **classificar notificações novas do SINAN de casos suspeitos de dengue** entre `{Descartado, Comum, Alerta/Grave}`. O sucesso é medido por **F1-macro** com restrição de **alto custo de falso-negativo na classe Alerta/Grave** (latência irrelevante — batch diário/semanal). Implicações éticas: viés demográfico em atributos com missing alto, privacidade clínica, impacto social desproporcional em populações vulneráveis.

| Classe | Descrição | Códigos SINAN | n (treino) | n (teste) |
|---|---|---|---|---|
| `0` Descartado | Caso não confirmado | 5 | 23.108 | 5.777 |
| `1` Comum | Dengue sem complicações | 1, 10 | 28.784 | 7.197 |
| `2` **Alerta/Grave** | Dengue com sinais de alarme/grave | 2, 3, 4, 11, 12 | 321 | 80 |

Classe `Alerta/Grave` é **0.6% da base** — desbalanceamento extremo. Métrica clínica secundária: **Recall em `Alerta/Grave`** (sensibilidade) + **PR-AUC macro** (preferida sobre ROC-AUC em desbalanceamento, vide Davis & Goadrich 2006).

---

## Dataset

- **Fonte primária**: [Portal de Dados Abertos — Prefeitura do Recife](https://dados.recife.pe.gov.br/es/dataset/casos-de-dengue-zika-e-chikungunya/resource/705f7da8-6d85-4fd6-9005-d5f168293d0b)
- **Harmonizado**: [`lucasddmc/recife-dengue-harmonizado` no HuggingFace Hub](https://huggingface.co/datasets/lucasddmc/recife-dengue-harmonizado)
- **Volume**: 65.267 linhas válidas (≥ 50k exigido pelo edital ✓), 3 classes ✓
- **Features**: 198 colunas brutas → 26 após cleanup (drop de leaks, colunas administrativas, endereços, datas) → 49 pós-OneHotEncoder

---

## Quickstart — reprodução end-to-end

**Requer**: Python 3.10+ (testado em 3.11). Recomendado virtualenv.

```bash
# 1. Setup do ambiente (1 comando)
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 2. Baixar dataset do HuggingFace (1 comando, idempotente)
python scripts/download_data.py

# 3. Rodar os 40 experimentos (1 comando, ~8-11h local OU ~2h paralelo Apuana)
bash scripts/run_all_experiments.sh
#    Subset: bash scripts/run_all_experiments.sh --variant v2
#    Subset: bash scripts/run_all_experiments.sh --algo lightgbm

# 4. Gerar 22 figuras + Wilcoxon + avaliação final no teste (1 comando, ~5min)
bash scripts/build_all_figures.sh

# 5. Inspecionar runs no MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow_dengue.db
```

**Alternativa cluster** (SLURM/Apuana CIn-UFPE): vide `scripts/apuana_run_variant.sh` (paramétrico por variante; ~2h em paralelo com array `%2`).

---

## Estrutura do projeto

```
triagem-dengue/
├── data/
│   └── dataset_harmonizado.parquet     # via HF (gitignored)
├── src/
│   ├── data_loader.py                  # load_raw, load_train, load_test(token), sentinela
│   ├── preprocessing.py                # cleanup determinístico (drop leaks/admin/data)
│   └── experiment_runner.py            # build_pipeline + busca HP + log MLflow
├── experiments/                        # 40 scripts: {algo}.py (v1), {algo}_v{2,3,4}.py
│   ├── decision_tree.py, decision_tree_v2.py, ..., decision_tree_v4.py
│   ├── ... (10 algoritmos × 4 variantes)
│   └── __init__.py                     # nota de design — duplicação intencional
├── scripts/
│   ├── download_data.py                # download HF Hub
│   ├── run_all_experiments.sh          # orquestra 40 runs
│   ├── build_all_figures.sh            # gera 22 figuras + análises
│   ├── build_validation_curves_grid.py # curvas treino vs val (1 grid por variante)
│   ├── build_confusion_matrices.py     # matrizes CV (1 grid por variante)
│   ├── build_variant_summary.py        # tabela + barplot top-10 por variante
│   ├── build_pareto_scatter.py         # F1 × Recall Alerta/Grave por variante
│   ├── build_cross_variant_comparison.py  # comparativo 4-variantes (slide 12)
│   ├── wilcoxon_paired.py              # 30 testes pareados + Holm-Bonferroni
│   ├── final_evaluation.py             # avaliação única no teste (sentinel único)
│   ├── recover_fold_scores.py          # recupera f1_fold_{0..4} retroativo (v1, v2)
│   ├── sync_mlflow_from_apuana.py      # rsync MLflow Apuana → SQLite local
│   └── apuana_run_variant.sh           # SLURM array paramétrico por variante
├── notebooks/
│   └── recife_dengue_ldmc_jvlm2.ipynb  # EDA + entrada didática (não end-to-end)
├── reports/
│   ├── figures/                        # 22 artifacts gerados por build_all_figures.sh
│   ├── wilcoxon_paired_summary.md      # 30 testes Wilcoxon (Holm-corrigido)
│   └── wilcoxon_paired.csv
├── mlflow_dengue.db                    # SQLite tracking (gitignored, regerável)
├── mlruns/                             # artifacts MLflow (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

> `mlruns/`, `mlflow_dengue.db`, `data/*.parquet`, `mlflow_apuana/`, `*_y_scores.csv` são **regeráveis** e ficam fora do Git.

---

## Workflow CRISP-DM (11 passos do edital)

| # | Passo | Status | Onde |
|---|---|---|---|
| 1 | Carga da base bruta | ✅ | `src/data_loader.load_raw` |
| 2 | Split estratificado treino/teste (random_state=42) com guarda anti-leakage | ✅ | `src/data_loader.load_test(unlock_token)` — sentinela `I_AM_IN_FINAL_EVALUATION` |
| 3 | EDA sobre o treino | ✅ | `notebooks/recife_dengue_ldmc_jvlm2.ipynb` |
| 4 | Preparação baseline (imputação + OneHot, sem balance/FE) | ✅ | `src.experiment_runner.build_preprocessor` |
| 5 | Busca de HP × 10 algoritmos + curvas treino vs validação | ✅ | `experiments/{algo}.py` × 10 + artifacts MLflow |
| 6 | Baseline consolidado (v1) | ✅ | `reports/figures/tabela_v1_baseline.csv` |
| 7 | 3 variantes: SMOTE, TargetEncoder, SelectKBest | ✅ | `experiments/{algo}_v{2,3,4}.py` × 30 |
| 8 | Nova busca de HP por variante | ✅ | mesmo |
| 9 | Comparação Wilcoxon pareado | ✅ | `scripts/wilcoxon_paired.py` — 30 testes + Holm-Bonferroni |
| 10 | Seleção do modelo final | ✅ | `lightgbm + v2_smote` (F1=0.4413, rank_biserial +1.000 vs v1) |
| 11 | Avaliação única no teste (CM + ROC + PR macro) | ✅ | `scripts/final_evaluation.py` |

---

## Variantes do treino (4)

| Variante | Técnica | Hipótese | Resultado empírico |
|---|---|---|---|
| **v1_baseline** | OneHot only, sem balance | ponto de comparação | F1 médio = 0.4163 |
| **v2_smote** | SMOTE no Pipeline (imblearn) | reduzir viés pra classe minoritária | **6/10 melhoraram**; lightgbm +0.0150 (best); LVQ/SVM degradaram |
| **v3_target_enc** | TargetEncoder em categóricas | reduzir sparsity OneHot | Δ ≤ \|0.0073\| em todos — **refutada** (categóricas de alta cardinalidade já tinham caído no cleanup) |
| **v4_selectk** | SelectKBest k=15 + MI | curse of dimensionality | **10/10 degradaram** (range -0.0025 a -0.0179) — k=15 corta sinal genuíno; refutada com efeito mais forte |

Detalhes em `reports/wilcoxon_paired_summary.md`.

---

## 10 algoritmos obrigatórios

| Algoritmo | Slug | Estimador final | Busca HP |
|---|---|---|---|
| K-NN | `knn` | `KNeighborsClassifier` | Grid |
| LVQ | `lvq` | `sklvq.GLVQ` (shim p/ sklearn ≥1.6 em `experiments/lvq.py:GLVQCompat`) | Grid |
| Árvore | `decision_tree` | `DecisionTreeClassifier` | Grid |
| SVM | `svm` | `LinearSVC` ⚠ (Linear, não RBF — vide Limitação L1) + `CalibratedClassifierCV(sigmoid)` em v2/3/4/final | Randomized |
| Random Forest | `random_forest` | `RandomForestClassifier` | Randomized |
| MLP | `mlp` | `MLPClassifier(early_stopping=True)` | Randomized |
| Comitê de RNAs | `rna_committee` | `BaggingClassifier(MLPClassifier)` | Randomized |
| Stacking | `stacking` | base={knn, rf, mlp}, meta=`LogisticRegression` ⚠ (sem SVM como base — vide L2) | Grid em combos pequenos |
| XGBoost | `xgboost` | `XGBClassifier(tree_method='hist')` | Randomized |
| LightGBM | `lightgbm` | `LGBMClassifier` | Randomized |

---

## Resultado principal — modelo final no teste

**Modelo escolhido**: `lightgbm + v2_smote` (justificativa estatística em [`reports/wilcoxon_paired_summary.md`](reports/wilcoxon_paired_summary.md))

| | CV (busca) | Teste | Δ (gap) |
|---|---|---|---|
| F1-macro | 0.4413 | **0.4363** | -0.0050 |
| ROC-AUC macro | — | 0.6918 | — |
| PR-AUC macro | — | 0.4579 | — |

**Per-class (teste)**:

| Classe | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 0 Descartado | 0.579 | 0.653 | **0.614** | 0.690 | 0.629 |
| 1 Comum | 0.689 | 0.613 | **0.649** | 0.687 | 0.724 |
| 2 **Alerta/Grave** | 0.036 | 0.063 | **0.046** | 0.698 | **0.021** |

**Generalization gap CV → teste = -0.0050** — modelo generaliza fielmente; CV foi estimador confiável.

**Lição didática (ROC × PR em desbalanceamento extremo)**: ROC-AUC da classe Alerta/Grave = 0.698 (a mais alta!) parece dizer "OK"; PR-AUC = **0.021** revela o problema real (teto estrutural: 80 amostras de teste, 0.6% prevalência). Por isso reportamos PR-AUC como métrica primária ao cliente (Davis & Goadrich 2006; Saito & Rehmsmeier 2015).

---

## Limitações metodológicas conhecidas

Documentadas integralmente em `wiki/faculdade/mineracao-de-dados/triagem-dengue.md` (vault Obsidian dos autores).

- **L1**: SVM apenas kernel **Linear** (`LinearSVC`), não RBF — runtime. Wrap em `CalibratedClassifierCV(sigmoid)` (Platt 1999) pra ter `predict_proba` em v2/3/4/final.
- **L2**: Stacking sem SVM como base estimator — runtime.
- **L3**: Stacking v2 (SMOTE) — distribution shift no meta-learner. Documentado em § "Limitações" do vault.
- **L4**: Wilcoxon retroativo para v1/v2 via `scripts/recover_fold_scores.py` (re-roda CV em vez de extrair `split{0..4}_test_score` do `cv_results_` — runs antigos não logaram esse artifact).
- **L5**: Implementação própria — não usamos o template oficial [`projeto-mda-crisp-dm`](https://github.com/leandrolma3/projeto-mda-crisp-dm). Paridade funcional verificada a posteriori (vide § "Reprodutibilidade" do relatório, a ser escrito).
- **L6** (Wilcoxon): N=5 folds limita p-value two-sided exato a 0.0625 — significância estrita α=0.05 fora do alcance. Reportamos `rank_biserial` + Δ médio como effect size.
- **L7** (Avaliação final): classe Alerta/Grave tem 321 amostras de treino + 80 de teste — teto matemático pra recall; F1=0.046 reflete limitação estrutural, **não falha do modelo**.

---

## Reprodutibilidade

- **`random_state = 42`** em TODOS os splits, estimadores e seeds.
- **`StratifiedKFold(5, shuffle=True, random_state=42)`** em todas as buscas HP.
- **Sentinela anti-leakage**: `src/data_loader.UNLOCK_TOKEN = "I_AM_IN_FINAL_EVALUATION"`. Única chamada autorizada de `load_test()` está em `scripts/final_evaluation.py` (verificável via `grep -rn "I_AM_IN_FINAL_EVALUATION" --include='*.py' .`).
- **MLflow como source of truth**: 40 runs com tags `algoritmo`, `variante`, `source` permitem replicar todas as análises sem rodar nada de novo.

---

## Uso de IA generativa (declaração de integridade)

Este projeto usou assistentes de IA generativa (Claude, da Anthropic) em três frentes: **(1) revisão de código e diagnóstico de bugs** (ex.: `_normalize_source_artifact_uris` em `sync_mlflow_from_apuana.py`); **(2) escrita de scripts auxiliares** (`recover_fold_scores.py`, `final_evaluation.py`, `build_cross_variant_comparison.py`, `wilcoxon_paired.py`); **(3) discussão de escolhas metodológicas e estatísticas**. Todas as decisões metodológicas finais foram revisadas e aprovadas pelos autores. Todo código foi testado funcionalmente. Os 40 runs experimentais foram conduzidos de forma reproduzível em cluster SLURM (Apuana/CIn-UFPE) com MLflow como tracking.

---

## Referências

1. **Demšar, J. (2006).** Statistical Comparisons of Classifiers over Multiple Data Sets. *JMLR* 7:1-30.
2. **Holm, S. (1979).** A simple sequentially rejective multiple test procedure. *Scand. J. Statistics* 6:65-70.
3. **Nadeau, C. & Bengio, Y. (2003).** Inference for the Generalization Error. *Machine Learning* 52:239-281.
4. **Chawla, N. V. et al. (2002).** SMOTE: Synthetic Minority Over-sampling Technique. *JAIR* 16:321-357.
5. **Micci-Barreca, D. (2001).** A Preprocessing Scheme for High-Cardinality Categorical Attributes…. *SIGKDD Explorations* 3(1):27-32.
6. **Platt, J. (1999).** Probabilistic Outputs for Support Vector Machines… *Advances in Large Margin Classifiers*.
7. **Hand, D. & Till, R. (2001).** A Simple Generalisation of the AUC for Multiple Class Classification. *Machine Learning* 45(2):171-186.
8. **Davis, J. & Goadrich, M. (2006).** The Relationship Between Precision-Recall and ROC Curves. *Proc. ICML*.
9. **Saito, T. & Rehmsmeier, M. (2015).** The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLOS ONE* 10(3):e0118432.
10. **Kerby, D. S. (2014).** The Simple Difference Formula: An Approach to Teaching Nonparametric Correlation. *Comprehensive Psychology* 3:1.
11. **Hughes, G. F. (1968).** On the mean accuracy of statistical pattern recognizers. *IEEE TIT* 14(1):55-63.
12. **Chapman, P. et al. (2000).** CRISP-DM 1.0: Step-by-step data mining guide. SPSS Inc.
13. **Wright, S. P. (1992).** Adjusted P-values for simultaneous inference. *Biometrics* 48:1005-13.
