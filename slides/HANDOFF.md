# HANDOFF — Apresentação 27/05/2026 (Triagem-Dengue, IF1014)

**Para**: jvlm2 (apresentador) + Lucas
**Duração**: 15 min cronometrados (interrupção se exceder)
**Formato**: 18 slides Beamer PDF (template em [`wiki/.../src-slides-template.md`](../../../Vaults/ObsidianVault/wiki/faculdade/mineracao-de-dados/src-slides-template.md))
**Restrição**: sem código nos slides — só bullets, tabelas, figuras

---

## Pre-flight checklist

Antes de abrir o Beamer:

- [ ] Conferir que a versão do repo no commit é a mais recente (`git log -1`)
- [ ] Conferir que os 22 artifacts estão em `reports/figures/` (`ls reports/figures/*.png | wc -l` deve dar ≥18)
- [ ] Conferir que `reports/wilcoxon_paired_summary.md` e `reports/figures/final_test_classification_report.txt` existem
- [ ] Decidir divisão de fala entre você (jvlm2) e Lucas — sugestão na § "Divisão sugerida" abaixo
- [ ] Cronometrar pelo menos 1 ensaio — 15min é apertado pra 18 slides (~50s/slide)

---

## Slide-by-slide — onde está cada parte

### Slide 1 — Capa

**Conteúdo**:
- Título: "Triagem de Dengue do Recife — Classificação Multiclasse com CRISP-DM"
- Cliente: SEVS/SESAU-Recife (Vigilância Epidemiológica)
- Dataset: SINAN/Recife 2016-2025 (65.267 casos)
- Grupo: Lucas Dantas (`ldmc`) + João V. Mota (`jvlm2`)
- IF1014 - CIn-UFPE - 2026.1
- Data: 27/05/2026

**Imagem**: nenhuma (capa textual)
**Onde estão os dados**: [`README.md`](../README.md) — primeira seção

---

### Slide 2 — Agenda

**Conteúdo**: bullets fixos do template (1-7):
1. Briefing do cliente
2. Dataset e EDA
3. Preparação baseline
4. Modeling + busca HP
5. Avaliação + comparação Wilcoxon
6. Avaliação final no teste + Deployment
7. Conclusões + próximos passos

**Imagem**: nenhuma

---

### Slide 3 — Briefing do cliente (Business Understanding)

**Conteúdo** (5 bullets):
- **Cliente**: SEVS/SESAU-Recife — Vigilância Epidemiológica Municipal
- **Problema**: classificar notificações SINAN suspeitas de dengue em 3 classes
- **Métrica principal**: F1-macro; secundárias: Recall Alerta/Grave, PR-AUC macro
- **Restrições**: alto custo de falso-negativo em Alerta/Grave; latência irrelevante (batch)
- **Riscos éticos**: viés demográfico, privacidade clínica, impacto social desproporcional

**Texto pronto pra copiar**: [`README.md § Briefing do cliente`](../README.md#briefing-do-cliente-fictício) OU `wiki/.../triagem-dengue.md § Briefing formal` (versão completa do edital)

**Imagem**: nenhuma (slide textual)

---

### Slide 4 — Dataset (Data Understanding)

**Conteúdo**:
- Fonte: Portal de Dados Abertos do Recife + harmonização HuggingFace
- Volume: **65.267 linhas** (≥50k exigido ✓)
- 3 classes (multiclasse ✓)
- 198 colunas brutas → 26 após cleanup → 49 pós-OneHot

**Imagem principal**: [`reports/figures/class_distribution.png`](../reports/figures/class_distribution.png)
- 6 barras (3 classes × treino/teste) em escala log; anotações com n absoluto + %
- Confirma visualmente split estratificado 80/20 (mesmo % nas duas pilhas)
- Caption no rodapé reforça: "Alerta/Grave = 0.6% → motiva v2_smote e PR-AUC"

**Tabela complementar**: [`README.md § Briefing do cliente`](../README.md#briefing-do-cliente-fictício) — tabela classe × códigos × n

---

### Slide 5 — 3 achados que orientaram o projeto

**Conteúdo** (3 bullets):
1. **Classe Alerta/Grave = 0.6% da base** (321 treino + 80 teste) — desbalanceamento extremo motivou v2_smote
2. **Colunas de alta cardinalidade já caíram no cleanup** (bairro, município com >70% missing ou leak) — motivou refutação do v3_target_enc
3. **49 features pós-OneHot pra dataset com long-tail de raridade** — motivou v4_selectk (que acabou refutado)

**Imagem**: nenhuma (slide textual com bullets)

**Texto-fonte**: [`wiki/.../triagem-dengue.md § Achados durante refactor`](../../../Vaults/ObsidianVault/wiki/faculdade/mineracao-de-dados/triagem-dengue.md) — § 1-14 são todos achados

---

### Slide 6 — Data Preparation: pipeline baseline

**Conteúdo**:
- Numéricas: `SimpleImputer(strategy='median')` + (sem escala — entra como variante v3)
- Categóricas: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`
- **Sem balanceamento** (entra como v2)
- **Sem feature engineering** (entra como v4)
- Justificativa: baseline = ponto de comparação para medir efeito de cada melhoria

**Imagem**: opcional — diagrama do pipeline (Beamer pode desenhar com TikZ ou usar caixas markdown)
**Código-fonte**: [`src/experiment_runner.py:build_preprocessor`](../src/experiment_runner.py) (linhas 110-126)

---

### Slide 7 — Variantes do treino (3)

**Conteúdo**: tabela das 3 variantes + hipóteses

| Variante | Técnica | Hipótese | Resultado |
|---|---|---|---|
| v2 | SMOTE no Pipeline | reduzir viés p/ minoritária | 6/10 melhoraram |
| v3 | TargetEncoder em categóricas | reduzir sparsity OneHot | **refutada** |
| v4 | SelectKBest k=15 + MI | curse of dimensionality | **refutada (mais forte)** |

**Imagem**: nenhuma (tabela inline)

**Texto-fonte**: [`README.md § Variantes do treino`](../README.md#variantes-do-treino-4)

---

### Slide 8 — Modeling: dez algoritmos obrigatórios

**Conteúdo**: tabela 2 colunas:
- **Parte 1**: K-NN, LVQ, Árvore, SVM, Random Forest
- **Parte 2**: MLP, Comitê RNA, Stacking, XGBoost, LightGBM

**⚠ Citar limitações L1+L2 aqui**: SVM = LinearSVC (não RBF); Stacking sem SVM como base.

**Imagem**: nenhuma
**Texto-fonte**: [`README.md § 10 algoritmos obrigatórios`](../README.md#10-algoritmos-obrigatórios)

---

### Slide 9 — Busca de hiperparâmetros: curva treino vs validação

**Imagem principal**: [`reports/figures/validation_curves_grid_v1_baseline.png`](../reports/figures/validation_curves_grid_v1_baseline.png)
- Grid 3×3 com 9 algoritmos do baseline (stacking sem curva pq não tem HP escalar única)
- Linha vermelha = config escolhida em cada algoritmo

**Mensagem-chave** (~30s):
- Trade-off viés-variância visível
- Para a apresentação, focar em 1-2 algoritmos no grid (ex: KNN → claro overfit com n_neighbors baixo; LightGBM → estabilidade após num_leaves=50)

**Alternativa de imagem**: usar um único algoritmo do grid (mais legível em 15min) — você teria que recortar.

---

### Slide 10 — Melhores configurações (baseline consolidado)

**Conteúdo**: tabela 10 linhas (1 por algoritmo) com:
- Modelo, F1-macro CV ± std, configuração (resumida)

**Dados-fonte**: [`reports/figures/tabela_v1_baseline.csv`](../reports/figures/tabela_v1_baseline.csv) — abrir no Excel/cat e formatar pro Beamer

**Top-3 a destacar**:
1. lightgbm: F1=0.4264
2. xgboost: F1=0.4246
3. decision_tree: F1=0.4223

---

### Slide 11 — Evaluation do baseline

**Imagem principal**: [`reports/figures/barplot_v1_baseline.png`](../reports/figures/barplot_v1_baseline.png)
- 10 algoritmos do baseline ranqueados por F1-macro CV

**Mensagem-chave** (~30s):
- Top-3 líderes: lightgbm, xgboost, decision_tree
- Diferenças ~0.2 desvios entre top-5 — empate técnico no baseline
- Motiva uso de teste pareado pra desempate (próximo slide)

---

### Slide 12 — ⭐ Comparação baseline vs variantes (Wilcoxon)

**Imagem PRINCIPAL** ⭐: [`reports/figures/cross_variant_comparison.png`](../reports/figures/cross_variant_comparison.png)
- 2 painéis: F1-macro à esquerda + Recall Alerta/Grave à direita
- 10 algoritmos × 4 variantes lado a lado

**Mensagem-chave** (~90s — slide mais denso da apresentação):
- **v2_smote** é a única variante com efeitos positivos consistentes — 6/10 algoritmos melhoraram
- **v3 e v4 refutadas** — Δ ≤ |0.0073| e 10/10 degradaram, respectivamente
- **Wilcoxon pareado com Holm-Bonferroni**: nenhum p_adj < α=0.05 por limite amostral (N=5 folds → p_min two-sided = 0.0625) — reportamos rank_biserial como effect size
- **Painel direito (Recall Alerta/Grave)** revela trade-off interessante: SMOTE leva SVM/LVQ de ~3% a **57-58% recall**, mas com colapso de precision (F1 cai)

**Tabela complementar (opcional, footnote)**: [`reports/wilcoxon_paired_summary.md`](../reports/wilcoxon_paired_summary.md) — pivot completo de 30 testes

---

### Slide 13 — Avaliação final no conjunto de teste

**Conteúdo principal** (tabela ou bullets):

| Métrica | CV (busca) | Teste | Δ (gap) |
|---|---|---|---|
| F1-macro | 0.4413 | **0.4363** | -0.0050 |
| Balanced accuracy | — | 0.4429 | — |
| ROC-AUC macro | — | 0.6918 | — |
| PR-AUC macro | — | 0.4579 | — |

**Imagem**: [`reports/figures/final_test_roc_pr.png`](../reports/figures/final_test_roc_pr.png)
- 1×2 subplots: ROC OvR (per-class + macro) à esquerda; PR OvR (per-class + macro) à direita

**Mensagem-chave** (~60s):
- Modelo escolhido: **`lightgbm + v2_smote`** (F1-macro CV mais alta entre as 40 combinações)
- **Gap ínfimo de -0.005** entre CV e teste → CV foi estimador fiel; sem overfit nos HPs
- ⭐ **Ponto didático canônico**: ROC-AUC da classe Alerta/Grave = 0.698 (a mais alta!) parece dizer "OK"; **PR-AUC = 0.021** revela o problema real (ver Davis & Goadrich 2006)

---

### Slide 14 — Matriz de confusão no teste

**Imagem principal**: [`reports/figures/final_test_confusion_matrix.png`](../reports/figures/final_test_confusion_matrix.png)
- 2 subplots: absoluta (esquerda) + normalizada por linha = recall por classe (direita)

**Mensagem-chave** (~60s):
- Classes 0 (Descartado) e 1 (Comum) — F1 ~0.61-0.65, modelo confunde simetricamente entre elas
- Classe 2 (Alerta/Grave): **5 acertos em 80 amostras → recall 6%**
- Implicação clínica: modelo NÃO substitui revisão médica; serve só como filtro de descarte de baixa precisão
- **Limitação estrutural**: 80 amostras de teste pra classe-alvo é teto matemático

**Texto-fonte**: [`reports/figures/final_test_classification_report.txt`](../reports/figures/final_test_classification_report.txt) — números exatos

---

### Slide 15 — Deployment

**Conteúdo** (5 bullets):
- **Forma de uso**: batch diário sobre notificações novas do SINAN, ranqueado por `proba_2` (Alerta/Grave)
- **Latência**: irrelevante (batch)
- **Riscos**: concept drift, classes raras, custo desigual de erro, viés demográfico
- **Mitigações**: thresholds calibrados com SES, monitoramento de drift, revisão médica obrigatória pra `proba_2` > threshold
- **Monitoramento**: F1-macro semanal, drift em `tp_gestante`/`tp_raca_cor`/semana epidemiológica
- **Retreinamento**: 3-6 meses ou quando drift detectado

**Imagem**: nenhuma
**Texto-fonte**: [`README.md § Limitações`](../README.md#limitações-metodológicas-conhecidas) + `wiki/.../triagem-dengue.md § Deployment plan`

---

### Slide 16 — Conclusões

**Conteúdo** (3 bullets):
1. **Atingimos o critério de sucesso?** Parcialmente — F1-macro 0.4363 é o teto possível dada a estrutura do dataset (0.6% prevalência da classe-alvo)
2. **Combinação superior ao baseline?** Sim — `lightgbm + v2_smote` com rank-biserial = +1.000 vs v1 (ganhou em todos os 5 folds)
3. **Aprendizado mais relevante**: SMOTE ajuda em modelos baseados em árvore mas degrada distance-based (SVM, LVQ) por gerar pontos sintéticos em região de baixa densidade; v3/v4 refutadas mostrou que decisões metodológicas devem ser empíricas

**Imagem**: nenhuma

---

### Slide 17 — Limites e próximos passos

**Conteúdo**:

**Limites** (3 bullets condensando L1-L7):
- SVM apenas kernel linear (LinearSVC) por restrição de runtime → L1
- Classe Alerta/Grave com 401 amostras totais → teto matemático pra recall → L7
- N=5 folds limita poder estatístico (p_min = 0.0625) → effect size como mitigação → L6

**Próximos passos** (3 bullets):
- **Coletar mais dados de Alerta/Grave** (única solução real pro recall baixo)
- Tentar SVM RBF + GridSearch externa de k em SelectKBest
- Calibração de threshold por classe pra otimizar Recall em Alerta/Grave

**Recomendação ao cliente**: modelo entra como **filtro de descarte** (não substituto de triagem médica) até atingir recall ≥80% na classe-alvo

**Imagem**: nenhuma
**Texto-fonte**: [`README.md § Limitações`](../README.md#limitações-metodológicas-conhecidas)

---

### Slide 18 — Obrigado / Perguntas

**Conteúdo**:
- Repositório: `https://github.com/lucasddmc/projeto-mineracao-crisp-dm`
- Contatos: `ldmc@cin.ufpe.br`, `jvlm2@cin.ufpe.br`
- 3-5 referências principais:
  - Demšar (2006) — Wilcoxon classifier comparison
  - Chawla et al. (2002) — SMOTE
  - Davis & Goadrich (2006) — PR vs ROC em desbalanceamento
  - Platt (1999) — SVM probability calibration
  - Hand & Till (2001) — multiclass ROC AUC OvR

**Imagem**: nenhuma

---

## 📊 Quick-reference — figuras por slide

| Slide | Figura (arquivo) | Tipo |
|---|---|---|
| 1 | — | (textual) |
| 2 | — | (textual) |
| 3 | — | (textual) |
| 4 | `reports/figures/class_distribution.png` | distribuição treino vs teste (log) |
| 5 | — | (textual) |
| 6 | — (opcional diagrama) | pipeline |
| 7 | — | (tabela inline) |
| 8 | — | (tabela inline) |
| 9 | `reports/figures/validation_curves_grid_v1_baseline.png` | grid 3×3 |
| 10 | — | (tabela inline de `tabela_v1_baseline.csv`) |
| 11 | `reports/figures/barplot_v1_baseline.png` | bar chart |
| 12 ⭐ | `reports/figures/cross_variant_comparison.png` | 2-panel grouped bars |
| 13 | `reports/figures/final_test_roc_pr.png` | ROC + PR macro |
| 14 | `reports/figures/final_test_confusion_matrix.png` | CM 2-panel |
| 15 | — | (textual) |
| 16 | — | (textual) |
| 17 | — | (textual) |
| 18 | — | (textual) |

**Total: 6 figuras embedded + 2 tabelas inline**

---

## 🔢 Quick-reference — números-chave (decorar 5)

Use estes números na apresentação. Confiável; vem do MLflow + Wilcoxon + final eval.

| Número | Valor | Onde |
|---|---|---|
| F1-macro CV do modelo final | **0.4413** | lightgbm_v2_smote |
| F1-macro test do modelo final | **0.4363** | final_evaluation |
| Generalization gap (CV-test) | **-0.005** | minúsculo, modelo generaliza |
| Recall classe Alerta/Grave (test) | **6%** (5/80) | matriz de confusão final |
| PR-AUC Alerta/Grave (test) | **0.021** | curvas PR per-class |
| ROC-AUC Alerta/Grave (test) | 0.698 | curvas ROC per-class (engana!) |
| Δ vs baseline (lightgbm v2) | **+0.0150** | Wilcoxon pareado |
| Rank-biserial vs baseline | **+1.000** | ganhou em todos os 5 folds |
| Prevalência classe Alerta/Grave | **0.6%** | (321+80) / 65.267 |
| N de runs total | **40** | 10 algos × 4 variantes |
| N testes Wilcoxon | **30** | 10 algos × 3 variantes vs v1 |
| N folds CV | **5** | StratifiedKFold |

---

## ⚠ Gaps a resolver antes de slides

Coisas que você (ou Lucas) precisa decidir/gerar antes de abrir o Beamer:

### Gap 1: figura de distribuição de classes (slide 4) — ✅ RESOLVIDO

Figura gerada em [`reports/figures/class_distribution.png`](../reports/figures/class_distribution.png) (2026-05-26).

Implementação: lê `y_train` via `load_train()` (sem restrição) e `y_test` do artifact já publicado em `reports/figures/final_test_y_pred.csv` (resultado da única liberação autorizada do sentinel) — **não toca o sentinel de novo**. Sanity check: razão treino:teste por classe = 4.00× / 4.00× / 4.01× (esperado 4.0 com split 80/20), confirma estratificação perfeita.

### Gap 2: divisão de fala entre Lucas e jvlm2

Sugestão (a calibrar com o que vocês decidirem):

| Slides | Fala | Tempo |
|---|---|---|
| 1-2 | jvlm2 (abertura) | 1min |
| 3-5 | jvlm2 (briefing + dataset + achados) | 3min |
| 6-8 | Lucas (preparação + 10 algos) | 2min |
| 9-12 | Lucas (modeling + comparação Wilcoxon — bloco técnico) | 5min |
| 13-14 | jvlm2 (avaliação final + matriz confusão) | 3min |
| 15-17 | jvlm2 (deployment + conclusões + limites) | 2min |
| 18 | ambos (perguntas) | — |
| **Total** | | **15min** |

Lógica: jvlm2 abre e fecha (apresentador principal); Lucas explica a parte técnica densa (modeling + Wilcoxon).

### Gap 3: cliente fictício — está finalizado?

Resposta: **sim, já está formalizado**. Cliente: **SEVS/SESAU-Recife** (Vigilância Epidemiológica Municipal). Texto completo em [`README.md`](../README.md) + vault.

### Gap 4: template Beamer

Lembrar: o repo do prof **não tem `.tex` template** (verificado em sessão anterior — vide `wiki/.../triagem-dengue.md § L5` se quiser detalhe). Vocês vão precisar:
- Escrever `slides.tex` do zero usando uma template Beamer minimalista qualquer (ex: Madrid, Berlin, Warsaw)
- Estrutura de 18 slides já dada acima
- OU usar Beamer Metropolis (visual moderno, popular em IA)

---

## 🎯 Roadmap pra escrever os slides — sugestão

1. **Hoje (26/05) noite — 4-5h**:
   - [ ] Decidir template Beamer
   - [ ] Escrever slides 1-8 (rotina, baixa densidade técnica)
   - [ ] Inserir as 5 figuras (slides 9, 11, 12, 13, 14)
   - [ ] Compilar PDF rascunho
   - [ ] Cronometrar com leitura corrida (não falando ainda)

2. **Amanhã (27/05) manhã — 2-3h**:
   - [ ] Ensaiar uma vez com cronômetro
   - [ ] Ajustar timing (cortar conteúdo dos slides que estouram)
   - [ ] Ensaiar segunda vez
   - [ ] Apresentar

---

## Arquivos-chave (referência rápida)

| O que | Onde |
|---|---|
| Briefing formal | [`README.md § Briefing`](../README.md) ou `wiki/.../triagem-dengue.md` |
| Resultado modelo final | [`reports/figures/final_test_*`](../reports/figures/) |
| Wilcoxon completo | [`reports/wilcoxon_paired_summary.md`](../reports/wilcoxon_paired_summary.md) |
| Tabelas por variante | [`reports/figures/tabela_v{1..4}.csv`](../reports/figures/) |
| Memória bibliográfica + limitações | `wiki/faculdade/mineracao-de-dados/triagem-dengue.md § Memória pra escrever relatório` |
| Estado completo do projeto | [`README.md`](../README.md) |

---

## ✅ Checklist final pré-apresentação

- [ ] Slides PDF compilado e aberto pra cronometragem
- [ ] 5 figuras inseridas (slides 9, 11, 12, 13, 14)
- [ ] Números-chave decorados (ver § Quick-reference)
- [ ] Cronometragem ≤ 14:45 min em ensaio (margem de segurança 15s)
- [ ] Repositório com último commit publicado no GitHub
- [ ] Prof já adicionado como colaborador ✓
- [ ] Backup do PDF em pen-drive + nuvem (caso o notebook falhe)

Boa apresentação. 🍀
