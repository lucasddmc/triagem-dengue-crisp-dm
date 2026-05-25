"""
Pasta `experiments/` — scripts auto-suficientes pra cada (algoritmo × variante).

## Nota de design — duplicação intencional

Cada arquivo `{algoritmo}_v{N}.py` é um script standalone que importa
`src.experiment_runner.run_experiment` e chama com:
- `algorithm` (tag MLflow)
- `variant` (v1_baseline | v2_smote | v3_target_enc | v4_selectk)
- `model_factory` (lambda que retorna instância nova do estimador)
- `param_grid` (dict sklearn-compatible)
- `search_method` ("grid" | "random") + `n_iter`
- `main_hp_for_curve` + `curve_range` (validation curve)
- `extra_pipeline_steps` (lista de tuplas pra inserir entre preprocessor e model)
- `preprocessor_builder` (callable opcional pra trocar OneHot por TargetEncoder etc)

A duplicação entre os 30+ arquivos (10 algoritmos × 3-4 variantes) é
**intencional**. Refatorar pra um runner paramétrico (ex: YAML config +
`run_all.py --algo X --variant Y`) foi avaliado em maio/2026 e **rejeitado**
pelas seguintes razões:

1. **Explícito > implícito**: o diff entre arquivos (factory + grid + extra_steps)
   é EXATAMENTE o que importa pra reprodutibilidade. Ler o arquivo de uma só vez
   é mais rápido que decifrar YAML + lookup table.
2. **SLURM compatibility**: o array job executa `python experiments/{ALGO}.py` por
   nome de arquivo, sem indireção.
3. **ROI baixo no prazo**: o projeto é de disciplina (entrega 10/06); refator pra
   runner paramétrico não justifica o risco/tempo.
4. **Princípio aplicado consistentemente**: o tracking URI também é explícito no
   `experiment_runner.py` (ver achado #2 em wiki/triagem-dengue.md sobre razão da
   decisão de fazer URI explícito vs implícito).

Ao adicionar nova variante (v5+), copiar template de uma variante existente,
trocar `variant=`, `extra_pipeline_steps=`/`preprocessor_builder=`, e (se
diferente) o `param_grid`. NÃO refatorar pra config externa sem revisar essa nota.
"""
