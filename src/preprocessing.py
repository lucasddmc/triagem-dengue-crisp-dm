"""
Preprocessing determinístico do dataset bruto.

Aplicado dentro de load_raw() — limpeza que NÃO depende de hiperparâmetro nem
de variante experimental. Decisões como balanceamento, escala e codificação
ficam nos Pipelines do sklearn (configuráveis por variante v1-v4).

Lógica portada do notebook (cell 29 + cell 30).
"""
import numpy as np
import pandas as pd

# ---------- Colunas a remover ----------

# Vazamento de target ou colunas com informação só disponível APÓS o diagnóstico.
# Dropar é mandatório (senão data leakage trivial).
LEAKAGE_AND_TARGET = [
    # Source columns do target
    "tp_classificacao_final", "classi_fin",
    # Critério de confirmação / evolução (definidos depois)
    "criterio", "evolucao", "tp_evolucao_caso", "tp_criterio_confirmacao",
    # Hospitalização (consequência clínica)
    "hospitaliz", "st_ocorreu_hospitalizacao", "hospital",
    "tel_hospital", "tel_hosp", "ddd_hospital", "ddd_hosp",
    "co_uf_hospital", "co_municipio_hospital",
    # Resultados de exame (vêm depois)
    "tp_result_exame", "tp_result_ns1", "tp_result_isolamento",
    "tp_result_rtpcr", "tp_result_histopatologia", "tp_result_imunohistoquimica",
    "resul_pcr_", "resul_ns1", "resul_prnt", "resul_soro", "resul_vi_n",
    "histopa_n", "imunoh_n", "sorotipo", "tp_sorotipo",
    "conf_fhd", "con_fhd", "laco_n", "plaq_menor",
    # Confirmação/classificação de outras arboviroses
    "clinc_chik", "res_chiks1", "res_chiks2",
    # Complicações já confirmadas (são consequência da gravidade)
    "complica", "mani_hemor",
]

# IDs, números de notificação, ano, etc. Sem valor preditivo.
ADMIN_AND_IDS = [
    "nu_notific", "tp_not", "id_agravo", "tp_notificacao", "nu_ano", "nu_idade_n",
    "notificacao_ano", "co_cid", "co_unidade_notificacao", "id_unidade",
    "id_regiona", "id_regional", "co_uf_notificacao", "sg_uf_not",
    "co_pais_residencia", "id_pais", "co_pais_infeccao", "__arquivo_origem",
    "__ano_recurso", "nu_notificacao", "ds_semana_notificacao", "ds_semana_sintoma",
    "sem_pri", "sem_not", "tp_autoctone_residencia", "nu_idade", "id_mn_resi",
]

# Endereço específico — bairro, CEP, município. Causaria overfitting trivial
# (cada bairro vira um "rótulo" altamente correlacionado com a classe).
ADDRESS_SPECIFIC = [
    # Residência
    "no_bairro_residencia", "nm_bairro", "id_bairro", "co_bairro_residencia",
    "nome_logradouro_residencia", "nm_logrado", "nu_cep_residencia", "nu_cep",
    "co_distrito_residencia", "id_distrit", "co_municipio_notificacao",
    "co_uf_residencia", "id_municip", "co_regional_residencia", "id_rg_resi",
    "tp_zona_residencia", "co_municipio_infeccao", "co_uf_infeccao",
    "co_municipio_residencia", "sg_uf",
    # Infecção
    "no_bairro_infeccao", "nobaiinf", "co_bairro_infeccao", "co_bainf",
    "co_distrito_infeccao", "codisinf", "copaisinf", "coufinf", "comuninf",
    "municipio", "uf", "tpautocto", "cs_zona",
    # Logradouro residencial extra
    "co_logradouro_residencia", "id_logrado", "nu_lote_i",
    # Ocupação (proxy de endereço/admin)
    "co_cbo_ocupacao", "id_ocupa_n",
    # Texto livre — não usável sem NLP, drop por simplicidade
    "ds_obs",
]


# ---------- Feature engineering ----------

def _add_semana_num(df):
    """Extrai número da semana epidemiológica de ds_semana_notificacao."""
    if "ds_semana_notificacao" not in df.columns:
        return df
    s = df["ds_semana_notificacao"].astype(str).str.strip()
    s = s.str.extract(r"(\d+)")[0]
    s = s.str[-2:]
    s = pd.to_numeric(s, errors="coerce")
    df = df.copy()
    df["semana_num"] = s.fillna(-1).astype(int)
    return df


# ---------- Helpers ----------

def _drop_existing(df, cols):
    """Drop colunas que existem; ignora silenciosamente as que não existem."""
    return df.drop(columns=[c for c in cols if c in df.columns])


def _drop_date_cols(df):
    """Drop colunas brutas de data (qualquer 'dt_*'). Substituídas por semana_num."""
    date_cols = [c for c in df.columns if str(c).startswith("dt_")]
    return df.drop(columns=date_cols)


def _drop_high_null_cols(df, threshold=0.90):
    """Drop colunas com proporção de NaN >= threshold."""
    null_pct = df.isnull().mean()
    cols = null_pct[null_pct >= threshold].index.tolist()
    return df.drop(columns=cols)


def _normalize_dtypes(df):
    """
    Compatibilidade com sklearn: converte colunas StringDtype (que usam pd.NA
    como missing) pra object dtype com np.nan. Necessário porque o sklearn
    espera np.nan, não pd.NA — caso contrário SimpleImputer levanta
    'boolean value of NA is ambiguous'.
    """
    out = df.copy()
    for col in out.columns:
        if isinstance(out[col].dtype, pd.StringDtype):
            out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
    return out


def _normalize_categorical_values(df):
    """
    Remove '.0' redundante de valores categóricos que viraram string.
    Ex.: '1' e '1.0' (que representam o mesmo valor mas estavam como
    int e float no parquet original) viram ambos '1'.
    Sem isso, cada coluna binária 1/2 vira 4 categorias e o OneHot explode.
    """
    out = df.copy()
    for col in out.select_dtypes(exclude="number").columns:
        s = out[col].astype(str).str.replace(r"\.0$", "", regex=True)
        # Restaura NaN onde a string 'nan' apareceu por causa do astype(str)
        out[col] = s.where(s != "nan", np.nan)
    return out


def _convert_binary_to_numeric(df, cols, na_value=None):
    """
    Converte colunas binárias categóricas ('1'/'2') pra numéricas (1/0).
    Evita duas dummies redundantes por coluna no OneHotEncoder.

    Tratamento de NaN:
    - `na_value=None` (default): preserva NaN como float (deixa o Pipeline de
      cada modelo decidir como imputar — recomendado).
    - `na_value=0`: assume "não preenchido = ausente" (perde info de missing).
    - `na_value=-1`: sentinel (preserva info; OK pra tree-based, distorce K-NN/SVM).
    """
    out = df.copy()
    mapping = {"1": 1, "2": 0}
    for col in cols:
        if col in out.columns:
            mapped = out[col].map(mapping)
            if na_value is None:
                # Preserva NaN como float (Int64 nullable não funciona bem em sklearn)
                out[col] = mapped.astype(float)
            else:
                out[col] = mapped.fillna(na_value).astype(int)
    return out


# tp_X são versões "novas" (SINAN atual); cs_X / sg_X são versões "antigas"
# das mesmas variáveis demográficas. Manter o tp_X e dropar o duplicado.
DEMOGRAPHIC_DUPLICATES = [
    "cs_sexo", "cs_gestant", "cs_raca", "cs_escol_n",
]

# Pares de colunas que representam a MESMA variável (typo / versão antiga vs nova
# do SINAN). Estratégia: manter a primeira, preencher faltantes com a segunda
# via fillna, então dropar a segunda.
DUPLICATE_PAIRS = [
    ("conjuntvit", "conjutivite"),   # conjuntivite (typo no SINAN)
    ("hipertensa", "hipertensao"),   # hipertensão
]

# Colunas binárias do SINAN ('1'=sim, '2'=não, NaN=não preenchido).
# Convertidas pra 0/1 numérico no clean_dataset → evita duas dummies redundantes
# por coluna no OneHotEncoder. NaN é tratado como 0 ("não" por convenção médica
# conservadora — se não foi marcado, assume ausência).
BINARY_SYMPTOMS_AND_COMORBIDITIES = [
    # Sintomas observados no momento da notificação
    "acido_pept", "artralgia", "artrite", "cefaleia", "conjuntvit",
    "dor_costas", "dor_retro", "exantema", "febre", "hematolog",
    "laco", "leucopenia", "mialgia", "nausea", "petequia_n", "vomito",
    # Comorbidades prévias
    "auto_imune", "diabetes", "hepatopat", "hipertensa", "renal",
]


def _unify_duplicate_pairs(df, pairs):
    """Unifica pares de colunas duplicadas via fillna; dropa a segunda."""
    df = df.copy()
    for keep, drop in pairs:
        if keep in df.columns and drop in df.columns:
            df[keep] = df[keep].fillna(df[drop])
            df = df.drop(columns=[drop])
        elif drop in df.columns and keep not in df.columns:
            # caso de borda: só a segunda existe; renomeia
            df = df.rename(columns={drop: keep})
    return df


# ---------- API principal ----------

def clean_dataset(df, high_null_threshold=0.90):
    """
    Aplica limpeza determinística:
    1. Feature engineering: semana_num
    2. Drop colunas de vazamento + target
    3. Drop admin/IDs
    4. Drop endereço específico
    5. Drop colunas duplicadas demográficas (cs_* mantém tp_*)
    6. Drop colunas de data bruta (dt_*)
    7. Unifica pares de colunas duplicadas (typos SINAN) via fillna
    8. Drop colunas com proporção de NaN >= threshold (default 90%)
    9. Normaliza dtype (pd.NA → np.nan p/ compat sklearn)
    10. Normaliza valores categóricos ('1.0' → '1')
    11. Converte binárias 1/2 → 1/0 (sintomas + comorbidades)
    """
    df = _add_semana_num(df)
    df = _drop_existing(df, LEAKAGE_AND_TARGET)
    df = _drop_existing(df, ADMIN_AND_IDS)
    df = _drop_existing(df, ADDRESS_SPECIFIC)
    df = _drop_existing(df, DEMOGRAPHIC_DUPLICATES)
    df = _drop_date_cols(df)
    df = _unify_duplicate_pairs(df, DUPLICATE_PAIRS)
    df = _drop_high_null_cols(df, threshold=high_null_threshold)
    df = _normalize_dtypes(df)
    df = _normalize_categorical_values(df)
    # na_value=None preserva NaN como float — cada Pipeline de modelo decide
    # como imputar. Razões:
    # - K-NN/SVM com sentinel (-1) distorce distâncias (missing fica "2× mais
    #   longe de sim do que ausente" sem justificativa semântica).
    # - XGBoost lida com NaN nativo se Pipeline não imputar antes.
    # - Imputação com mediana/moda no Pipeline funciona pra ambos.
    df = _convert_binary_to_numeric(df, BINARY_SYMPTOMS_AND_COMORBIDITIES, na_value=None)
    return df
