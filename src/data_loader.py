# src/data_loader.py
"""
Carregamento de dados com split treino/teste e guarda anti-leakage.
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import clean_dataset

# Path absoluto pro dataset, relativo ao arquivo .py (independe da cwd do Jupyter)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset_harmonizado.parquet"

CODE_TO_CLASS = {
    "1": 1, "10": 1,                                       # Comum
    "11": 2, "12": 2, "2": 2, "3": 2, "4": 2,             # Alerta/Grave
    "5": 0,                                                # Descartado
}

RANDOM_STATE = 42
TEST_SIZE = 0.2

UNLOCK_TOKEN = "I_AM_IN_FINAL_EVALUATION"


def load_raw():
    """Carrega o parquet, mapeia target unificada, aplica limpeza determinística. Retorna (X, y)."""
    df = pd.read_parquet(DATA_PATH)

    # 1. Unifica as 2 colunas-fonte do target (versão nova com fallback pra antiga)
    code = df["tp_classificacao_final"].astype("string").fillna(df["classi_fin"].astype("string"))

    # 2. Mapeia código → classe 0/1/2. Códigos não mapeados (incl. "8" inconclusivo) viram NaN.
    y = code.map(CODE_TO_CLASS)

    # 3. Mantém só linhas com classe válida
    mask = y.notna()
    df = df.loc[mask].copy()
    y = y[mask].astype(int)

    # 4. Preprocessing determinístico (drop colunas de leak/admin/endereço/data + FE semana)
    X = clean_dataset(df)

    return X, y


def load_train():
    """Retorna apenas treino, fazendo split estratificado fixo internamente."""
    X, y = load_raw()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_train, y_train


def load_test(unlock_token):
    """Retorna teste só se o token mágico for passado."""
    if unlock_token != UNLOCK_TOKEN:
        raise RuntimeError("Acesso negado: token de desbloqueio inválido.")
    
    X, y = load_raw()
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_test, y_test