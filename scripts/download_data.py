"""
Baixa o dataset harmonizado do HuggingFace pra `data/dataset_harmonizado.parquet`.

Reproduz: `lucasddmc/recife-dengue-harmonizado` (68.140 notificações SINAN
2016-2025, 3 classes — Descartado, Comum, Alerta/Grave; ~30 features pós-cleanup).

Idempotente: se o arquivo já existe e tem tamanho > 0, pula download.

Uso:
    python scripts/download_data.py [--force]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset_harmonizado.parquet"
HF_REPO = "lucasddmc/recife-dengue-harmonizado"
HF_FILENAME = "data/dataset_harmonizado.parquet"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true",
                   help="Re-baixa mesmo se arquivo já existe.")
    args = p.parse_args(argv)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0 and not args.force:
        size_mb = DATA_PATH.stat().st_size / (1024 * 1024)
        print(f"✓ Dataset já presente em {DATA_PATH} ({size_mb:.1f} MB). Use --force pra re-baixar.")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface_hub não instalado. Rode: pip install -r requirements.txt")
        return 1

    print(f"Baixando {HF_REPO}/{HF_FILENAME} do HuggingFace Hub...")
    cached = hf_hub_download(HF_REPO, repo_type="dataset", filename=HF_FILENAME)
    shutil.copy(cached, DATA_PATH)
    size_mb = DATA_PATH.stat().st_size / (1024 * 1024)
    print(f"✓ Salvo em {DATA_PATH} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
