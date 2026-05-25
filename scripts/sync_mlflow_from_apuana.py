"""
Importa runs do MLflow do Apuana pro MLflow local.

Pré-requisito: arquivos já transferidos pra máquina local via rsync.
Suporta source em SQLite ou file backend; target idem.

Fluxo:
1. Lê todos os runs do experimento <experiment_name> no source.
2. Pra cada run, cria um equivalente no target (mesmo nome de experimento;
   se não existir, cria).
3. Copia tags + params + métricas + artifacts físicos.
4. Anexa metadados:
   - tag `source=apuana`
   - tag `apuana_run_id=<id_original>`
   - tag `synced_at=<ISO>`

Uso:
    # 1) Antes, rsync do Apuana:
    rsync -avz -e "ssh -i ~/.ssh/id_ed25519_apuana" \\
        ldmc@slurm-client1.cin.ufpe.br:~/triagem-dengue/mlruns/ \\
        ./mlflow_apuana/mlruns/

    # 2) Sync:
    python scripts/sync_mlflow_from_apuana.py \\
        --source-uri file://$PWD/mlflow_apuana/mlruns \\
        --target-uri sqlite:///$PWD/mlflow_dengue.db \\
        --experiment-name triagem-dengue

Notas:
- O script é IDEMPOTENTE: já-sincronizados (mesmo `apuana_run_id`) são pulados.
- Funciona pra qualquer combinação source/target (sqlite ↔ file ↔ remote http).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def _get_or_create_experiment(client: MlflowClient, name: str) -> str:
    """Retorna id do experimento; cria se não existir."""
    exp = client.get_experiment_by_name(name)
    if exp is None:
        exp_id = client.create_experiment(name)
        print(f"  Criado experimento target '{name}' (id={exp_id})")
        return exp_id
    return exp.experiment_id


def _already_synced_run_ids(client: MlflowClient, target_exp_id: str) -> set[str]:
    """Retorna conjunto de `apuana_run_id` já presentes no target (idempotência)."""
    synced = set()
    runs = client.search_runs(
        experiment_ids=[target_exp_id],
        filter_string="tags.source = 'apuana'",
        max_results=10_000,
    )
    for r in runs:
        original = r.data.tags.get("apuana_run_id")
        if original:
            synced.add(original)
    return synced


def _copy_artifacts(source_client: MlflowClient, source_run_id: str,
                    target_client: MlflowClient, target_run_id: str) -> int:
    """Copia todos os artifacts (não-recursivo na raiz; recursivo em subdirs).
    Retorna número de arquivos copiados.
    """
    count = 0
    with tempfile.TemporaryDirectory() as tmp:
        for art in source_client.list_artifacts(source_run_id):
            local_path = source_client.download_artifacts(source_run_id, art.path, dst_path=tmp)
            target_client.log_artifact(target_run_id, local_path,
                                       artifact_path=str(Path(art.path).parent) if "/" in art.path else None)
            count += 1
    return count


def _copy_metrics_with_history(source_client: MlflowClient, source_run_id: str,
                               target_client: MlflowClient, target_run_id: str) -> None:
    """Copia métricas preservando histórico (timestamps + steps) quando possível."""
    src_run = source_client.get_run(source_run_id)
    for metric_key in src_run.data.metrics.keys():
        history = source_client.get_metric_history(source_run_id, metric_key)
        for m in history:
            target_client.log_metric(target_run_id, key=metric_key,
                                     value=m.value, timestamp=m.timestamp, step=m.step)


def sync(source_uri: str, target_uri: str, experiment_name: str,
         dry_run: bool = False, target_experiment_name: str | None = None) -> dict:
    """Executa sync. Retorna stats.

    Se `target_experiment_name` for fornecido, runs do `experiment_name` no
    source são gravados no `target_experiment_name` no target. Útil pra mover
    runs órfãos que caíram no `Default` no source pra um experimento nomeado
    no target.
    """
    if target_experiment_name is None:
        target_experiment_name = experiment_name

    print(f"Source:           {source_uri}")
    print(f"Target:           {target_uri}")
    print(f"Source experiment: {experiment_name}")
    print(f"Target experiment: {target_experiment_name}")
    print(f"Dry run:          {dry_run}\n")

    # Clients independentes (cada um aponta pra seu tracking URI)
    source_client = MlflowClient(tracking_uri=source_uri)
    target_client = MlflowClient(tracking_uri=target_uri)
    # mlflow.start_run() usa tracking URI global do módulo — precisa configurar
    # explicitamente, senão tenta usar default e não acha o experiment criado.
    mlflow.set_tracking_uri(target_uri)

    # Source experiment
    src_exp = source_client.get_experiment_by_name(experiment_name)
    if src_exp is None:
        print(f"❌ Experiment '{experiment_name}' não encontrado no source.")
        return {"copied": 0, "skipped": 0, "error": "no_source_experiment"}

    # Target experiment (cria se não existir)
    if dry_run:
        target_exp = target_client.get_experiment_by_name(target_experiment_name)
        target_exp_id = target_exp.experiment_id if target_exp else "(would-create)"
        synced = set()
    else:
        target_exp_id = _get_or_create_experiment(target_client, target_experiment_name)
        synced = _already_synced_run_ids(target_client, target_exp_id)
    print(f"Target exp_id={target_exp_id}, já sincronizados: {len(synced)}")

    # Itera runs do source
    src_runs = source_client.search_runs(
        experiment_ids=[src_exp.experiment_id],
        max_results=10_000,
    )
    print(f"Source tem {len(src_runs)} runs.\n")

    stats = {"copied": 0, "skipped_already_synced": 0, "skipped_failed": 0}

    for src_run in src_runs:
        rid = src_run.info.run_id
        rname = src_run.info.run_name or "(unnamed)"
        if rid in synced:
            print(f"  ⏭️  {rname} (run_id={rid[:8]}) — já sincronizado, pulando")
            stats["skipped_already_synced"] += 1
            continue
        if src_run.info.status != "FINISHED":
            print(f"  ⏭️  {rname} status={src_run.info.status}, pulando")
            stats["skipped_failed"] += 1
            continue

        print(f"  → {rname} (run_id={rid[:8]})")
        if dry_run:
            stats["copied"] += 1
            continue

        try:
            with mlflow.start_run(
                experiment_id=target_exp_id, run_name=rname,
                tags={
                    "source": "apuana",
                    "apuana_run_id": rid,
                    "apuana_run_name": rname,
                    "synced_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    # Preserva tags úteis do original
                    **{k: v for k, v in src_run.data.tags.items()
                       if not k.startswith("mlflow.")},
                },
            ) as new_run:
                new_rid = new_run.info.run_id
                # Params
                if src_run.data.params:
                    target_client.log_batch(
                        new_rid,
                        params=[mlflow.entities.Param(k, v) for k, v in src_run.data.params.items()],
                    )
                # Métricas (com histórico)
                _copy_metrics_with_history(source_client, rid, target_client, new_rid)
                # Artifacts
                n_arts = _copy_artifacts(source_client, rid, target_client, new_rid)
                print(f"     ✓ {len(src_run.data.params)} params, "
                      f"{len(src_run.data.metrics)} metrics, {n_arts} artifacts")
                stats["copied"] += 1
        except Exception as e:
            print(f"     ❌ ERRO: {e}")
            stats["skipped_failed"] += 1

    print("\n=== Resumo ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-uri", required=True, help="URI MLflow source (Apuana, ex: sqlite:///./mlflow_apuana/mlflow_dengue.db ou file://./mlflow_apuana/mlruns)")
    p.add_argument("--target-uri", required=True, help="URI MLflow target (local)")
    p.add_argument("--experiment-name", default="triagem-dengue", help="Nome do experimento no source")
    p.add_argument("--target-experiment-name", default=None, help="Nome do experimento no target (default: igual ao source). Use pra mover runs órfãos do 'Default' pro 'triagem-dengue'.")
    p.add_argument("--dry-run", action="store_true", help="Não copia, só lista o que faria")
    args = p.parse_args(argv)

    stats = sync(args.source_uri, args.target_uri, args.experiment_name,
                 dry_run=args.dry_run,
                 target_experiment_name=args.target_experiment_name)
    return 0 if stats.get("copied", 0) >= 0 and "error" not in stats else 1


if __name__ == "__main__":
    sys.exit(main())
