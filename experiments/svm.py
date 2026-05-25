"""
SVM (v1_baseline) — LinearSVC.

DECISÃO DE EXECUÇÃO (2026-05-21): a 1ª submissão usava `SVC` com grid
{linear, rbf, poly}, C log-uniforme 0.01-100, gamma 6 valores. Em 52k amostras,
SVC com kernel rbf é O(n²-n³): rodou 13h+ sem terminar (job 665, cluster-node3
em estado ocioso, descartada hipótese de contenção). Cancelado e trocado por
`LinearSVC` (resolve dual problem via liblinear → linear em n e p, viável em
52k amostras).

Trade-off documentado: perdemos exploração de kernels não-lineares. Mantemos C
log-uniforme e mais combos no random search pra compensar.

Grid restrito: C 0.01-100 (log), loss {hinge, squared_hinge}, dual {auto}.

Execução: python experiments/svm.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import loguniform
from sklearn.svm import LinearSVC

from src.data_loader import RANDOM_STATE
from src.experiment_runner import run_experiment


def main():
    run_experiment(
        algorithm="svm",
        variant="v1_baseline",
        model_factory=lambda: LinearSVC(
            random_state=RANDOM_STATE,
            dual="auto",
            max_iter=5000,
        ),
        param_grid={
            "model__C": loguniform(0.01, 100),
            "model__loss": ["hinge", "squared_hinge"],
        },
        search_method="random",
        n_iter=20,
        main_hp_for_curve="model__C",
        curve_range=[0.01, 0.1, 1.0, 10.0, 100.0],
    )


if __name__ == "__main__":
    main()
