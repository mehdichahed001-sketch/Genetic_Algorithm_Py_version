from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to sys.path when run directly
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gamo.core.ga_utils import analyze_prop_prog
from gamo.core.gamo_types import StoichiometricModel


def _load_results(results_file: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(results_file, Mapping):
        return dict(results_file)
    path = Path(results_file)
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, Mapping) and "results" in payload:
        return dict(payload["results"])
    return dict(payload)


def analyze_gamo_results(
    model: StoichiometricModel,
    results_file: str | Path | Mapping[str, Any],
    target: str,
    *,
    fitMin: float = 1.0e-10,
    numRelTarget: int = 15,
    plot: bool = False,
    targetNotation: str = "name",
) -> dict[str, Any]:
    results = _load_results(results_file)
    res_best = list(results.get("res_best", []))
    fitness_best = np.asarray([float(entry.get("fitness", 0.0)) for entry in res_best], dtype=float)
    fit_best_pos = np.where(fitness_best >= float(fitMin))[0]

    if target == "rxns":
        strategies_best = [np.asarray(res_best[i].get("KO", []), dtype=int) for i in fit_best_pos.tolist()]
    elif target == "genes":
        strategies_best = [np.asarray(res_best[i].get("Gene_KO", []), dtype=int) for i in fit_best_pos.tolist()]
    else:
        target = "genes"
        strategies_best = [np.asarray(res_best[i].get("Gene_KO", []), dtype=int) for i in fit_best_pos.tolist()]

    unique_map: dict[tuple[int, ...], float] = {}
    for pos, strat in zip(fit_best_pos.tolist(), strategies_best, strict=False):
        key = tuple(sorted(int(v) for v in np.asarray(strat, dtype=int).reshape(-1)))
        if key not in unique_map:
            unique_map[key] = float(fitness_best[pos])
    unique_strategies = [np.asarray(key, dtype=int) for key in unique_map.keys()]
    unique_fitness = np.asarray(list(unique_map.values()), dtype=float)
    num_strat_unique = len(unique_strategies)

    if not unique_strategies:
        return {"meanFitness_targets": np.array([]), "occurence_targets": np.array([]), "targetNames": []}

    target_ids = np.unique(np.concatenate(unique_strategies))
    target_count = np.zeros(target_ids.size, dtype=float)
    fit_sum = np.zeros(target_ids.size, dtype=float)
    for strat, fit in zip(unique_strategies, unique_fitness, strict=False):
        match = np.isin(target_ids, strat)
        target_count += match.astype(float)
        fit_sum += match.astype(float) * fit

    order = np.argsort(target_count)[::-1]
    target_ids_sorted = target_ids[order]
    target_count_sorted = target_count[order]
    occurrence_norm = 100.0 * (target_count_sorted / max(num_strat_unique, 1))
    mean_fitness = fit_sum[order] / max(num_strat_unique, 1)

    num_rel = min(int(numRelTarget), len(target_ids_sorted))
    target_ids_rel = target_ids_sorted[:num_rel]
    if target == "genes":
        target_names = [model.genes[idx] if idx < len(model.genes) else str(idx) for idx in target_ids_rel.tolist()]
    elif targetNotation == "ID":
        target_names = [model.rxns[idx] if idx < len(model.rxns) else str(idx) for idx in target_ids_rel.tolist()]
    else:
        target_names = [model.rxnNames[idx] if idx < len(model.rxnNames) else model.rxns[idx] for idx in target_ids_rel.tolist()]

    out = {
        "meanFitness_targets": mean_fitness[:num_rel],
        "occurence_targets": occurrence_norm[:num_rel],
        "targetNames": target_names,
    }

    if plot and num_rel:
        plt.figure(figsize=(max(8, num_rel * 0.6), 5))
        plt.bar(np.arange(num_rel), occurrence_norm[:num_rel])
        plt.xticks(np.arange(num_rel), target_names, rotation=45, ha="right")
        plt.ylabel("Occurrence [%]")
        plt.title(f"Occurrences in design strategies (N_unique={num_strat_unique})")
        plt.tight_layout()

        plt.figure(figsize=(max(8, num_rel * 0.6), 5))
        plt.bar(np.arange(num_rel), mean_fitness[:num_rel])
        plt.xticks(np.arange(num_rel), target_names, rotation=45, ha="right")
        plt.ylabel("Mean fitness")
        plt.title("Mean fitness of designs including respective targets")
        plt.tight_layout()

    return out


if __name__ == "__main__":
    print("analysis_utils.py imported successfully!")
