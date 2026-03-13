from __future__ import annotations

import itertools
import math
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse

# Add the parent directory to sys.path when run directly
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gamo.core.model_utils import compress_model, create_ref_fd, fd_rev2irr, red_target_space, rev2irr
from gamo.core.solver_utils import (
    TOL,
    compute_multiobjective_normalization,
    fit_fun_mimbl,
    fit_fun_multi_obj,
    manual_fva_bounds,
)
from gamo.core.target_utils import (
    decode,
    encode,
    evaluate_targets,
    initialize_population,
    translate_pop,
)
from gamo.core.gamo_types import EncodingInfo, FitFunctionOptions, GAMOOptions, StoichiometricModel, TargetSet


def _chr_key(indv: Sequence[int] | np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in np.sort(np.asarray(indv, dtype=int).reshape(-1)))


def adapt_mut_rate(
    pop_tbin: np.ndarray,
    hd_prev: float,
    static_gen_c: int,
    opt_mut: Mapping[str, Any],
) -> tuple[float, float]:
    pop_tbin = np.asarray(pop_tbin, dtype=float)
    if pop_tbin.ndim == 1:
        pop_tbin = pop_tbin.reshape(1, -1)
    n = pop_tbin.shape[0]
    if n <= 1:
        hd = 0.0
    else:
        diffs = np.abs(pop_tbin[:, None, :] - pop_tbin[None, :, :]).mean(axis=2)
        hd = np.triu(diffs, k=1).sum() * float(opt_mut.get("hd_fac", 1.0))
    mut_rate = (
        -float(opt_mut.get("adaptMut_P", 0.0))
        + (float(opt_mut.get("adaptMut_P", 0.0)) * hd)
        + (float(opt_mut.get("adaptMut_I", 0.0)) * float(static_gen_c))
    )
    return float(mut_rate), float(hd)


def transform_fitness(pop_fit: np.ndarray, pop_tbin: np.ndarray, opt_fit_fun: FitFunctionOptions) -> np.ndarray:
    pop_fit = np.asarray(pop_fit, dtype=float).reshape(-1)
    num_int_eff = np.asarray(pop_tbin, dtype=float).sum(axis=1)
    return pop_fit + (pop_fit * float(opt_fit_fun.FIRF) * (float(opt_fit_fun.numIntMaxFit) - num_int_eff))


def _fitfun_payload(value: Any, fit_fun_type: int) -> tuple[float, np.ndarray | None]:
    if fit_fun_type == 0:
        return float(value), None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0, np.zeros(4, dtype=float)
    return float(arr[0]), arr[1:] if arr.size > 1 else np.zeros(4, dtype=float)


def _ensure_2d(pop: np.ndarray) -> np.ndarray:
    pop = np.asarray(pop, dtype=int)
    if pop.ndim == 1:
        pop = pop.reshape(1, -1)
    return pop


def eval_fitness(
    pop: np.ndarray,
    pop_tbin: np.ndarray | None,
    model: StoichiometricModel,
    model_i: StoichiometricModel,
    Np: int,
    targets: TargetSet,
    fit_fun_type: int,
    opt_fit_fun: FitFunctionOptions,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    pop = _ensure_2d(pop)
    pop_fit = np.zeros(Np, dtype=float)
    pop_fd: list[np.ndarray] = [np.array([], dtype=float) for _ in range(Np)]
    pop_obj_val: list[np.ndarray] = [np.array([], dtype=float) for _ in range(Np)]
    for i in range(Np):
        if fit_fun_type == 0:
            _, act_targets_i, _, act_target_bounds_i = evaluate_targets(pop[i, :], targets)
            pop_fit[i], pop_fd[i] = fit_fun_mimbl(model, model_i, act_targets_i, act_target_bounds_i, opt_fit_fun)
        elif fit_fun_type == 1:
            act_targets, act_targets_i, act_target_bounds, act_target_bounds_i = evaluate_targets(pop[i, :], targets)
            pop_fit[i], pop_fd[i], pop_obj_val[i] = fit_fun_multi_obj(
                model,
                model_i,
                act_targets_i,
                act_targets,
                act_target_bounds_i,
                act_target_bounds,
                opt_fit_fun,
            )
        else:
            raise ValueError("Unknown fitness function type")

    pop_fit_obj = pop_fit.copy()
    if pop_tbin is not None and int(opt_fit_fun.minInt):
        pop_fit = transform_fitness(pop_fit, pop_tbin, opt_fit_fun)
    return pop_fit, pop_fit_obj, pop_fd, pop_obj_val


def eval_fitness_mem(
    pop: np.ndarray,
    pop_tbin: np.ndarray,
    chr_map: dict[tuple[int, ...], Any],
    model: StoichiometricModel,
    model_i: StoichiometricModel,
    Np: int,
    targets: TargetSet,
    fit_fun_type: int,
    opt_fit_fun: FitFunctionOptions,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], dict[tuple[int, ...], Any]]:
    pop = _ensure_2d(pop)
    pop_fit = np.zeros(Np, dtype=float)
    pop_fd: list[np.ndarray] = [np.array([], dtype=float) for _ in range(Np)]
    pop_obj_val: list[np.ndarray] = [np.array([], dtype=float) for _ in range(Np)]

    for i in range(Np):
        key = _chr_key(pop[i, :])
        if key in chr_map:
            fit_val, obj_val = _fitfun_payload(chr_map[key], fit_fun_type)
            pop_fit[i] = fit_val
            pop_obj_val[i] = np.array([], dtype=float) if obj_val is None else np.asarray(obj_val, dtype=float)
            continue
        if fit_fun_type == 0:
            _, act_targets_i, _, act_target_bounds_i = evaluate_targets(pop[i, :], targets)
            pop_fit[i], pop_fd[i] = fit_fun_mimbl(model, model_i, act_targets_i, act_target_bounds_i, opt_fit_fun)
            chr_map[key] = float(pop_fit[i])
        elif fit_fun_type == 1:
            act_targets, act_targets_i, act_target_bounds, act_target_bounds_i = evaluate_targets(pop[i, :], targets)
            pop_fit[i], pop_fd[i], pop_obj_val[i] = fit_fun_multi_obj(
                model,
                model_i,
                act_targets_i,
                act_targets,
                act_target_bounds_i,
                act_target_bounds,
                opt_fit_fun,
            )
            chr_map[key] = np.concatenate([[pop_fit[i]], np.asarray(pop_obj_val[i], dtype=float)]).astype(float)
        else:
            raise ValueError("Unknown fitness function type")

    pop_fit_obj = pop_fit.copy()
    if int(opt_fit_fun.minInt):
        pop_fit = transform_fitness(pop_fit, pop_tbin, opt_fit_fun)
    return pop_fit, pop_fit_obj, pop_fd, pop_obj_val, chr_map


def eval_fitness_comb(
    pop: np.ndarray,
    model: StoichiometricModel,
    model_i: StoichiometricModel,
    Np: int,
    targets: TargetSet,
    fit_fun_type: int,
    opt_fit_fun: FitFunctionOptions,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    pop = _ensure_2d(pop)
    pop_fit = np.zeros(Np, dtype=float)
    pop_fd: list[np.ndarray] = [np.array([], dtype=float) for _ in range(Np)]
    pop_obj_val: list[np.ndarray] = [np.array([], dtype=float) for _ in range(Np)]
    for i in range(Np):
        if fit_fun_type == 0:
            _, act_targets_i, _, act_target_bounds_i = evaluate_targets(pop[i, :], targets)
            if act_targets_i.size:
                pop_fit[i], pop_fd[i] = fit_fun_mimbl(model, model_i, act_targets_i, act_target_bounds_i, opt_fit_fun)
        elif fit_fun_type == 1:
            act_targets, act_targets_i, act_target_bounds, act_target_bounds_i = evaluate_targets(pop[i, :], targets)
            if act_targets.size:
                pop_fit[i], pop_fd[i], pop_obj_val[i] = fit_fun_multi_obj(
                    model,
                    model_i,
                    act_targets_i,
                    act_targets,
                    act_target_bounds_i,
                    act_target_bounds,
                    opt_fit_fun,
                )
        else:
            raise ValueError("Unknown fitness function type")
    return pop_fit, pop_fd, pop_obj_val


def _objective_string_to_mask(opt_fit_fun: FitFunctionOptions) -> FitFunctionOptions:
    lead = opt_fit_fun.leadObj
    if isinstance(lead, str):
        lead_map = {"M": 1, "O": 2, "R": 3, "G": 4}
        if lead not in lead_map:
            raise ValueError(f"Invalid lead objective: {lead!r}")
        opt_fit_fun.leadObj = lead_map[lead]
    is_obj = np.zeros(4, dtype=int)
    lead_idx = int(opt_fit_fun.leadObj) - 1
    is_obj[lead_idx] = 1
    if isinstance(opt_fit_fun.isObj, str):
        for char in opt_fit_fun.isObj:
            if char == "M":
                is_obj[0] = 1
            elif char == "O":
                is_obj[1] = 1
            elif char == "R":
                is_obj[2] = 1
            elif char == "G":
                is_obj[3] = 1
            elif char.isspace():
                continue
            else:
                raise ValueError(f"Unknown objective selector: {char!r}")
    else:
        arr = np.asarray(opt_fit_fun.isObj, dtype=int).reshape(-1)
        is_obj[: min(4, arr.size)] = arr[: min(4, arr.size)]
        is_obj[lead_idx] = 1
    opt_fit_fun.isObj = is_obj
    return opt_fit_fun


def _rxn_bound_to_irreversible(model_i: StoichiometricModel, rxn_idx: int, bound: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    irr_idx = np.asarray(model_i.rev2irrev[int(rxn_idx)], dtype=int).reshape(-1)
    lb = np.asarray(bound, dtype=float)[0]
    ub = np.asarray(bound, dtype=float)[1]
    if irr_idx.size == 1:
        return irr_idx, (np.array([lb], dtype=float), np.array([ub], dtype=float))
    if ub <= 0.0:
        return irr_idx, (
            np.array([0.0, max(0.0, -ub)], dtype=float),
            np.array([0.0, max(0.0, -lb)], dtype=float),
        )
    if lb >= 0.0:
        return irr_idx, (
            np.array([lb, 0.0], dtype=float),
            np.array([ub, 0.0], dtype=float),
        )
    return irr_idx, (
        np.array([0.0, 0.0], dtype=float),
        np.array([max(0.0, ub), max(0.0, -lb)], dtype=float),
    )


def initialize_fit_fun(
    model: StoichiometricModel,
    targets: TargetSet,
    pop: np.ndarray,
    pop_tbin: np.ndarray,
    fit_fun_type: int,
    opt_fit_fun: FitFunctionOptions | Mapping[str, Any],
    optFocus: str,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, ...], Any], TargetSet, StoichiometricModel, StoichiometricModel, FitFunctionOptions]:
    opt_fit_fun = FitFunctionOptions.from_mapping(opt_fit_fun)
    pop = _ensure_2d(pop)
    pop_size, K_tot = pop.shape
    opt_fit_fun.numIntMaxFit = int(K_tot)
    opt_fit_fun.optFocus = int(optFocus == "rxns")

    model_i = rev2irr(model)
    model_i.fd_ref = fd_rev2irr(model, model_i, model.fd_ref)
    if model_i.hetRxnNum.size:
        model_i.lb[model_i.hetRxnNum] = 0.0
        model_i.ub[model_i.hetRxnNum] = 0.0
    if model.hetRxnNum.size:
        model.lb[model.hetRxnNum] = 0.0
        model.ub[model.hetRxnNum] = 0.0

    reaction_target_num = np.asarray(targets.rxnNum, dtype=int)
    bound_i: list[tuple[np.ndarray, np.ndarray]] = []
    rxn_num_i: list[np.ndarray] = []
    for rxn_idx, bound in zip(reaction_target_num.tolist(), np.asarray(targets.bound, dtype=float), strict=False):
        irr_idx, (lb_i, ub_i) = _rxn_bound_to_irreversible(model_i, int(rxn_idx), np.asarray(bound, dtype=float))
        rxn_num_i.append(irr_idx)
        bound_i.append((lb_i, ub_i))
    targets.rxnNum_i = rxn_num_i
    targets.bound_i = bound_i
    targets.rev2irrev = model_i.rev2irrev

    excl_rxns_model = np.asarray(opt_fit_fun.excl_rxns, dtype=int).reshape(-1)
    if excl_rxns_model.size:
        if model.comprMapMat is not None and model.comprMapMat.shape[0] != model.n_rxns:
            valid = excl_rxns_model[(excl_rxns_model >= 0) & (excl_rxns_model < model.comprMapMat.shape[0])]
            if valid.size:
                mapped = model.comprMapMat[valid, :]
                excl_rxns_model = np.flatnonzero(np.asarray(mapped.sum(axis=0)).reshape(-1) != 0)
            else:
                excl_rxns_model = np.array([], dtype=int)
        else:
            excl_rxns_model = excl_rxns_model[(excl_rxns_model >= 0) & (excl_rxns_model < model.n_rxns)]

    if fit_fun_type == 0:
        if excl_rxns_model.size:
            excl_rows = model_i.mapIrr2Rev[excl_rxns_model, :]
            opt_fit_fun.excl_rxns_i = np.flatnonzero(np.asarray(excl_rows.sum(axis=0)).reshape(-1) != 0)
    elif fit_fun_type == 1:
        opt_fit_fun = _objective_string_to_mask(opt_fit_fun)
        if excl_rxns_model.size:
            excl_rows = model_i.mapIrr2Rev[excl_rxns_model, :]
            opt_fit_fun.excl_rxns_i = np.flatnonzero(np.asarray(excl_rows.sum(axis=0)).reshape(-1) != 0)
        opt_fit_fun = compute_multiobjective_normalization(model, opt_fit_fun, np.asarray(opt_fit_fun.isObj, dtype=int))
    else:
        raise ValueError("Unknown fitness function type")

    pop_fit, pop_fit_obj, _, pop_obj_val = eval_fitness(pop, pop_tbin, model, model_i, pop_size, targets, fit_fun_type, opt_fit_fun)
    chr_map: dict[tuple[int, ...], Any] = {}
    for i in range(pop_size):
        key = _chr_key(pop[i, :])
        if fit_fun_type == 0:
            chr_map[key] = float(pop_fit_obj[i])
        else:
            chr_map[key] = np.concatenate([[pop_fit_obj[i]], np.asarray(pop_obj_val[i], dtype=float)]).astype(float)
    return pop_fit, pop_fit_obj, chr_map, targets, model, model_i, opt_fit_fun


def selection(
    pop: np.ndarray,
    pop_tbin: np.ndarray,
    pop_bin: np.ndarray,
    pop_fit: np.ndarray,
    Nslct: int,
    Ndel: int,
    num_pairs: int,
    avoid_dubl: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_int = np.asarray(pop_tbin, dtype=float).sum(axis=1)
    order = np.lexsort((num_int, -np.asarray(pop_fit, dtype=float)))
    pop_sort_slct = order[:Nslct]
    pop_fit_slct = np.asarray(pop_fit, dtype=float)[pop_sort_slct]
    min_prob = 1.0 / max(10 * Nslct, 1)

    if np.all(pop_fit_slct <= 0.0) or np.allclose(pop_fit_slct, pop_fit_slct[0]):
        probs = np.ones(Nslct, dtype=float) / Nslct
    else:
        worst_removed = np.sort(np.asarray(pop_fit, dtype=float))[::-1][Nslct] if len(pop_fit) > Nslct else 0.0
        pop_fit_norm = pop_fit_slct - (worst_removed if worst_removed >= 0.0 else 0.0)
        if np.any(pop_fit_norm <= 0.0):
            pop_fit_norm = pop_fit_norm + ((np.sum(pop_fit_norm) * min_prob) / (1.0 - (min_prob * Nslct)))
        probs = pop_fit_norm / np.sum(pop_fit_norm)

    mating_pairs: set[tuple[int, int]] = set()
    mating_mat: list[list[int]] = []
    max_sample = 100
    for _ in range(num_pairs):
        for _trial in range(max_sample):
            first = int(np.random.choice(Nslct, p=probs))
            second_candidates = np.arange(Nslct) != first
            second_probs = probs.copy()
            second_probs[~second_candidates] = 0.0
            second_probs = second_probs / second_probs.sum()
            second = int(np.random.choice(Nslct, p=second_probs))
            actual_pair = tuple(sorted((int(pop_sort_slct[first]), int(pop_sort_slct[second]))))
            if not avoid_dubl or actual_pair not in mating_pairs:
                mating_pairs.add(actual_pair)
                mating_mat.append([actual_pair[0], actual_pair[1]])
                break
        else:
            first, second = np.random.choice(pop_sort_slct, size=2, replace=False)
            actual_pair = tuple(sorted((int(first), int(second))))
            mating_mat.append([actual_pair[0], actual_pair[1]])
            mating_pairs.add(actual_pair)
    return np.asarray(mating_mat, dtype=int), pop_sort_slct.astype(int)


def mating(
    pop_bin: np.ndarray,
    K_tot: int,
    mating_mat: np.ndarray,
    num_pairs: int,
    numKntChr: int,
    crossType: int,
    enc: EncodingInfo,
) -> np.ndarray:
    num_bits_tot = int(enc.numBits_tot)
    pop_bin = np.asarray(pop_bin, dtype=int)
    mating_mat = np.asarray(mating_mat, dtype=int)

    if crossType == 0:
        pos_cross = np.random.randint(1, K_tot, size=(num_pairs, numKntChr)) if K_tot > 1 else np.zeros((num_pairs, numKntChr), dtype=int)
        pos_cuts = np.asarray(enc.posGene, dtype=int)
    else:
        pos_cross = np.random.randint(1, num_bits_tot, size=(num_pairs, numKntChr)) if num_bits_tot > 1 else np.zeros((num_pairs, numKntChr), dtype=int)
        pos_cuts = np.arange(num_bits_tot, dtype=int)

    offspring1 = np.zeros((num_pairs, num_bits_tot), dtype=int)
    offspring2 = np.zeros((num_pairs, num_bits_tot), dtype=int)
    if numKntChr == 1:
        for i in range(num_pairs):
            cut_idx = int(pos_cross[i, 0])
            if crossType == 0:
                cut = int(pos_cuts[cut_idx])
            else:
                cut = cut_idx
            p1 = pop_bin[mating_mat[i, 0], :]
            p2 = pop_bin[mating_mat[i, 1], :]
            offspring1[i, :] = np.concatenate([p1[:cut], p2[cut:]])
            offspring2[i, :] = np.concatenate([p2[:cut], p1[cut:]])
    else:
        pos_cross = np.sort(pos_cross, axis=1)
        for i in range(num_pairs):
            if crossType == 0:
                cuts = [0] + [int(pos_cuts[c]) for c in pos_cross[i].tolist()] + [num_bits_tot]
            else:
                cuts = [0] + pos_cross[i].astype(int).tolist() + [num_bits_tot]
            p1 = pop_bin[mating_mat[i, 0], :]
            p2 = pop_bin[mating_mat[i, 1], :]
            take_first = True
            for start, stop in zip(cuts[:-1], cuts[1:], strict=False):
                if take_first:
                    offspring1[i, start:stop] = p1[start:stop]
                    offspring2[i, start:stop] = p2[start:stop]
                else:
                    offspring1[i, start:stop] = p2[start:stop]
                    offspring2[i, start:stop] = p1[start:stop]
                take_first = not take_first
    return np.vstack([offspring1, offspring2]).astype(int)


def mutation(
    pop_bin: np.ndarray,
    pop_fit_slct: np.ndarray,
    Np: int,
    Nt: int,
    K_tot: int,
    Nslct: int,
    enc: EncodingInfo,
    mut_rate: float,
    elite: int,
    targets: TargetSet,
    chr_map: dict[tuple[int, ...], Any],
    opt: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pop_bin = np.asarray(pop_bin, dtype=int).copy()
    num_bits_tot = int(enc.numBits_tot)
    max_sample = 100

    if Np > elite:
        mut_mask = np.random.rand(Np - elite, num_bits_tot) < float(mut_rate)
        pop_bin[elite:, :] = np.abs(pop_bin[elite:, :] - mut_mask.astype(int))

    pop = decode(pop_bin, enc)
    pop_tbin = np.zeros((Np, targets.Nt_tot), dtype=int)
    for i in range(Np):
        pop_tbin[i, np.unique(pop[i, :])] = 1

    target_map = np.asarray(targets.map, dtype=int) if not sparse.issparse(targets.map) else np.asarray(targets.map.toarray(), dtype=int)
    rxn_flag = (pop_tbin[:, :Nt] @ target_map) > 1
    rxn_flag_pos = np.where(np.any(rxn_flag, axis=1) if rxn_flag.ndim > 1 else rxn_flag)[0]

    mut_num = max(1, int(math.ceil(float(mut_rate) * num_bits_tot))) if num_bits_tot else 0
    for pos in rxn_flag_pos.tolist():
        pop_bin_s = pop_bin[pos, :].copy()
        for _ in range(max_sample):
            if mut_num:
                mut_bit = np.random.choice(num_bits_tot, size=mut_num, replace=True)
                pop_bin_s[mut_bit] = np.abs(pop_bin_s[mut_bit] - 1)
            pop_s = decode(pop_bin_s, enc).reshape(-1)
            test_flag = np.zeros(targets.Nt_tot, dtype=int)
            test_flag[np.unique(pop_s)] = 1
            rxnflag = test_flag[:Nt] @ target_map
            if not np.any(rxnflag > 1):
                pop[pos, :] = pop_s
                pop_tbin[pos, :] = test_flag
                pop_bin[pos, :] = pop_bin_s
                break
    return pop, pop_tbin, pop_bin


def genetic_algorithm(
    model: StoichiometricModel,
    model_i: StoichiometricModel,
    targets: TargetSet,
    pop: np.ndarray,
    pop_tbin: np.ndarray,
    pop_bin: np.ndarray,
    pop_fit: np.ndarray,
    enc: EncodingInfo,
    chr_map: dict[tuple[int, ...], Any],
    fit_fun_type: int,
    opt_ga: GAMOOptions | Mapping[str, Any],
    opt_fit_fun: FitFunctionOptions,
) -> tuple[dict[str, Any], dict[tuple[int, ...], Any], dict[str, Any]]:
    opt_ga = GAMOOptions.from_mapping(opt_ga)
    pop = _ensure_2d(pop)
    pop_tbin = np.asarray(pop_tbin, dtype=int)
    pop_bin = np.asarray(pop_bin, dtype=int)
    pop_fit = np.asarray(pop_fit, dtype=float).reshape(-1)

    Np = pop.shape[0]
    K_tot = pop.shape[1]
    threads = max(int(opt_ga.threads), 1)
    if Np % threads != 0:
        threads = 1
    Np_s = Np // threads
    Nslct = int(math.ceil(float(opt_ga.slctRate) * Np_s))
    if Np_s >= 2:
        Nslct = max(2, Nslct)
    Nslct = min(Nslct, max(Np_s - 2, 1))
    Ndel = Np_s - Nslct
    if Ndel % 2 != 0:
        if Nslct > 2:
            Nslct -= 1
        else:
            Nslct += 1
        Ndel = Np_s - Nslct
    if Ndel < 2 and Np_s >= 4:
        Ndel = 2
        Nslct = Np_s - Ndel
    num_pairs = max(Ndel // 2, 0)

    timing_drifts = np.zeros(int(opt_ga.maxGen), dtype=float)
    timing_gen: list[np.ndarray] = []
    pop_gen_tot: list[list[np.ndarray]] = []
    pop_fit_gen_tot: list[list[np.ndarray]] = []
    pop_fit_obj_gen_tot: list[list[np.ndarray]] = []
    gen_fit: list[np.ndarray] = []
    pop_obj_fit = pop_fit.copy()
    pop_obj_val: list[np.ndarray] = [np.array([], dtype=float) for _ in range(pop.shape[0])]

    for gen_num in range(int(opt_ga.maxGen)):
        gen_start = time.perf_counter()
        order = np.argsort(pop_fit)[::-1]
        shuffle = np.random.permutation(len(order))
        pop = pop[order[shuffle], :]
        pop_tbin = pop_tbin[order[shuffle], :]
        pop_bin = pop_bin[order[shuffle], :]
        pop_fit = pop_fit[order[shuffle]]

        island_gen: list[np.ndarray] = []
        island_fit: list[np.ndarray] = []
        island_fit_obj: list[np.ndarray] = []
        merged_pop: list[np.ndarray] = []
        merged_tbin: list[np.ndarray] = []
        merged_bin: list[np.ndarray] = []
        merged_fit: list[np.ndarray] = []
        merged_fit_obj: list[np.ndarray] = []
        merged_obj_val: list[np.ndarray] = []
        gen_timing = np.zeros((int(opt_ga.genSize), threads), dtype=float)

        for p in range(threads):
            start = p * Np_s
            stop = start + Np_s
            pop_s = pop[start:stop, :].copy()
            pop_tbin_s = pop_tbin[start:stop, :].copy()
            pop_bin_s = pop_bin[start:stop, :].copy()
            pop_fit_s = pop_fit[start:stop].copy()
            pop_fit_obj_s = pop_fit_s.copy()
            pop_obj_val_s: list[np.ndarray] = [np.array([], dtype=float) for _ in range(Np_s)]
            island_chr_map = chr_map if int(opt_ga.memPop) else {}

            island_pop_hist: list[np.ndarray] = []
            island_fit_hist: list[np.ndarray] = []
            island_fit_obj_hist: list[np.ndarray] = []
            for g in range(int(opt_ga.genSize)):
                t0 = time.perf_counter()
                mating_mat, pop_slct = selection(
                    pop_s,
                    pop_tbin_s,
                    pop_bin_s,
                    pop_fit_s,
                    Nslct,
                    Ndel,
                    num_pairs,
                    int(opt_ga.noMatingIdent),
                )
                offspring_bin = mating(
                    pop_bin_s,
                    K_tot,
                    mating_mat,
                    num_pairs,
                    int(opt_ga.numKntChr),
                    int(opt_ga.crossType),
                    enc,
                )
                pop_bin_s_new = np.vstack([pop_bin_s[pop_slct, :], offspring_bin])
                pop_fit_slct = pop_fit_s[pop_slct]
                pop_s_new, pop_tbin_s_new, pop_bin_s_new = mutation(
                    pop_bin_s_new,
                    pop_fit_slct,
                    Np_s,
                    int(targets.Nt),
                    K_tot,
                    Nslct,
                    enc,
                    float(opt_ga.mutRate),
                    int(opt_ga.elite),
                    targets,
                    island_chr_map,
                    {"mutRate": opt_ga.mutRate},
                )
                if int(opt_ga.memPop):
                    pop_fit_s_new, pop_fit_obj_s_new, _, pop_obj_val_s, island_chr_map = eval_fitness_mem(
                        pop_s_new,
                        pop_tbin_s_new,
                        island_chr_map,
                        model,
                        model_i,
                        Np_s,
                        targets,
                        fit_fun_type,
                        opt_fit_fun,
                    )
                else:
                    pop_fit_s_new, pop_fit_obj_s_new, _, pop_obj_val_s = eval_fitness(
                        pop_s_new,
                        pop_tbin_s_new,
                        model,
                        model_i,
                        Np_s,
                        targets,
                        fit_fun_type,
                        opt_fit_fun,
                    )
                pop_s = pop_s_new
                pop_tbin_s = pop_tbin_s_new
                pop_bin_s = pop_bin_s_new
                pop_fit_s = pop_fit_s_new
                pop_fit_obj_s = pop_fit_obj_s_new
                island_pop_hist.append(pop_s_new.copy())
                island_fit_hist.append(pop_fit_s_new.copy())
                island_fit_obj_hist.append(pop_fit_obj_s_new.copy())
                gen_timing[g, p] = time.perf_counter() - t0

            merged_pop.append(pop_s)
            merged_tbin.append(pop_tbin_s)
            merged_bin.append(pop_bin_s)
            merged_fit.append(pop_fit_s)
            merged_fit_obj.append(pop_fit_obj_s)
            merged_obj_val.extend(pop_obj_val_s)
            island_gen.append(np.array(island_pop_hist, dtype=object))
            island_fit.append(np.array(island_fit_hist, dtype=object))
            island_fit_obj.append(np.array(island_fit_obj_hist, dtype=object))
            chr_map = island_chr_map

        pop_merge = np.vstack(merged_pop)
        pop_tbin_merge = np.vstack(merged_tbin)
        pop_bin_merge = np.vstack(merged_bin)
        pop_fit_merge = np.concatenate(merged_fit_obj)
        pop_obj_fit_merge = pop_fit_merge.copy()
        if int(opt_fit_fun.minInt):
            pop_fit_merge = transform_fitness(pop_fit_merge, pop_tbin_merge, opt_fit_fun)
        shuffle = np.random.permutation(pop_merge.shape[0])
        pop = pop_merge[shuffle, :]
        pop_tbin = pop_tbin_merge[shuffle, :]
        pop_bin = pop_bin_merge[shuffle, :]
        pop_fit = pop_fit_merge[shuffle]
        pop_obj_fit = pop_obj_fit_merge[shuffle]
        pop_obj_val = [merged_obj_val[i] for i in shuffle.tolist()]

        gen_fit.append(np.sort(pop_fit_merge)[::-1])
        timing_drifts[gen_num] = time.perf_counter() - gen_start
        timing_gen.append(gen_timing)
        pop_gen_tot.append([arr.tolist() for arr in island_gen])
        pop_fit_gen_tot.append([arr.tolist() for arr in island_fit])
        pop_fit_obj_gen_tot.append([arr.tolist() for arr in island_fit_obj])

    final_pop = {
        "pop": pop,
        "pop_Tbin": pop_tbin,
        "pop_bin": pop_bin,
        "popFit": pop_fit,
        "popObjFit": pop_obj_fit,
        "popObjVal": pop_obj_val,
        "genFit": gen_fit,
        "timing": {"drifts": timing_drifts, "gen": timing_gen},
    }
    total_data = {
        "pop": pop_gen_tot,
        "popFit": pop_fit_gen_tot,
        "popFitObj": pop_fit_obj_gen_tot,
        "timing_drifts": timing_drifts,
        "timing_gen": timing_gen,
    }
    return final_pop, chr_map, total_data


def analyze_prop_prog(pop_fit_mean: Sequence[float]) -> tuple[int, np.ndarray, np.ndarray]:
    pop_fit_mean = np.asarray(pop_fit_mean, dtype=float).reshape(-1)
    num_gen = len(pop_fit_mean)
    if num_gen <= 3:
        return -1, np.zeros((0, 2), dtype=float), np.ones(0, dtype=float)
    limit = 0.04
    plateau_gen = -1
    denom = max(np.max(pop_fit_mean), TOL)
    pop_fit_mean_norm = pop_fit_mean / denom
    x = np.zeros((num_gen - 1, 2), dtype=float)
    resnorm = np.ones(num_gen - 1, dtype=float)
    for i in range(num_gen - 3):
        if (num_gen - i) < 50:
            v = np.arange(i, num_gen)
        else:
            v = np.arange(i, i + 51)
        coeff = np.polyfit(v, pop_fit_mean_norm[v], 1)
        x[i, :] = coeff
        y = np.polyval(coeff, v)
        resnorm[i] = np.mean((y - pop_fit_mean_norm[v]) ** 2)
        if i > 50:
            if coeff[0] <= 0 and (num_gen - i) >= 50:
                plateau_gen = i
                break
            if np.all(x[i - 50 : i + 1, 0] < limit):
                plateau_gen = i - 50
                break
    return int(plateau_gen), x, resnorm


def _decompress_flux(model_original: StoichiometricModel, model_work: StoichiometricModel, flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=float).reshape(-1)
    if flux.size == 0:
        return np.array([], dtype=float)
    if model_work.comprMapVec is None:
        return flux.copy()
    out = np.zeros(model_original.n_rxns, dtype=float)
    map_vec = np.asarray(model_work.comprMapVec, dtype=int)
    out[map_vec[: flux.size]] = flux[: len(map_vec)]
    return out


def _build_results_from_population(
    model_original: StoichiometricModel,
    model_work: StoichiometricModel,
    final_pop: dict[str, Any],
    chr_map: dict[tuple[int, ...], Any],
    targets: TargetSet,
    optFocus: str,
    enc: EncodingInfo,
    fit_fun_type: int,
    model_i: StoichiometricModel,
    opt_fit_fun: FitFunctionOptions,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[dict[str, Any]]]]:
    pop = np.asarray(final_pop["pop"], dtype=int)
    pop_fit = np.asarray(final_pop["popFit"], dtype=float)
    order = np.argsort(pop_fit)[::-1]
    pop_sort = pop[order, :]
    pop_fit_sort = np.maximum(pop_fit[order], 0.0)
    res: list[dict[str, Any]] = []
    for indv, fit in zip(pop_sort, pop_fit_sort, strict=False):
        KO, KD, Gene_KO, Gene_KD, Ins = translate_pop(model_original, model_work, indv, optFocus, targets, enc.K, enc.K_hri)
        res.append(
            {
                "KO": KO,
                "KD": KD,
                "Gene_KO": Gene_KO,
                "Gene_KD": Gene_KD,
                "Ins": Ins,
                "fitness": float(fit),
                "pop": indv.copy(),
            }
        )

    best_items = sorted(chr_map.items(), key=lambda item: float(np.asarray(item[1]).reshape(-1)[0]), reverse=True)
    res_best: list[dict[str, Any]] = []
    for key, value in best_items:
        indv = np.asarray(key, dtype=int)
        KO, KD, Gene_KO, Gene_KD, Ins = translate_pop(model_original, model_work, indv, optFocus, targets, enc.K, enc.K_hri)
        entry = {
            "KO": KO,
            "KD": KD,
            "Ins": Ins,
            "Gene_KO": Gene_KO,
            "Gene_KD": Gene_KD,
            "fitness": float(np.asarray(value).reshape(-1)[0]),
            "pop": indv.copy(),
        }
        if fit_fun_type == 1:
            entry["objVal"] = np.asarray(value).reshape(-1)[1:]
        res_best.append(entry)

    opt_fit_fun_eval = FitFunctionOptions.from_mapping(opt_fit_fun)
    opt_fit_fun_eval.minInt = 0
    for collection in (res_best[:50], res[:50]):
        for entry in collection:
            _, _, pop_fd, pop_obj_val = eval_fitness(entry["pop"], None, model_work, model_i, 1, targets, fit_fun_type, opt_fit_fun_eval)
            entry["fluxes"] = _decompress_flux(model_original, model_work, pop_fd[0])
            entry["popObjVal"] = pop_obj_val[0]

    best_sol = min(10, len(res_best))
    res_comb: list[list[dict[str, Any]]] = []
    for i in range(best_sol):
        comb_entries: list[dict[str, Any]] = []
        best_pop = np.asarray(res_best[i]["pop"], dtype=int)
        insertion_targets = set(best_pop[enc.K : enc.K_tot].tolist()) if enc.K_hri else set()
        for j in range(1, enc.K_tot):
            for combo in itertools.combinations(best_pop.tolist(), j):
                combo_arr = np.asarray(combo, dtype=int)
                K_hri_c = int(np.isin(combo_arr, list(insertion_targets)).sum())
                K_c = len(combo_arr) - K_hri_c
                targets_h = TargetSet(**targets.__dict__)
                targets_h.K = K_c
                targets_h.K_hri = K_hri_c
                KO, KD, _, _, Ins = translate_pop(model_original, model_work, combo_arr, optFocus, targets_h, K_c, K_hri_c)
                pop_fit_c, _, _ = eval_fitness_comb(combo_arr, model_work, model_i, 1, targets_h, fit_fun_type, opt_fit_fun_eval)
                comb_entries.append({"KO": KO, "KD": KD, "Ins": Ins, "fit": float(pop_fit_c[0]), "pop": combo_arr})
        res_comb.append(comb_entries)
    return res, res_best, res_comb


def GAMO(
    model: StoichiometricModel,
    opt: GAMOOptions | Mapping[str, Any] | None,
    opt_fit_fun: FitFunctionOptions | Mapping[str, Any] | None,
    prob: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prob_dict = dict(prob or {})
    opt = GAMOOptions.from_mapping(opt)
    opt_fit_fun = FitFunctionOptions.from_mapping(opt_fit_fun)
    model = model.copy()

    model.refresh_special_reaction_indices()
    if model.fd_ref is None or np.asarray(model.fd_ref).size == 0:
        model, _, _ = create_ref_fd(model, None, 0, {})
    model_original = model.copy()

    if model.bmRxnNum is None and model.bmRxn is not None:
        model.refresh_special_reaction_indices()
    if model.subsRxnNum is None and model.subsRxn is not None:
        model.refresh_special_reaction_indices()
    if model.targetRxnNum is None and model.targetRxn is not None:
        model.refresh_special_reaction_indices()

    non_target: list[int] = []
    for item in opt.nonTarget:
        if isinstance(item, str):
            if item in model.rxns:
                non_target.append(model.reaction_index(item))
        else:
            non_target.append(int(item))

    if int(opt.compress):
        model = compress_model(model)
        if model.comprMapVec is None:
            model.comprMapVec = np.arange(model.n_rxns, dtype=int)
    else:
        model.comprMapVec = np.arange(model.n_rxns, dtype=int)
        model.comprMapMat = sparse.eye(model.n_rxns, format="csr")

    if int(opt.redFlag):
        non_target_red = red_target_space(model, [], int(opt.modelType))
        min_flux, max_flux = manual_fva_bounds(model)
        blocked = np.where((np.abs(min_flux) <= TOL) & (np.abs(max_flux) <= TOL))[0]
        non_target.extend(non_target_red.tolist())
        non_target.extend(blocked.tolist())

    if model.hetRxnNum.size:
        non_target.extend(model.hetRxnNum.tolist())
    non_target = sorted(set(v for v in non_target if 0 <= int(v) < model.n_rxns))

    if prob_dict.get("targets") is not None:
        targets = TargetSet(**dict(prob_dict["targets"])) if not isinstance(prob_dict["targets"], TargetSet) else prob_dict["targets"]
    else:
        tar_rxn_num = np.setdiff1d(np.arange(model.n_rxns, dtype=int), np.asarray(non_target, dtype=int))
        targets = TargetSet(
            rxnNum=tar_rxn_num,
            score=np.ones(tar_rxn_num.size, dtype=float) / max(tar_rxn_num.size, 1),
            bound=np.zeros((tar_rxn_num.size, 2), dtype=float),
            map=np.eye(tar_rxn_num.size, dtype=int),
            rxnNum_hri=np.asarray(model.hetRxnNum, dtype=int).copy(),
            score_hri=np.ones(model.hetRxnNum.size, dtype=float),
            bound_hri=np.column_stack([model.lb[model.hetRxnNum], model.ub[model.hetRxnNum]]) if model.hetRxnNum.size else np.zeros((0, 2), dtype=float),
            Nt_hri=int(model.hetRxnNum.size),
            fd_ref=np.asarray(model.fd_ref, dtype=float).copy(),
            K=int(opt.numInt),
            K_hri=int(opt.numInsertions if model.hetRxnNum.size else 0),
        )

    if prob_dict.get("pop") is not None and prob_dict.get("enc") is not None:
        pop = _ensure_2d(np.asarray(prob_dict["pop"], dtype=int))
        enc = prob_dict["enc"] if isinstance(prob_dict["enc"], EncodingInfo) else EncodingInfo(**dict(prob_dict["enc"]))
        pop_bin = np.asarray(prob_dict.get("pop_bin", encode(pop, enc.numBits, enc.numBits_hri, enc.encodeVec_midPos, enc.K, enc.K_hri)), dtype=int)
        if prob_dict.get("pop_Tbin") is not None:
            pop_tbin = np.asarray(prob_dict["pop_Tbin"], dtype=int)
        else:
            pop_tbin = np.zeros((pop.shape[0], targets.Nt_tot), dtype=int)
            for i in range(pop.shape[0]):
                pop_tbin[i, np.unique(pop[i, :])] = 1
    else:
        pop, pop_tbin, pop_bin, enc, targets = initialize_population(
            model,
            targets,
            int(opt.popSize),
            float(opt.slctPressure),
            int(opt.numInt),
            int(opt.numInsertions if model.hetRxnNum.size else 0),
            opt.optFocus,
            int(opt.threads),
            int(opt.initPopType),
            {"elite": opt.elite},
        )

    pop_fit, pop_fit_obj, chr_map, targets, model, model_i, opt_fit_fun = initialize_fit_fun(
        model,
        targets,
        pop,
        pop_tbin,
        int(opt.fitFun),
        opt_fit_fun,
        opt.optFocus,
    )

    final_pop, chr_map, total_data = genetic_algorithm(
        model,
        model_i,
        targets,
        pop,
        pop_tbin,
        pop_bin,
        pop_fit,
        enc,
        chr_map,
        int(opt.fitFun),
        opt,
        opt_fit_fun,
    )

    res, res_best, res_comb = _build_results_from_population(
        model_original,
        model,
        final_pop,
        chr_map,
        targets,
        opt.optFocus,
        enc,
        int(opt.fitFun),
        model_i,
        opt_fit_fun,
    )

    results = {
        "res": res,
        "res_best": res_best,
        "res_comb": res_comb,
        "timing": final_pop["timing"],
        "totalData": total_data,
    }

    prob_out = {
        "model": model,
        "model_i": model_i,
        "targets": targets,
        "enc": enc,
        "pop": final_pop["pop"],
        "pop_Tbin": final_pop["pop_Tbin"],
        "pop_bin": final_pop["pop_bin"],
        "popFit": final_pop["popFit"],
        "popObjFit": final_pop["popObjFit"],
        "popObjVal": final_pop["popObjVal"],
        "chr_map": chr_map,
        "opt": opt,
        "opt_fitFun": opt_fit_fun,
        "totalData": total_data,
    }

    if opt.saveFile:
        save_folder = Path(opt.saveFolder or ".")
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / (str(opt.saveFile) if str(opt.saveFile).endswith(".pkl") else f"{opt.saveFile}.pkl")
        with open(save_path, "wb") as handle:
            pickle.dump({"results": results, "prob": prob_out}, handle)
        results["save_path"] = str(save_path)

    return results, prob_out


if __name__ == "__main__":
    print("ga_utils.py imported successfully!")
