from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import sparse

# Add the parent directory to sys.path when run directly
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gamo.core.solver_utils import (
    INF,
    TOL,
    manual_fva,
    minimize_quadratic_with_linear_constraints,
    optimize_cb_model,
    reaction_extrema,
    solve_lp,
)
from gamo.core.gamo_types import FitFunctionOptions, LinearSolution, StoichiometricModel, as_numpy_int_array


def _mat_array_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value.decode() if isinstance(value, bytes) else value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return [_scalarize(value.item())]
        return [_scalarize(v) for v in value.tolist()]
    return [_scalarize(value)]


def _scalarize(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _scalarize(value.item())
        return [_scalarize(v) for v in value.tolist()]
    if sparse.issparse(value) or isinstance(value, sparse.sparray):
        return sparse.csr_matrix(value)
    if hasattr(value, "_fieldnames"):
        return {name: _scalarize(getattr(value, name)) for name in value._fieldnames}
    return value


def _array_of_str(value: Any) -> list[str]:
    items = _mat_array_to_list(value)
    out: list[str] = []
    for item in items:
        if item is None:
            out.append("")
        elif isinstance(item, (list, tuple)):
            out.append(str(item[0]) if item else "")
        else:
            out.append(str(item))
    return out


def _array_of_float(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    return arr


def _normalize_sparse_matrix(value: Any) -> sparse.csr_matrix:
    if sparse.issparse(value) or isinstance(value, sparse.sparray):
        return sparse.csr_matrix(value)
    return sparse.csr_matrix(np.asarray(value, dtype=float))


def _normalize_subsystems(value: Any, n: int) -> list[str]:
    if value is None:
        return [""] * n
    arr = _mat_array_to_list(value)
    if len(arr) < n:
        arr.extend([""] * (n - len(arr)))
    out: list[str] = []
    for item in arr[:n]:
        if isinstance(item, list):
            out.append("; ".join(str(v) for v in item if str(v)))
        else:
            out.append(str(item))
    return out


def load_matlab_model(path: str | Path, variable_name: str | None = None) -> StoichiometricModel:
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if variable_name is None:
        candidates = [k for k in data.keys() if not k.startswith("__")]
        if len(candidates) == 1:
            variable_name = candidates[0]
        elif "model" in data:
            variable_name = "model"
        else:
            raise ValueError(f"Could not infer model variable from {path!s}")
    raw = data[variable_name]
    fields = {name: getattr(raw, name) for name in raw._fieldnames}
    rxns = _array_of_str(fields.get("rxns"))
    mets = _array_of_str(fields.get("mets"))
    model = StoichiometricModel(
        rxns=rxns,
        mets=mets,
        S=_normalize_sparse_matrix(fields.get("S")),
        lb=_array_of_float(fields.get("lb")),
        ub=_array_of_float(fields.get("ub")),
        c=_array_of_float(fields.get("c")),
        genes=_array_of_str(fields.get("genes")),
        grRules=_array_of_str(fields.get("grRules")),
        rxnGeneMat=_normalize_sparse_matrix(fields.get("rxnGeneMat")) if fields.get("rxnGeneMat") is not None else sparse.csr_matrix((len(rxns), 0)),
        csense=str(fields.get("csense")) if fields.get("csense") is not None else None,
        b=_array_of_float(fields.get("b")) if fields.get("b") is not None else np.zeros(len(mets), dtype=float),
        rev=_array_of_float(fields.get("rev")) if fields.get("rev") is not None else None,
        description=str(fields.get("description")) if fields.get("description") is not None else None,
        rxnNames=_array_of_str(fields.get("rxnNames")) if fields.get("rxnNames") is not None else list(rxns),
        metNames=_array_of_str(fields.get("metNames")) if fields.get("metNames") is not None else list(mets),
        metFormulas=_array_of_str(fields.get("metFormulas")) if fields.get("metFormulas") is not None else [""] * len(mets),
        subSystems=_normalize_subsystems(fields.get("subSystems"), len(rxns)),
    )
    model.refresh_special_reaction_indices()
    if model.c.size == 0:
        model.c = np.zeros(model.n_rxns, dtype=float)
    if model.rxnGeneMat is None:
        model.rxnGeneMat = sparse.csr_matrix((model.n_rxns, len(model.genes)))
    return model


def load_mat_struct(path: str | Path, variable_name: str | None = None) -> dict[str, Any]:
    data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if variable_name is None:
        candidates = [k for k in data.keys() if not k.startswith("__")]
        if len(candidates) == 1:
            variable_name = candidates[0]
        else:
            raise ValueError(f"Could not infer variable from {path!s}")
    raw = data[variable_name]
    return _scalarize(raw)


def save_pickle(obj: Any, path: str | Path) -> None:
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def change_rxn_bounds(
    model: StoichiometricModel,
    reactions: str | int | Sequence[str | int],
    values: float | Sequence[float],
    bound_type: str,
) -> StoichiometricModel:
    out = model.copy()
    if isinstance(reactions, (str, int)):
        reactions = [reactions]
    if np.isscalar(values):
        values = [float(values)] * len(reactions)
    for rxn, value in zip(reactions, values, strict=False):
        idx = out.reaction_index(rxn)
        if bound_type == "l":
            out.lb[idx] = float(value)
        elif bound_type == "u":
            out.ub[idx] = float(value)
        elif bound_type == "b":
            out.lb[idx] = float(value)
            out.ub[idx] = float(value)
        else:
            raise ValueError("bound_type must be one of {'l', 'u', 'b'}")
    return out


def change_objective(model: StoichiometricModel, reaction: str | int) -> StoichiometricModel:
    out = model.copy()
    out.c = np.zeros(out.n_rxns, dtype=float)
    out.c[out.reaction_index(reaction)] = 1.0
    return out


def delete_reactions(model: StoichiometricModel, del_rxns: Sequence[int]) -> StoichiometricModel:
    del_idx = np.array(sorted(set(int(v) for v in del_rxns)), dtype=int)
    if del_idx.size == 0:
        return model.copy()
    keep_rxns = np.setdiff1d(np.arange(model.n_rxns), del_idx)
    out = model.copy()

    orig_rxns = list(model.rxns)
    orig_rxn_names = list(model.rxnNames) if model.rxnNames else list(model.rxns)
    orig_subsystems = list(model.subSystems) if model.subSystems else [""] * model.n_rxns
    orig_gr_rules = list(model.grRules) if model.grRules else [""] * model.n_rxns

    out.S = out.S[:, keep_rxns]
    out.rxns = [orig_rxns[i] for i in keep_rxns]
    out.rxnNames = [orig_rxn_names[i] if i < len(orig_rxn_names) else orig_rxns[i] for i in keep_rxns]
    out.subSystems = [orig_subsystems[i] if i < len(orig_subsystems) else "" for i in keep_rxns]
    out.lb = out.lb[keep_rxns]
    out.ub = out.ub[keep_rxns]
    out.c = out.c[keep_rxns] if out.c.size else np.zeros(keep_rxns.size, dtype=float)
    if out.rev is not None and out.rev.size:
        out.rev = out.rev[keep_rxns]
    if out.fd_ref is not None and out.fd_ref.size:
        out.fd_ref = out.fd_ref[keep_rxns]
    if out.rxnGeneMat is not None:
        out.rxnGeneMat = sparse.csr_matrix(out.rxnGeneMat)[keep_rxns, :]
    if orig_gr_rules:
        out.grRules = [orig_gr_rules[i] for i in keep_rxns]

    if out.hetRxnNum.size:
        keep_map = {old: new for new, old in enumerate(keep_rxns.tolist())}
        out.hetRxnNum = np.asarray([keep_map[idx] for idx in out.hetRxnNum.tolist() if idx in keep_map], dtype=int)

    orphan_mets = np.where(np.asarray(out.S.getnnz(axis=1)).reshape(-1) == 0)[0]
    keep_mets = np.setdiff1d(np.arange(len(out.mets)), orphan_mets)
    orig_mets = list(model.mets)
    orig_met_names = list(model.metNames) if model.metNames else list(model.mets)
    orig_met_formulas = list(model.metFormulas) if model.metFormulas else [""] * model.n_mets
    out.S = out.S[keep_mets, :]
    out.mets = [orig_mets[i] for i in keep_mets]
    out.metNames = [orig_met_names[i] if i < len(orig_met_names) else orig_mets[i] for i in keep_mets]
    if orig_met_formulas:
        out.metFormulas = [orig_met_formulas[i] if i < len(orig_met_formulas) else "" for i in keep_mets]
    if out.b is not None and out.b.size:
        out.b = out.b[keep_mets]
    if isinstance(out.csense, str) and len(out.csense) >= len(model.mets):
        out.csense = "".join(out.csense[i] for i in keep_mets)
    elif isinstance(out.csense, list):
        out.csense = [out.csense[i] for i in keep_mets]

    out.refresh_special_reaction_indices()
    return out


def rev2irr(model: StoichiometricModel) -> StoichiometricModel:
    rows: list[sparse.csr_matrix] = []
    rxns: list[str] = []
    rxn_names: list[str] = []
    lb: list[float] = []
    ub: list[float] = []
    c: list[float] = []
    sub_systems: list[str] = []
    rev2irrev: list[np.ndarray] = []
    irrev2rev: list[int] = []
    match_rev: list[bool] = []
    rxn_gene_rows: list[sparse.csr_matrix] = []

    rxn_gene_mat = sparse.csr_matrix(model.rxnGeneMat) if model.rxnGeneMat is not None else sparse.csr_matrix((model.n_rxns, len(model.genes)))

    for j, rxn_id in enumerate(model.rxns):
        col = model.S[:, j]
        if model.lb[j] < 0 < model.ub[j]:
            f_idx = len(rxns)
            rows.append(col)
            rxns.append(f"{rxn_id}_f")
            rxn_names.append(model.rxnNames[j] if j < len(model.rxnNames) else rxn_id)
            lb.append(0.0)
            ub.append(float(model.ub[j]))
            c.append(float(model.c[j]))
            sub_systems.append(model.subSystems[j] if j < len(model.subSystems) else "")
            rxn_gene_rows.append(rxn_gene_mat.getrow(j))
            irrev2rev.append(j)
            match_rev.append(True)

            b_idx = len(rxns)
            rows.append(-col)
            rxns.append(f"{rxn_id}_b")
            rxn_names.append(model.rxnNames[j] if j < len(model.rxnNames) else rxn_id)
            lb.append(0.0)
            ub.append(float(-model.lb[j]))
            c.append(0.0)
            sub_systems.append(model.subSystems[j] if j < len(model.subSystems) else "")
            rxn_gene_rows.append(rxn_gene_mat.getrow(j))
            irrev2rev.append(j)
            match_rev.append(True)
            rev2irrev.append(np.array([f_idx, b_idx], dtype=int))
        elif model.lb[j] < 0 and model.ub[j] <= 0:
            idx = len(rxns)
            rows.append(-col)
            rxns.append(f"{rxn_id}_r")
            rxn_names.append(model.rxnNames[j] if j < len(model.rxnNames) else rxn_id)
            lb.append(0.0)
            ub.append(float(-model.lb[j]))
            c.append(float(model.c[j]))
            sub_systems.append(model.subSystems[j] if j < len(model.subSystems) else "")
            rxn_gene_rows.append(rxn_gene_mat.getrow(j))
            irrev2rev.append(j)
            match_rev.append(True)
            rev2irrev.append(np.array([idx], dtype=int))
        else:
            idx = len(rxns)
            rows.append(col)
            rxns.append(rxn_id)
            rxn_names.append(model.rxnNames[j] if j < len(model.rxnNames) else rxn_id)
            lb.append(float(max(model.lb[j], 0.0)))
            ub.append(float(model.ub[j]))
            c.append(float(model.c[j]))
            sub_systems.append(model.subSystems[j] if j < len(model.subSystems) else "")
            rxn_gene_rows.append(rxn_gene_mat.getrow(j))
            irrev2rev.append(j)
            match_rev.append(False)
            rev2irrev.append(np.array([idx], dtype=int))

    S_i = sparse.hstack(rows, format="csr").tocsr() if rows else sparse.csr_matrix((model.n_mets, 0))
    rxn_gene_i = sparse.vstack(rxn_gene_rows, format="csr") if rxn_gene_rows else sparse.csr_matrix((0, len(model.genes)))
    out = StoichiometricModel(
        rxns=rxns,
        mets=list(model.mets),
        S=S_i,
        lb=np.asarray(lb, dtype=float),
        ub=np.asarray(ub, dtype=float),
        c=np.asarray(c, dtype=float),
        genes=list(model.genes),
        grRules=list(model.grRules),
        rxnGeneMat=rxn_gene_i,
        csense=model.csense,
        b=None if model.b is None else model.b.copy(),
        rev=np.zeros(len(rxns), dtype=float),
        description=model.description,
        rxnNames=rxn_names,
        metNames=list(model.metNames),
        metFormulas=list(model.metFormulas),
        subSystems=sub_systems,
        bmRxn=model.bmRxn,
        subsRxn=model.subsRxn,
        targetRxn=model.targetRxn,
        fd_ref=None,
        hetRxnNum=np.array([], dtype=int),
    )
    out.matchRev = match_rev
    out.rev2irrev = rev2irrev
    out.irrev2rev = np.asarray(irrev2rev, dtype=int)
    B = sparse.lil_matrix((len(rxns), model.n_rxns), dtype=float)
    for i, orig_idx in enumerate(out.irrev2rev):
        B[i, orig_idx] = 1.0
    out.B = B.tocsr()
    map_irrev_to_rev = sparse.lil_matrix((model.n_rxns, len(rxns)), dtype=float)
    for orig_idx, irr_indices in enumerate(rev2irrev):
        if irr_indices.size > 1:
            map_irrev_to_rev[orig_idx, irr_indices[0]] = 1.0
            map_irrev_to_rev[orig_idx, irr_indices[1]] = -1.0
        else:
            name = rxns[irr_indices[0]]
            map_irrev_to_rev[orig_idx, irr_indices[0]] = -1.0 if name.endswith("_r") else 1.0
    out.mapIrr2Rev = map_irrev_to_rev.tocsr()

    if out.subsRxn is not None and out.subsRxn in model.rxns:
        idx = model.reaction_index(out.subsRxn)
        if model.lb[idx] < 0 and model.ub[idx] <= 0:
            out.subsRxn = f"{out.subsRxn}_r"
        elif model.lb[idx] < 0 < model.ub[idx]:
            out.subsRxn = f"{out.subsRxn}_b"
    if out.targetRxn is not None and out.targetRxn in model.rxns:
        idx = model.reaction_index(out.targetRxn)
        if model.lb[idx] < 0 and model.ub[idx] <= 0:
            out.targetRxn = f"{out.targetRxn}_r"
        elif model.lb[idx] < 0 < model.ub[idx]:
            out.targetRxn = f"{out.targetRxn}_f"
    if model.bmRxn is not None:
        out.bmRxn = model.bmRxn if model.bmRxn in out.rxns else model.bmRxn

    if model.hetRxnNum.size:
        het_i: list[int] = []
        for ridx in model.hetRxnNum:
            het_i.extend(rev2irrev[int(ridx)].tolist())
        out.hetRxnNum = np.asarray(het_i, dtype=int)
    out.refresh_special_reaction_indices()
    return out


def fd_rev2irr(model: StoichiometricModel, model_i: StoichiometricModel, fd: np.ndarray) -> np.ndarray:
    fd = np.asarray(fd, dtype=float).reshape(-1)
    fd_i = np.zeros(model_i.n_rxns, dtype=float)
    for i, value in enumerate(fd):
        irr = model_i.rev2irrev[i]
        if irr.size > 1:
            if value > 0:
                fd_i[irr[0]] = value
            elif value < 0:
                fd_i[irr[1]] = -value
        else:
            name = model_i.rxns[irr[0]]
            fd_i[irr[0]] = -value if name.endswith("_r") else value
    return fd_i


def red_target_space(model: StoichiometricModel, no_target: Sequence[int] | None = None, model_type: int = 0) -> np.ndarray:
    no_target_idx = set(int(v) for v in (no_target or []))
    non_target_subsystems = {
        "cell envelope biosynthesis",
        "transport",
        "membrane lipid metabolism",
        "murein biosynthesis",
        "trna charging",
        "glycerophospholipid metabolism",
    }
    if model_type == 0:
        for i, rxn_name in enumerate(model.rxnNames):
            rxn_name_l = rxn_name.lower()
            if any(token in rxn_name_l for token in ["exchange", "diffusion", "transport", "spontaneous"]):
                no_target_idx.add(i)
        for i, subsystem in enumerate(model.subSystems):
            subsystem_l = subsystem.lower()
            if any(token in subsystem_l for token in non_target_subsystems):
                no_target_idx.add(i)
        if model.grRules:
            for i, rule in enumerate(model.grRules):
                if not str(rule).strip():
                    no_target_idx.add(i)
    return np.array(sorted(no_target_idx), dtype=int)


def _free_flux_null_reactions(model: StoichiometricModel) -> np.ndarray:
    n = model.n_rxns
    lb = np.full(n, -np.inf)
    ub = np.full(n, np.inf)
    min_flux, max_flux, _, _ = reaction_extrema(
        model,
        np.arange(n, dtype=int),
        lb=lb,
        ub=ub,
        return_flux_vectors=False,
        failure_value=-1.0,
    )
    return np.where((abs(max_flux) < TOL) & (abs(min_flux) < TOL))[0]


def compress_model(model: StoichiometricModel) -> StoichiometricModel:
    def _fallback_model() -> StoichiometricModel:
        out = model.copy()
        out.comprMapMat = sparse.eye(model.n_rxns, format="csr")
        out.comprMapVec = np.arange(model.n_rxns, dtype=int)
        out.geneMatCompress = list(model.genes)
        out.refresh_special_reaction_indices()
        return out

    model_u = model.copy()
    keep_map = sparse.eye(model.n_rxns, format="csr")
    while True:
        if model_u.n_rxns == 0:
            return _fallback_model()
        model_i = rev2irr(model_u)
        null_rxns = _free_flux_null_reactions(model_u)
        S_i = model_i.S.toarray()
        enabled = (abs(model_i.lb) > TOL) | (abs(model_i.ub) > TOL)
        rxns_per_met = np.sum((S_i * enabled.reshape(1, -1)) != 0, axis=1)
        source_sink_mets = ((np.all(S_i >= 0, axis=1) | np.all(S_i <= 0, axis=1)) & np.any(S_i != 0, axis=1)) | (rxns_per_met < 2)
        ss_rxns_i = np.where(np.any(S_i[source_sink_mets, :] != 0, axis=0))[0]
        ss_rxns: list[int] = []
        for orig_idx, irr_indices in enumerate(model_i.rev2irrev):
            if irr_indices.size and np.all(np.isin(irr_indices, ss_rxns_i)):
                ss_rxns.append(orig_idx)
        del_rxns = np.array(sorted(set(null_rxns.tolist() + ss_rxns)), dtype=int)
        if del_rxns.size == 0:
            break
        if del_rxns.size >= model_u.n_rxns:
            return _fallback_model()
        model_u = delete_reactions(model_u, del_rxns)
        if model_u.n_rxns == 0:
            return _fallback_model()
        keep_map = keep_map[:, np.setdiff1d(np.arange(keep_map.shape[1]), del_rxns)]

    if model_u.n_rxns == 0 or keep_map.shape[1] == 0:
        return _fallback_model()
    model_u.comprMapMat = keep_map.tocsr()
    model_u.comprMapVec = np.argmax(keep_map.toarray(), axis=0).astype(int)

    # Merge exclusive genes that only encode a single compressed reaction.
    gene_mat = sparse.csr_matrix(model_u.rxnGeneMat)
    genes_save = list(model_u.genes)
    gene_mat_compress = list(model_u.genes)
    for rxn_idx in range(model_u.n_rxns):
        gene_inc = np.where(gene_mat[rxn_idx, :].toarray().reshape(-1) > 0)[0]
        if gene_inc.size <= 1:
            continue
        exclusive_mask = np.asarray(gene_mat[:, gene_inc].sum(axis=0)).reshape(-1) == 1
        if np.sum(exclusive_mask) <= 1:
            continue
        exclusive = gene_inc[exclusive_mask]
        while True:
            uni_gene = "a" + "".join(np.random.randint(0, 10, size=4).astype(str))
            if uni_gene not in model_u.genes:
                break
        for gidx in exclusive:
            gene = model_u.genes[gidx]
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(gene)}(?![A-Za-z0-9_])")
            model_u.grRules = [pattern.sub(uni_gene, rule) for rule in model_u.grRules]
        if gene_inc.size == exclusive.size:
            model_u.grRules[rxn_idx] = uni_gene
        for gidx in exclusive:
            orig_pos = genes_save.index(model_u.genes[gidx]) if model_u.genes[gidx] in genes_save else None
            if orig_pos is not None:
                gene_mat_compress[orig_pos] = uni_gene
        keep_gene_mask = np.ones(len(model_u.genes), dtype=bool)
        keep_gene_mask[exclusive[1:]] = False
        model_u.genes[exclusive[0]] = uni_gene
        model_u.genes = [g for g, keep in zip(model_u.genes, keep_gene_mask, strict=False) if keep]
        gene_mat = gene_mat[:, keep_gene_mask]
        model_u.rxnGeneMat = gene_mat
    model_u.geneMatCompress = gene_mat_compress
    model_u.refresh_special_reaction_indices()
    return model_u


def _parse_formula_side(side: str, sign: float) -> dict[str, float]:
    mets: dict[str, float] = {}
    for term in [t.strip() for t in side.split("+") if t.strip()]:
        match = re.match(r"^(?:(-?\d+(?:\.\d+)?)\s+)?(.+)$", term)
        if not match:
            raise ValueError(f"Cannot parse reaction term: {term!r}")
        coeff = float(match.group(1)) if match.group(1) else 1.0
        met = match.group(2).strip()
        mets[met] = mets.get(met, 0.0) + sign * coeff
    return mets


def _parse_reaction_formula(formula: str) -> tuple[dict[str, float], float, float]:
    if "<=>" in formula:
        lhs, rhs = formula.split("<=>", 1)
        lb, ub = -INF, INF
    elif "-->" in formula:
        lhs, rhs = formula.split("-->", 1)
        lb, ub = 0.0, INF
    elif "<--" in formula:
        rhs, lhs = formula.split("<--", 1)
        lb, ub = 0.0, INF
    else:
        raise ValueError(f"Unknown reaction arrow in formula: {formula!r}")
    stoich = _parse_formula_side(lhs, -1.0)
    for met, coeff in _parse_formula_side(rhs, 1.0).items():
        stoich[met] = stoich.get(met, 0.0) + coeff
    return stoich, lb, ub


def add_network_branches(model: StoichiometricModel, reaction_list: Sequence[Sequence[Any]]) -> StoichiometricModel:
    model_d = model.copy()
    ref_mu = model.fd_ref[model.reaction_index(model.bmRxn)] if model.fd_ref is not None and model.bmRxn is not None else 0.0
    subs_up_rate = model.fd_ref[model.reaction_index(model.subsRxn)] if model.fd_ref is not None and model.subsRxn is not None else 0.0
    mu_fac = np.array([0.98, 1.0], dtype=float)
    flux_rate_std = 2.0
    max_flux_rate = 20.0
    lim_gibbs = 15.0

    rxn_valid: list[bool] = []
    ref_flux: list[float] = []
    inserted_rows: list[Sequence[Any]] = []
    inserted_indices: list[int] = []

    for row in reaction_list:
        formula = str(row[0])
        rxn_id = str(row[1])
        rxn_name = str(row[2])
        gibbs_pred = float(row[7]) if len(row) > 7 and row[7] not in (None, "") else 0.0
        if rxn_id in model_d.rxns or rxn_name in model_d.rxnNames:
            rxn_valid.append(False)
            ref_flux.append(0.0)
            continue
        try:
            model_p = add_reaction(model_d, rxn_id, rxn_name, formula, subsystem="Heterologous reaction")
        except Exception:
            rxn_valid.append(False)
            ref_flux.append(0.0)
            continue
        if model_p.S[:, -1].nnz == 0:
            rxn_valid.append(False)
            ref_flux.append(0.0)
            continue

        model_t = add_reaction(model.copy(), rxn_id, rxn_name, formula, subsystem="Heterologous reaction")
        if model_t.subsRxn is not None:
            model_t = change_rxn_bounds(model_t, model_t.subsRxn, subs_up_rate, "l")
        model_t = change_objective(model_t, model_t.bmRxn)
        sol = optimize_cb_model(model_t, sense="max", one_norm=True)
        if not sol.success:
            rxn_valid.append(False)
            ref_flux.append(0.0)
            continue
        max_mu = sol.f
        flux_max_mu = sol.x[-1]

        flux_rates_ub = np.zeros(2, dtype=float)
        flux_rates_lb = np.zeros(2, dtype=float)
        obj = np.zeros(model_t.n_rxns, dtype=float)
        obj[-1] = 1.0
        for t, fac in enumerate(mu_fac):
            lb = model_t.lb.copy()
            ub = model_t.ub.copy()
            bm_idx = model_t.reaction_index(model_t.bmRxn)
            lb[bm_idx] = max_mu * fac
            ub[bm_idx] = max_mu * fac
            sol_ub = optimize_cb_model(model_t, sense="max", objective=obj, lb_override=lb, ub_override=ub)
            sol_lb = optimize_cb_model(model_t, sense="min", objective=obj, lb_override=lb, ub_override=ub)
            if sol_ub.success:
                flux_rates_ub[t] = sol_ub.f
            if sol_lb.success:
                flux_rates_lb[t] = sol_lb.f

        if abs(flux_max_mu) < TOL:
            if np.all(abs(np.concatenate([flux_rates_ub, flux_rates_lb])) < TOL):
                rxn_valid.append(False)
                ref_flux.append(0.0)
                continue
            if np.all(flux_rates_lb * flux_rates_ub < 0):
                if abs(gibbs_pred) > lim_gibbs:
                    flux = flux_rate_std - ((flux_rate_std * 2.0) * float(gibbs_pred > 0.0))
                else:
                    flux = (flux_rate_std * -gibbs_pred) / lim_gibbs
            else:
                flux = 0.0
                for i, fac in enumerate(mu_fac):
                    if flux_rates_ub[i] > 0.0 and abs(flux_rates_ub[i]) < max_flux_rate:
                        flux = (flux_rates_ub[i] / max(max_mu * fac, TOL)) * ref_mu
                        break
                    if flux_rates_lb[i] < 0.0 and abs(flux_rates_lb[i]) < max_flux_rate:
                        flux = (flux_rates_lb[i] / max(max_mu * fac, TOL)) * ref_mu
                        break
        else:
            if abs(flux_max_mu) < max_flux_rate:
                flux = (flux_max_mu / max(max_mu, TOL)) * ref_mu
            else:
                flux = flux_rate_std - ((flux_rate_std * 2.0) * float(flux_max_mu > 0.0))

        model_d = model_p
        rxn_valid.append(True)
        ref_flux.append(float(flux))
        inserted_rows.append(row)
        inserted_indices.append(model_d.reaction_index(rxn_id))

    if model_d.fd_ref is None:
        model_d.fd_ref = np.zeros(model_d.n_rxns, dtype=float)
    else:
        if model_d.fd_ref.size < model_d.n_rxns:
            model_d.fd_ref = np.concatenate([model_d.fd_ref, np.zeros(model_d.n_rxns - model_d.fd_ref.size)])
    for idx, flux in zip(inserted_indices, ref_flux, strict=False):
        if idx < model_d.fd_ref.size:
            model_d.fd_ref[idx] = flux
    model_d.hetRxnNum = np.asarray(inserted_indices, dtype=int)
    model_d.hetRxns = [list(row) for row in inserted_rows]
    model_d.subSystems = [ss or "NONE" for ss in model_d.subSystems]
    return model_d


def add_reaction(
    model: StoichiometricModel,
    reaction_id: str,
    reaction_name: str,
    reaction_formula: str,
    *,
    subsystem: str = "",
) -> StoichiometricModel:
    stoich, lb, ub = _parse_reaction_formula(reaction_formula)
    out = model.copy()
    new_mets = [met for met in stoich.keys() if met not in out.mets]
    if new_mets:
        add_rows = sparse.csr_matrix((len(new_mets), out.n_rxns), dtype=float)
        out.S = sparse.vstack([out.S, add_rows], format="csr")
        out.mets.extend(new_mets)
        out.metNames.extend(new_mets)
        out.metFormulas.extend([""] * len(new_mets))
        if out.b is None:
            out.b = np.zeros(len(out.mets), dtype=float)
        else:
            out.b = np.concatenate([out.b, np.zeros(len(new_mets), dtype=float)])
        if isinstance(out.csense, str):
            out.csense = out.csense + ("E" * len(new_mets))
    rxn_col = sparse.lil_matrix((len(out.mets), 1), dtype=float)
    for met, coeff in stoich.items():
        rxn_col[out.mets.index(met), 0] = coeff
    out.S = sparse.hstack([out.S, rxn_col.tocsr()], format="csr")
    out.rxns.append(reaction_id)
    out.rxnNames.append(reaction_name)
    out.subSystems.append(subsystem)
    out.lb = np.concatenate([out.lb, np.array([lb])])
    out.ub = np.concatenate([out.ub, np.array([ub])])
    out.c = np.concatenate([out.c, np.array([0.0])])
    if out.rev is not None:
        out.rev = np.concatenate([out.rev, np.array([1.0 if lb < 0 < ub else 0.0])])
    if out.rxnGeneMat is None:
        out.rxnGeneMat = sparse.csr_matrix((out.n_rxns - 1, len(out.genes)))
    out.rxnGeneMat = sparse.vstack([sparse.csr_matrix(out.rxnGeneMat), sparse.csr_matrix((1, len(out.genes)))], format="csr")
    out.grRules.append("")
    out.refresh_special_reaction_indices()
    return out


def create_ref_fd(
    model: StoichiometricModel,
    model_i: StoichiometricModel | None,
    src_flag: int,
    opt: Mapping[str, Any] | None = None,
) -> tuple[StoichiometricModel, StoichiometricModel | None, dict[str, Any]]:
    opt = dict(opt or {})
    filename = opt.get("filename")
    flux_fac = float(opt.get("fluxFac", 1.0))
    res: dict[str, Any] = {}

    if src_flag == 0:
        model = change_objective(model, model.bmRxn)
        sol = optimize_cb_model(model, sense="max", one_norm=True)
        model.fd_ref = sol.x.copy()
        res = {"sol": sol}
    elif src_flag == 1:
        if not filename:
            raise ValueError("No experimental flux data file provided")
        df = pd.read_excel(filename)
        expected_columns = ["Rxn Identifier", "Mean", "LB", "UB"]
        if list(df.columns[:4]) != expected_columns:
            raise ValueError("Excel file with measured fluxes is not consistent with the expected template")

        model_m = model.copy()
        rxns_exp = df["Rxn Identifier"].astype(str).tolist()
        rxns_exp_f = df["Mean"].to_numpy(dtype=float) * flux_fac
        rxns_exp_lb = df["LB"].to_numpy(dtype=float) * flux_fac
        rxns_exp_ub = df["UB"].to_numpy(dtype=float) * flux_fac
        if model.bmRxn in rxns_exp:
            bm_pos = rxns_exp.index(model.bmRxn)
            rxns_exp_f[bm_pos] /= flux_fac
            rxns_exp_lb[bm_pos] /= flux_fac
            rxns_exp_ub[bm_pos] /= flux_fac

        rxn_num_exp: list[int] = []
        rxns_exp_filtered: list[str] = []
        mean_f: list[float] = []
        lb_f: list[float] = []
        ub_f: list[float] = []
        for rxn_id, mean_val, lb_val, ub_val in zip(rxns_exp, rxns_exp_f, rxns_exp_lb, rxns_exp_ub, strict=False):
            if rxn_id == "BIOMASS" and model_m.bmRxn is not None:
                rxn_id = model_m.bmRxn
            if rxn_id not in model_m.rxns:
                continue
            rxn_num_exp.append(model_m.reaction_index(rxn_id))
            rxns_exp_filtered.append(rxn_id)
            mean_f.append(float(mean_val))
            lb_f.append(float(lb_val))
            ub_f.append(float(ub_val))
        rxn_num_exp_arr = np.asarray(rxn_num_exp, dtype=int)
        rxns_exp_mean = np.asarray(mean_f, dtype=float)
        rxns_exp_lb_arr = np.asarray(lb_f, dtype=float)
        rxns_exp_ub_arr = np.asarray(ub_f, dtype=float)

        bm_constr_flag = False
        for rxn_id, lb_val, ub_val in zip(rxns_exp_filtered, rxns_exp_lb_arr, rxns_exp_ub_arr, strict=False):
            rxn_idx = model_m.reaction_index(rxn_id)
            rxn_name = model_m.rxnNames[rxn_idx].lower() if rxn_idx < len(model_m.rxnNames) else ""
            if "exchange" in rxn_name or rxn_id == model_m.bmRxn:
                model_m = change_rxn_bounds(model_m, rxn_id, lb_val, "l")
                model_m = change_rxn_bounds(model_m, rxn_id, ub_val, "u")
                if rxn_id == model_m.bmRxn:
                    bm_constr_flag = True

        H_diag = np.zeros(model_m.n_rxns, dtype=float)
        H_diag[rxn_num_exp_arr] = 1.0
        target = np.zeros(model_m.n_rxns, dtype=float)
        target[rxn_num_exp_arr] = rxns_exp_mean
        fit = minimize_quadratic_with_linear_constraints(
            H_diag,
            target,
            A_eq=model_m.S,
            b_eq=np.zeros(model_m.n_mets),
            lb=model_m.lb,
            ub=model_m.ub,
        )
        res["sol_minFluxDiff"] = fit
        res["objval_fac"] = fit.objective_value / max(flux_fac, TOL)
        rxn_flux_exp = fit.x[rxn_num_exp_arr]
        model_m = change_rxn_bounds(model_m, rxns_exp_filtered, rxn_flux_exp, "b")

        if not bm_constr_flag:
            model_growth = change_objective(model_m, model_m.bmRxn)
            growth_sol = optimize_cb_model(model_growth, sense="max")
            if growth_sol.success:
                bm_flux = growth_sol.x[model_growth.reaction_index(model_growth.bmRxn)]
                model_m = change_rxn_bounds(model_m, model_m.bmRxn, bm_flux, "l")

        rxns_gene = np.zeros(model_m.n_rxns, dtype=float)
        if model_m.rxnGeneMat is not None:
            rxns_gene[np.where(np.asarray(model_m.rxnGeneMat.getnnz(axis=1)).reshape(-1) > 0)[0]] = 1.0
        n = model_m.n_rxns
        A_eq = sparse.vstack(
            [
                sparse.hstack([model_m.S, sparse.csr_matrix((model_m.n_mets, n))], format="csr"),
            ],
            format="csr",
        )
        b_eq = np.zeros(model_m.n_mets, dtype=float)
        A_ub = sparse.vstack(
            [
                sparse.hstack([sparse.eye(n, format="csr"), -sparse.eye(n, format="csr")], format="csr"),
                sparse.hstack([-sparse.eye(n, format="csr"), -sparse.eye(n, format="csr")], format="csr"),
            ],
            format="csr",
        )
        b_ub = np.zeros(2 * n, dtype=float)
        c = np.concatenate([np.zeros(n), rxns_gene])
        lb_all = np.concatenate([model_m.lb, np.zeros(n)])
        ub_all = np.concatenate([model_m.ub, np.full(n, np.inf)])
        sol2 = solve_lp(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, lb=lb_all, ub=ub_all, sense="min")
        res["sol_minFluxNorm"] = sol2
        fd_ref = sol2.x[:n] if sol2.success else np.zeros(n, dtype=float)
        model.fd_ref = fd_ref
        res["expFlux_mean"] = rxns_exp_mean
        res["simFlux"] = fd_ref[rxn_num_exp_arr]
        with np.errstate(divide="ignore", invalid="ignore"):
            res["diffFlux"] = np.where(abs(rxns_exp_mean) > TOL, (rxns_exp_mean - res["simFlux"]) / rxns_exp_mean, 0.0)
        res["expRxn"] = rxns_exp_filtered
    else:
        raise ValueError("src_flag must be 0 or 1")

    if model_i is not None:
        fd_ref_i = fd_rev2irr(model, model_i, model.fd_ref)
        model_i.fd_ref = fd_ref_i
    return model, model_i, res


if __name__ == "__main__":
    print("model_utils.py imported successfully!")
