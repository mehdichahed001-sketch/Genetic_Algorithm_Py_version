from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse

# Add the parent directory to sys.path when run directly
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gamo.core.gamo_types import EncodingInfo, StoichiometricModel, TargetSet


GENE_TOKEN_RE = re.compile(r"\(|\)|\band\b|\bor\b|[^\s()]+")


class _RuleParser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> Any:
        if not self.tokens:
            return None
        node = self._parse_or()
        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected tokens remaining in gene rule: {self.tokens[self.pos:]}")
        return node

    def _parse_or(self) -> Any:
        node = self._parse_and()
        while self._peek() == "or":
            self.pos += 1
            rhs = self._parse_and()
            node = ("or", node, rhs)
        return node

    def _parse_and(self) -> Any:
        node = self._parse_atom()
        while self._peek() == "and":
            self.pos += 1
            rhs = self._parse_atom()
            node = ("and", node, rhs)
        return node

    def _parse_atom(self) -> Any:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of gene rule")
        if tok == "(":
            self.pos += 1
            node = self._parse_or()
            if self._peek() != ")":
                raise ValueError("Unbalanced parentheses in gene rule")
            self.pos += 1
            return node
        if tok in {"and", "or", ")"}:
            raise ValueError(f"Unexpected token in gene rule: {tok!r}")
        self.pos += 1
        return ("gene", tok)

    def _peek(self) -> str | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]


def _tokenize_gene_rule(rule: str) -> list[str]:
    if not rule or not rule.strip():
        return []
    return [tok.strip() for tok in GENE_TOKEN_RE.findall(rule) if tok.strip()]


def _parse_gene_rule(rule: str) -> Any:
    tokens = _tokenize_gene_rule(rule)
    if not tokens:
        return None
    return _RuleParser(tokens).parse()


def _eval_rule_bool(node: Any, values: Mapping[str, bool]) -> bool:
    if node is None:
        return True
    kind = node[0]
    if kind == "gene":
        return bool(values.get(node[1], True))
    if kind == "and":
        return _eval_rule_bool(node[1], values) and _eval_rule_bool(node[2], values)
    if kind == "or":
        return _eval_rule_bool(node[1], values) or _eval_rule_bool(node[2], values)
    raise ValueError(f"Unknown rule node: {kind!r}")


def _eval_rule_fraction(node: Any, values: Mapping[str, float]) -> float:
    if node is None:
        return 1.0
    kind = node[0]
    if kind == "gene":
        return float(values.get(node[1], 1.0))
    if kind == "and":
        return min(_eval_rule_fraction(node[1], values), _eval_rule_fraction(node[2], values))
    if kind == "or":
        return _eval_rule_fraction(node[1], values) * _eval_rule_fraction(node[2], values)
    raise ValueError(f"Unknown rule node: {kind!r}")


def _collect_rule_genes(node: Any) -> list[str]:
    if node is None:
        return []
    kind = node[0]
    if kind == "gene":
        return [str(node[1])]
    if kind in {"and", "or"}:
        return _collect_rule_genes(node[1]) + _collect_rule_genes(node[2])
    return []


def _to_target_set(targets: TargetSet | Mapping[str, Any]) -> TargetSet:
    if isinstance(targets, TargetSet):
        return targets
    data = dict(targets)
    return TargetSet(**data)


def _dense_target_map(target_map: np.ndarray | sparse.spmatrix | None, nt: int) -> np.ndarray:
    if target_map is None:
        return np.eye(nt, dtype=int)
    if sparse.issparse(target_map) or isinstance(target_map, sparse.sparray):
        return np.asarray(target_map.toarray(), dtype=int)
    arr = np.asarray(target_map, dtype=int)
    if arr.size == 0:
        return np.eye(nt, dtype=int)
    return arr


def _representative_midpoints(encode_vec: np.ndarray, num_targets: int) -> np.ndarray:
    mid = np.zeros(num_targets, dtype=int)
    for target in range(num_targets):
        pos = np.flatnonzero(encode_vec == target)
        if pos.size == 0:
            raise ValueError(f"Target {target} has no representation in the encoding vector")
        mid[target] = int(np.round((pos[0] + pos[-1]) / 2.0))
    return mid


def _binary_from_int(values: np.ndarray, num_bits: int) -> np.ndarray:
    if num_bits <= 0:
        return np.zeros((values.shape[0], 0), dtype=int)
    bits = ((values[:, None] >> np.arange(num_bits, dtype=int)) & 1).astype(int)
    return bits


def _int_from_binary(bits: np.ndarray) -> np.ndarray:
    if bits.size == 0:
        return np.zeros(bits.shape[0], dtype=int)
    weights = (1 << np.arange(bits.shape[1], dtype=int)).astype(int)
    return (bits * weights).sum(axis=1).astype(int)


def _canonicalize_population(pop: np.ndarray, k: int, k_hri: int) -> np.ndarray:
    pop = np.asarray(pop, dtype=int)
    if pop.ndim == 1:
        pop = pop.reshape(1, -1)
    out = pop.copy()
    if k > 1:
        out[:, :k] = np.sort(out[:, :k], axis=1)
    if k_hri > 1:
        out[:, k : (k + k_hri)] = np.sort(out[:, k : (k + k_hri)], axis=1)
    return out


def encode(
    pop: np.ndarray,
    num_bits: int,
    num_bits_hri: int,
    encode_vec_mid_pos: np.ndarray,
    k: int,
    k_hri: int,
) -> np.ndarray:
    pop = np.asarray(pop, dtype=int)
    if pop.ndim == 1:
        pop = pop.reshape(1, -1)
    npop = pop.shape[0]
    total_bits = (k * num_bits) + (k_hri * num_bits_hri)
    pop_bin = np.zeros((npop, total_bits), dtype=int)
    pop_discr = encode_vec_mid_pos[pop]
    num_idx = (1 << num_bits) if num_bits > 0 else 0

    pos = 0
    for i in range(k):
        segment = _binary_from_int(pop_discr[:, i], num_bits)
        pop_bin[:, pos : pos + num_bits] = segment
        pos += num_bits
    for i in range(k, k + k_hri):
        segment = _binary_from_int(pop_discr[:, i] - num_idx, num_bits_hri)
        pop_bin[:, pos : pos + num_bits_hri] = segment
        pos += num_bits_hri
    return pop_bin


def decode(pop_bin: np.ndarray, enc: EncodingInfo | Mapping[str, Any]) -> np.ndarray:
    if not isinstance(enc, EncodingInfo):
        enc = EncodingInfo(**dict(enc))
    pop_bin = np.asarray(pop_bin, dtype=int)
    if pop_bin.ndim == 1:
        pop_bin = pop_bin.reshape(1, -1)
    npop = pop_bin.shape[0]
    pop = np.zeros((npop, enc.K_tot), dtype=int)
    num_idx = (1 << enc.numBits) if enc.numBits > 0 else 0
    pos = 0
    for i in range(enc.K):
        pop[:, i] = _int_from_binary(pop_bin[:, pos : pos + enc.numBits])
        pos += enc.numBits
    for i in range(enc.K, enc.K_tot):
        pop[:, i] = _int_from_binary(pop_bin[:, pos : pos + enc.numBits_hri]) + num_idx
        pos += enc.numBits_hri
    pop = enc.encodeVec[np.clip(pop, 0, len(enc.encodeVec) - 1)]
    return _canonicalize_population(pop, enc.K, enc.K_hri)


def write_log_gene_rules(
    model: StoichiometricModel,
    targets: TargetSet,
    opt_ga: Mapping[str, Any] | None = None,
) -> TargetSet:
    cache = dict(targets.gene_rule_cache)
    reaction_targets = np.asarray(targets.rxnNum, dtype=int).reshape(-1)
    kd_flags = np.asarray(targets.KDID if targets.KDID is not None else np.zeros_like(reaction_targets), dtype=int).reshape(-1)

    expr_trees: list[Any] = []
    rule_genes: list[list[str]] = []
    for rxn_idx in reaction_targets.tolist():
        rule = model.grRules[int(rxn_idx)] if int(rxn_idx) < len(model.grRules) else ""
        expr = _parse_gene_rule(rule)
        expr_trees.append(expr)
        rule_genes.append(sorted(set(_collect_rule_genes(expr))))

    base_gene_names = cache.get("base_gene_names", list(targets.genes))
    is_kd_target = np.asarray(cache.get("is_kd_target", np.zeros(len(base_gene_names), dtype=bool)), dtype=bool)
    if is_kd_target.size == 0 and targets.bound_gene is not None:
        is_kd_target = np.any(np.asarray(targets.bound_gene, dtype=float) != 0.0, axis=1)
    if is_kd_target.size == 0:
        is_kd_target = np.zeros(len(base_gene_names), dtype=bool)

    ub_fraction = np.ones(len(base_gene_names), dtype=float)
    if targets.bound_gene is not None and len(targets.bound_gene):
        bound_gene = np.asarray(targets.bound_gene, dtype=float)
        ub_fraction = np.where(bound_gene[:, 1] < 0.0, -bound_gene[:, 0], bound_gene[:, 1])
        ub_fraction[~np.isfinite(ub_fraction)] = 1.0
        ub_fraction = np.clip(ub_fraction, 0.0, None)

    gene_model_index = np.asarray(cache.get("gene_model_index", np.arange(len(base_gene_names))), dtype=int)
    gene_display_names = list(cache.get("gene_display_names", base_gene_names))

    cache.update(
        {
            "expr_trees": expr_trees,
            "rule_genes": rule_genes,
            "reaction_targets": reaction_targets.copy(),
            "kd_flags": kd_flags.copy(),
            "base_gene_names": list(base_gene_names),
            "gene_display_names": gene_display_names,
            "is_kd_target": is_kd_target,
            "ub_fraction": ub_fraction,
            "gene_model_index": gene_model_index,
        }
    )
    targets.gene_rule_cache = cache
    return targets


def write_rxn_rules(opt_ga: Mapping[str, Any] | None = None) -> None:
    return None


def _reaction_bound_to_irreversible(targets: TargetSet, rxn_idx: int, bound: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    if targets.rev2irrev is None:
        raise ValueError("Irreversible target map not initialized")
    irr_idx = np.asarray(targets.rev2irrev[int(rxn_idx)], dtype=int).reshape(-1)
    lb = np.asarray(bound, dtype=float)[0]
    ub = np.asarray(bound, dtype=float)[1]
    irr_bounds = np.zeros((irr_idx.size, 2), dtype=float)
    if irr_idx.size == 0:
        return irr_idx, irr_bounds
    if irr_idx.size == 1:
        irr_bounds[0, :] = [lb, ub]
        return irr_idx, irr_bounds

    if ub <= 0.0:
        irr_bounds[0, :] = [0.0, 0.0]
        irr_bounds[1, :] = [max(0.0, -ub), max(0.0, -lb)]
    elif lb >= 0.0:
        irr_bounds[0, :] = [lb, ub]
        irr_bounds[1, :] = [0.0, 0.0]
    else:
        irr_bounds[0, :] = [0.0, max(0.0, ub)]
        irr_bounds[1, :] = [0.0, max(0.0, -lb)]
    return irr_idx, irr_bounds


def evaluate_targets(gene_targets: Sequence[int] | np.ndarray, targets: TargetSet | Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets = _to_target_set(targets)
    gene_targets_arr = np.asarray(gene_targets, dtype=int).reshape(-1)

    gene_mode = bool(targets.gene_rule_cache) and targets.geneNum is not None and len(targets.gene_rule_cache.get("reaction_targets", [])) > 0
    if not gene_mode:
        if gene_targets_arr.size == 0:
            empty = np.array([], dtype=int)
            empty_bounds = np.zeros((0, 2), dtype=float)
            return empty, empty, empty_bounds, empty_bounds
        target_rxn_num = np.asarray(targets.rxnNum, dtype=int)[gene_targets_arr]
        target_rxn_bounds = np.asarray(targets.bound, dtype=float)[gene_targets_arr, :]
        target_rxn_num_i_parts: list[np.ndarray] = []
        target_rxn_bounds_i_parts: list[np.ndarray] = []
        if targets.rxnNum_i is not None and len(targets.rxnNum_i):
            for idx in gene_targets_arr.tolist():
                rxn_idx_i = np.asarray(targets.rxnNum_i[idx], dtype=int).reshape(-1)
                target_rxn_num_i_parts.append(rxn_idx_i)
                if targets.bound_i:
                    lb_i = np.asarray(targets.bound_i[idx][0], dtype=float).reshape(-1)
                    ub_i = np.asarray(targets.bound_i[idx][1], dtype=float).reshape(-1)
                    target_rxn_bounds_i_parts.append(np.column_stack([lb_i, ub_i]))
        target_rxn_num_i = np.concatenate(target_rxn_num_i_parts) if target_rxn_num_i_parts else np.array([], dtype=int)
        target_rxn_bounds_i = np.vstack(target_rxn_bounds_i_parts) if target_rxn_bounds_i_parts else np.zeros((0, 2), dtype=float)
        return target_rxn_num, target_rxn_num_i, target_rxn_bounds, target_rxn_bounds_i

    cache = targets.gene_rule_cache
    nt = int(targets.Nt)
    deletion_targets = gene_targets_arr[gene_targets_arr < nt]
    insertion_targets = gene_targets_arr[gene_targets_arr >= nt]

    base_gene_names: list[str] = list(cache["base_gene_names"])
    is_kd_target = np.asarray(cache["is_kd_target"], dtype=bool)
    ub_fraction = np.asarray(cache["ub_fraction"], dtype=float)

    ko_values: dict[str, bool] = {}
    kd_values: dict[str, bool] = {}
    kd_ub: dict[str, float] = {}
    for target_idx in deletion_targets.tolist():
        gene_name = base_gene_names[target_idx]
        if is_kd_target[target_idx]:
            kd_values[gene_name] = False
            kd_ub[gene_name] = min(kd_ub.get(gene_name, 1.0), float(ub_fraction[target_idx]))
        else:
            ko_values[gene_name] = False

    reaction_targets = np.asarray(cache["reaction_targets"], dtype=int)
    kd_flags = np.asarray(cache["kd_flags"], dtype=int)
    expr_trees: list[Any] = list(cache["expr_trees"])

    target_rxn_num_list: list[int] = []
    target_rxn_bounds_list: list[np.ndarray] = []
    target_rxn_num_i_list: list[np.ndarray] = []
    target_rxn_bounds_i_list: list[np.ndarray] = []

    fd_ref = np.asarray(targets.fd_ref if targets.fd_ref is not None else np.zeros(reaction_targets.size), dtype=float)

    for pos, rxn_idx in enumerate(reaction_targets.tolist()):
        expr = expr_trees[pos]
        ko_ok = _eval_rule_bool(expr, ko_values)
        if kd_flags[pos]:
            kd_ok = _eval_rule_bool(expr, kd_values)
            active = ko_ok and (not kd_ok)
            if not active:
                continue
            frac = float(_eval_rule_fraction(expr, kd_ub))
            frac = max(frac, 0.0)
            ref_flux = float(fd_ref[rxn_idx]) if rxn_idx < fd_ref.size else 0.0
            if ref_flux >= 0.0:
                bound = np.array([0.0, ref_flux * frac], dtype=float)
            else:
                bound = np.array([ref_flux * frac, 0.0], dtype=float)
        else:
            active = not ko_ok
            if not active:
                continue
            bound = np.array([0.0, 0.0], dtype=float)

        target_rxn_num_list.append(rxn_idx)
        target_rxn_bounds_list.append(bound)
        irr_idx, irr_bounds = _reaction_bound_to_irreversible(targets, rxn_idx, bound)
        target_rxn_num_i_list.append(irr_idx)
        target_rxn_bounds_i_list.append(irr_bounds)

    for target_idx in insertion_targets.tolist():
        actual_idx = int(target_idx + targets.shift)
        rxn_idx = int(np.asarray(targets.rxnNum, dtype=int)[actual_idx])
        bound = np.asarray(targets.bound, dtype=float)[actual_idx, :]
        target_rxn_num_list.append(rxn_idx)
        target_rxn_bounds_list.append(bound)
        irr_idx, irr_bounds = _reaction_bound_to_irreversible(targets, rxn_idx, bound)
        target_rxn_num_i_list.append(irr_idx)
        target_rxn_bounds_i_list.append(irr_bounds)

    if target_rxn_num_list:
        target_rxn_num = np.asarray(target_rxn_num_list, dtype=int)
        target_rxn_bounds = np.vstack(target_rxn_bounds_list).astype(float)
        target_rxn_num_i = np.concatenate(target_rxn_num_i_list).astype(int)
        target_rxn_bounds_i = np.vstack(target_rxn_bounds_i_list).astype(float)
    else:
        target_rxn_num = np.array([], dtype=int)
        target_rxn_bounds = np.zeros((0, 2), dtype=float)
        target_rxn_num_i = np.array([], dtype=int)
        target_rxn_bounds_i = np.zeros((0, 2), dtype=float)
    return target_rxn_num, target_rxn_num_i, target_rxn_bounds, target_rxn_bounds_i


def translate_pop(
    model: StoichiometricModel,
    model_c: StoichiometricModel,
    indv: Sequence[int] | np.ndarray,
    optFocus: str,
    targets: TargetSet | Mapping[str, Any],
    K: int,
    K_hri: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    targets = _to_target_set(targets)
    indv = np.asarray(indv, dtype=int).reshape(-1)
    indv_del = indv[:K]
    indv_ins = indv[K : (K + K_hri)] if K_hri > 0 else np.array([], dtype=int)

    gene_ko: list[int] = []
    gene_kd: list[int] = []
    ko: list[int] = []
    kd: list[int] = []
    ins: list[int] = []

    if optFocus == "gene":
        target_rxn_num, _, target_rxn_bounds, _ = evaluate_targets(indv, targets)
        cache = targets.gene_rule_cache
        base_gene_names: list[str] = list(cache.get("base_gene_names", targets.genes))
        is_kd_target = np.asarray(cache.get("is_kd_target", np.zeros(len(base_gene_names), dtype=bool)), dtype=bool)
        gene_model_index = np.asarray(cache.get("gene_model_index", np.arange(len(base_gene_names))), dtype=int)
        for idx in indv_del.tolist():
            if idx >= len(base_gene_names):
                continue
            gene_idx = int(gene_model_index[idx])
            if is_kd_target[idx]:
                gene_kd.append(gene_idx)
            else:
                gene_ko.append(gene_idx)

        for rxn_idx, bound in zip(target_rxn_num[: max(0, len(target_rxn_num) - K_hri)], target_rxn_bounds[: max(0, len(target_rxn_num) - K_hri)], strict=False):
            orig_idx = int(model_c.comprMapVec[int(rxn_idx)]) if model_c.comprMapVec is not None and int(rxn_idx) < len(model_c.comprMapVec) else int(rxn_idx)
            if np.allclose(bound, 0.0):
                ko.append(orig_idx)
            else:
                kd.append(orig_idx)
        for idx in indv_ins.tolist():
            actual_idx = int(idx + targets.shift)
            if actual_idx >= len(targets.rxnNum):
                continue
            rxn_idx = int(np.asarray(targets.rxnNum, dtype=int)[actual_idx])
            orig_idx = int(model_c.comprMapVec[int(rxn_idx)]) if model_c.comprMapVec is not None and int(rxn_idx) < len(model_c.comprMapVec) else int(rxn_idx)
            ins.append(orig_idx)
    else:
        for idx in indv_del.tolist():
            rxn_idx = int(np.asarray(targets.rxnNum, dtype=int)[idx])
            bound = np.asarray(targets.bound, dtype=float)[idx, :]
            orig_idx = int(model_c.comprMapVec[int(rxn_idx)]) if model_c.comprMapVec is not None and int(rxn_idx) < len(model_c.comprMapVec) else int(rxn_idx)
            rxn_gene_mat = sparse.csr_matrix(model_c.rxnGeneMat) if model_c.rxnGeneMat is not None else sparse.csr_matrix((model_c.n_rxns, len(model.genes)))
            gene_targets_num = np.where(rxn_gene_mat[int(rxn_idx), :].toarray().reshape(-1) > 0)[0]
            if gene_targets_num.size:
                gene_names = [model_c.genes[g] for g in gene_targets_num if g < len(model_c.genes)]
                gene_model_idx = [model.genes.index(g) for g in gene_names if g in model.genes]
                if np.allclose(bound, 0.0):
                    gene_ko.extend(gene_model_idx)
                else:
                    gene_kd.extend(gene_model_idx)
            if np.allclose(bound, 0.0):
                ko.append(orig_idx)
            else:
                kd.append(orig_idx)
        for idx in indv_ins.tolist():
            rxn_idx = int(np.asarray(targets.rxnNum, dtype=int)[idx])
            orig_idx = int(model_c.comprMapVec[int(rxn_idx)]) if model_c.comprMapVec is not None and int(rxn_idx) < len(model_c.comprMapVec) else int(rxn_idx)
            ins.append(orig_idx)

    return (
        np.asarray(sorted(set(ko)), dtype=int),
        np.asarray(sorted(set(kd)), dtype=int),
        np.asarray(sorted(set(gene_ko)), dtype=int),
        np.asarray(sorted(set(gene_kd)), dtype=int),
        np.asarray(sorted(set(ins)), dtype=int),
    )


def initialize_population(
    model: StoichiometricModel,
    targets: TargetSet | Mapping[str, Any],
    pop_size: int,
    b: float,
    K: int,
    K_hri: int,
    opt_focus: str,
    threads: int,
    init_pop_type: int,
    opt_ga: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, EncodingInfo, TargetSet]:
    opt_ga = dict(opt_ga or {})
    elite = int(opt_ga.get("elite", 2))
    targets = _to_target_set(targets)

    K_tot = int(K + K_hri)
    Nt = int(len(np.asarray(targets.rxnNum, dtype=int)))
    Nt_hri = int(targets.Nt_hri)
    Np = int(pop_size * threads)
    crit_sample_rate = 100
    discr_fac = 50

    if opt_focus == "gene":
        rxn_gene_mat = sparse.csr_matrix(model.rxnGeneMat)
        target_rxn_num = np.asarray(targets.rxnNum, dtype=int)
        target_bounds = np.asarray(targets.bound, dtype=float)

        g_targets_ko: list[int] = []
        kd_gene_meta: list[tuple[int, np.ndarray]] = []
        for rxn_idx, bound in zip(target_rxn_num.tolist(), target_bounds, strict=False):
            genes_for_rxn = np.where(rxn_gene_mat[int(rxn_idx), :].toarray().reshape(-1) > 0)[0]
            if genes_for_rxn.size == 0:
                continue
            if np.any(np.abs(bound) > 0.0):
                ref_flux = float(model.fd_ref[int(rxn_idx)]) if model.fd_ref is not None and int(rxn_idx) < len(model.fd_ref) and abs(model.fd_ref[int(rxn_idx)]) > 1e-12 else 1.0
                rel_bound = np.asarray(bound, dtype=float) / ref_flux
                for gene_idx in genes_for_rxn.tolist():
                    kd_gene_meta.append((int(gene_idx), rel_bound.copy()))
            else:
                g_targets_ko.extend(int(g) for g in genes_for_rxn.tolist())
        g_targets_ko = sorted(set(g_targets_ko))

        ko_genes = [model.genes[g] for g in g_targets_ko]
        kd_unique_names: list[str] = []
        base_gene_names: list[str] = []
        gene_model_index: list[int] = []
        is_kd_target: list[bool] = []
        bound_gene_rows: list[np.ndarray] = []

        for gene_idx, gene_name in zip(g_targets_ko, ko_genes, strict=False):
            kd_unique_names.append(gene_name)
            base_gene_names.append(gene_name)
            gene_model_index.append(int(gene_idx))
            is_kd_target.append(False)
            bound_gene_rows.append(np.array([0.0, 0.0], dtype=float))

        for kd_pos, (gene_idx, rel_bound) in enumerate(kd_gene_meta):
            gene_name = model.genes[int(gene_idx)]
            unique_name = f"{gene_name}__KD__{kd_pos:04d}"
            kd_unique_names.append(unique_name)
            base_gene_names.append(gene_name)
            gene_model_index.append(int(gene_idx))
            is_kd_target.append(True)
            bound_gene_rows.append(np.asarray(rel_bound, dtype=float))

        if g_targets_ko:
            r_targets_ko = np.unique(np.where(rxn_gene_mat[:, g_targets_ko].toarray() > 0)[0]).astype(int)
        else:
            r_targets_ko = np.array([], dtype=int)
        if kd_gene_meta:
            kd_gene_indices = sorted({idx for idx, _ in kd_gene_meta})
            r_targets_kd = np.unique(np.where(rxn_gene_mat[:, kd_gene_indices].toarray() > 0)[0]).astype(int)
        else:
            r_targets_kd = np.array([], dtype=int)

        targets.rxnNum = np.concatenate([r_targets_ko, r_targets_kd]).astype(int)
        targets.KDID = np.concatenate([np.zeros(r_targets_ko.size, dtype=int), np.ones(r_targets_kd.size, dtype=int)])
        targets.geneNum = np.asarray(gene_model_index, dtype=int)
        targets.bound = np.zeros((targets.rxnNum.size, 2), dtype=float)
        targets.genes = list(kd_unique_names)
        targets.bound_gene = np.vstack(bound_gene_rows) if bound_gene_rows else np.zeros((0, 2), dtype=float)
        targets.lb_gene_rxn_mat = np.full((len(kd_unique_names), targets.rxnNum.size), np.nan, dtype=float)
        targets.ub_gene_rxn_mat = np.full((len(kd_unique_names), targets.rxnNum.size), np.nan, dtype=float)
        targets.Nt = len(kd_unique_names)
        targets.map = np.eye(targets.Nt, dtype=int)
        targets.score = np.ones(targets.Nt, dtype=float) / max(targets.Nt, 1)
        targets.gene_rule_cache = {
            "base_gene_names": base_gene_names,
            "gene_display_names": [model.genes[i] for i in gene_model_index],
            "gene_model_index": np.asarray(gene_model_index, dtype=int),
            "is_kd_target": np.asarray(is_kd_target, dtype=bool),
        }
        targets = write_log_gene_rules(model, targets, opt_ga)
        Nt = int(targets.Nt)
        score = np.asarray(targets.score, dtype=float)
        target_map = _dense_target_map(targets.map, Nt)
    else:
        score = np.asarray(targets.score, dtype=float).reshape(-1)
        target_map = _dense_target_map(targets.map, Nt)
        targets.Nt = Nt
        write_rxn_rules(opt_ga)

    targets.rxnNum_KO = np.asarray(targets.rxnNum, dtype=int).copy()
    targets.shift = int(len(np.asarray(targets.rxnNum, dtype=int)) - Nt)

    if targets.rxnNum_hri.size:
        targets.rxnNum = np.concatenate([np.asarray(targets.rxnNum, dtype=int), np.asarray(targets.rxnNum_hri, dtype=int)]).astype(int)
        targets.bound = np.vstack([np.asarray(targets.bound, dtype=float), np.asarray(targets.bound_hri, dtype=float)])
        targets.score = np.concatenate([np.asarray(targets.score, dtype=float), np.asarray(targets.score_hri, dtype=float)]).astype(float)

    targets.Nt = Nt
    targets.Nt_tot = Nt + Nt_hri
    Nt_tot = targets.Nt_tot

    max_comb = 0
    for i in range(1, K + 1):
        if Nt >= i:
            max_comb += math.comb(Nt, i)
    if max_comb and max_comb < Np:
        pop_size = max(max_comb // max(threads, 1), elite + 1)
        Np = pop_size * threads

    if score.size == 0:
        raise ValueError("Target space is empty after preprocessing")

    order = np.argsort(score)
    if np.any(np.abs(score - score[0]) > 1e-4):
        a = 2.0 - float(b)
        ranks = np.arange(1, Nt + 1, dtype=float)
        W = (a + ((ranks / Nt) * (float(b) - a))) / (Nt + 1.0)
        probs = np.zeros(Nt, dtype=float)
        probs[order] = W / np.sum(W)
    else:
        probs = np.ones(Nt, dtype=float) / Nt

    pop = np.zeros((Np, K_tot), dtype=int)
    pop_tbin = np.zeros((Np, Nt_tot), dtype=int)
    existing: set[tuple[int, ...]] = set()

    def is_valid(chrom: np.ndarray) -> bool:
        chrom = _canonicalize_population(chrom.reshape(1, -1), K, K_hri).reshape(-1)
        flag = np.zeros(Nt_tot, dtype=int)
        flag[chrom] = 1
        rxnflag = flag[:Nt] @ target_map
        if np.any(rxnflag > 1):
            return False
        key = tuple(np.flatnonzero(flag).tolist())
        if key in existing:
            return False
        return True

    if init_pop_type == 0:
        for i in range(Np):
            sample_rate = 0
            while True:
                sample_rate += 1
                if K > 0:
                    if Nt >= K:
                        del_targets = np.random.choice(Nt, size=K, replace=False, p=probs)
                    else:
                        del_targets = np.random.choice(Nt, size=K, replace=True, p=probs)
                else:
                    del_targets = np.array([], dtype=int)
                if K_hri > 0:
                    ins_targets = Nt + np.random.choice(max(Nt_hri, 1), size=K_hri, replace=Nt_hri < K_hri)
                else:
                    ins_targets = np.array([], dtype=int)
                chrom = np.concatenate([del_targets, ins_targets]).astype(int)
                chrom = _canonicalize_population(chrom.reshape(1, -1), K, K_hri).reshape(-1)
                if is_valid(chrom) or sample_rate >= crit_sample_rate:
                    pop[i, :] = chrom
                    flag = np.zeros(Nt_tot, dtype=int)
                    flag[chrom] = 1
                    pop_tbin[i, :] = flag
                    existing.add(tuple(np.flatnonzero(flag).tolist()))
                    break
    elif init_pop_type == 1:
        rand_target_num = np.random.randint(0, Nt, size=Np) if Nt else np.zeros(Np, dtype=int)
        rand_insert_num = Nt + np.random.randint(0, Nt_hri, size=Np) if Nt_hri else np.zeros(Np, dtype=int)
        for i in range(Np):
            chrom = np.concatenate([
                np.repeat(rand_target_num[i], K),
                np.repeat(rand_insert_num[i], K_hri) if K_hri else np.array([], dtype=int),
            ]).astype(int)
            chrom = _canonicalize_population(chrom.reshape(1, -1), K, K_hri).reshape(-1)
            pop[i, :] = chrom
            flag = np.zeros(Nt_tot, dtype=int)
            flag[np.unique(chrom)] = 1
            pop_tbin[i, :] = flag
    else:
        raise ValueError("init_pop_type must be 0 or 1")

    num_bits = int(round(math.log(max(discr_fac * max(Nt, 1), 1), 2))) if Nt > 0 else 0
    num_bits = max(num_bits, 1 if K > 0 else 0)
    encode_vec = np.ceil((np.arange(1, (1 << num_bits) + 1) / ((1 << num_bits) / max(Nt, 1)))).astype(int) - 1 if num_bits > 0 else np.array([], dtype=int)
    encode_vec = np.clip(encode_vec, 0, max(Nt - 1, 0)) if encode_vec.size else encode_vec
    pos_gene: list[int] = list(range(0, K * num_bits, num_bits)) if num_bits > 0 else []

    if Nt_hri > 0:
        num_bits_hri = int(round(math.log(max(discr_fac * Nt_hri, 1), 2)))
        num_bits_hri = max(num_bits_hri, 1)
        encode_vec_hri = np.ceil((np.arange(1, (1 << num_bits_hri) + 1) / ((1 << num_bits_hri) / Nt_hri))).astype(int) - 1
        encode_vec_hri = np.clip(encode_vec_hri, 0, Nt_hri - 1) + Nt
        encode_vec = np.concatenate([encode_vec, encode_vec_hri]).astype(int)
        pos_gene.extend(list(np.arange(0, K_hri * num_bits_hri, num_bits_hri) + (K * num_bits)))
    else:
        num_bits_hri = 0

    encode_vec_mid_pos = _representative_midpoints(encode_vec, Nt_tot)
    pop_bin = encode(pop, num_bits, num_bits_hri, encode_vec_mid_pos, K, K_hri)
    enc = EncodingInfo(
        encodeVec=np.asarray(encode_vec, dtype=int),
        encodeVec_midPos=np.asarray(encode_vec_mid_pos, dtype=int),
        numBits=int(num_bits),
        numBits_hri=int(num_bits_hri),
        posGene=np.asarray(pos_gene, dtype=int),
        K=int(K),
        K_hri=int(K_hri),
        K_tot=int(K_tot),
        numBits_tot=int((K * num_bits) + (K_hri * num_bits_hri)),
    )
    return pop, pop_tbin, pop_bin, enc, targets


if __name__ == "__main__":
    print("target_utils.py imported successfully!")
