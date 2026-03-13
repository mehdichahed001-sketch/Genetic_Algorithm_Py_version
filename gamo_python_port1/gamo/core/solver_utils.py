from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

# Add the parent directory to sys.path when run directly
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gamo.core.gamo_types import FitFunctionOptions, LinearSolution, StoichiometricModel, as_numpy_int_array

try:
    from highspy import Highs, HighsLp, HighsModelStatus, HighsStatus, MatrixFormat, ObjSense

    _HAS_HIGHSPY = True
except Exception:  # pragma: no cover - optional dependency
    Highs = HighsLp = HighsModelStatus = HighsStatus = MatrixFormat = ObjSense = None  # type: ignore[assignment]
    _HAS_HIGHSPY = False


INF = 1.0e9
TOL = 1.0e-9


@dataclass
class _ReusableEqLPSolver:
    solver: Highs
    n_cols: int
    n_rows: int
    cost: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    sense: ObjSense


_EQ_LP_CACHE: dict[int, _ReusableEqLPSolver] = {}


def _linprog_status_to_matlab_style(status: int, success: bool) -> str:
    if success:
        return "OPTIMAL"
    if status == 2:
        return "INFEASIBLE"
    if status == 3:
        return "UNBOUNDED"
    return "FAILED"


def _normalize_bounds(lb: np.ndarray, ub: np.ndarray) -> list[tuple[float | None, float | None]]:
    bounds: list[tuple[float | None, float | None]] = []
    for lo, hi in zip(lb, ub, strict=False):
        lo_out = None if np.isneginf(lo) or lo <= -INF else float(lo)
        hi_out = None if np.isposinf(hi) or hi >= INF else float(hi)
        bounds.append((lo_out, hi_out))
    return bounds


def solve_lp(
    c: np.ndarray,
    *,
    A_eq: sparse.spmatrix | np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    A_ub: sparse.spmatrix | np.ndarray | None = None,
    b_ub: np.ndarray | None = None,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
    sense: str = "min",
) -> LinearSolution:
    c = np.asarray(c, dtype=float).reshape(-1)
    n = c.size
    if lb is None:
        lb = np.full(n, -np.inf)
    if ub is None:
        ub = np.full(n, np.inf)
    if sense not in {"min", "max"}:
        raise ValueError("sense must be 'min' or 'max'")
    c_eff = c.copy()
    if sense == "max":
        c_eff = -c_eff
    res = linprog(
        c=c_eff,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=_normalize_bounds(np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)),
        method="highs",
    )
    if res.success:
        obj = float(-res.fun if sense == "max" else res.fun)
        x = np.asarray(res.x, dtype=float)
    else:
        obj = float("nan")
        x = np.zeros(n, dtype=float)
    return LinearSolution(
        status=_linprog_status_to_matlab_style(res.status, res.success),
        success=bool(res.success),
        x=x,
        objective_value=obj,
        message=res.message,
    )


def _highs_model_status_to_matlab_style(model_status: HighsModelStatus) -> str:
    if model_status == HighsModelStatus.kOptimal:
        return "OPTIMAL"
    if model_status == HighsModelStatus.kInfeasible:
        return "INFEASIBLE"
    if model_status in (HighsModelStatus.kUnbounded, HighsModelStatus.kUnboundedOrInfeasible):
        return "UNBOUNDED"
    return "FAILED"


def _get_reusable_eq_lp_solver(model: StoichiometricModel, lb: np.ndarray, ub: np.ndarray) -> _ReusableEqLPSolver | None:
    if not _HAS_HIGHSPY:
        return None
    key = id(model)
    cached = _EQ_LP_CACHE.get(key)
    if cached is not None:
        if cached.n_cols == model.n_rxns and cached.n_rows == model.n_mets:
            return cached
        _EQ_LP_CACHE.pop(key, None)

    solver = _build_reusable_highs_solver(model.S, np.zeros(model.n_mets, dtype=float), lb, ub)
    if solver is None:
        return None
    state = _ReusableEqLPSolver(
        solver=solver,
        n_cols=model.n_rxns,
        n_rows=model.n_mets,
        cost=np.zeros(model.n_rxns, dtype=float),
        lb=np.asarray(lb, dtype=float).copy(),
        ub=np.asarray(ub, dtype=float).copy(),
        sense=ObjSense.kMinimize,
    )
    _EQ_LP_CACHE[key] = state
    return state


def _solve_eq_lp_with_reuse(
    model: StoichiometricModel,
    objective: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    sense: str,
) -> LinearSolution | None:
    state = _get_reusable_eq_lp_solver(model, lb, ub)
    if state is None:
        return None
    try:
        obj = np.asarray(objective, dtype=float).reshape(-1)
        lb_arr = np.asarray(lb, dtype=float).reshape(-1)
        ub_arr = np.asarray(ub, dtype=float).reshape(-1)
        if obj.size != state.n_cols or lb_arr.size != state.n_cols or ub_arr.size != state.n_cols:
            return None

        cost_diff = np.flatnonzero(state.cost != obj)
        if cost_diff.size:
            idx = cost_diff.astype(np.int32, copy=False)
            vals = obj[cost_diff].astype(np.float64, copy=False)
            status_cost = state.solver.changeColsCost(int(idx.size), idx, vals)
            if status_cost != HighsStatus.kOk:
                return None
            state.cost[cost_diff] = obj[cost_diff]

        bounds_diff = np.flatnonzero((state.lb != lb_arr) | (state.ub != ub_arr))
        if bounds_diff.size:
            idx_b = bounds_diff.astype(np.int32, copy=False)
            lb_b = lb_arr[bounds_diff].astype(np.float64, copy=False)
            ub_b = ub_arr[bounds_diff].astype(np.float64, copy=False)
            status_bounds = state.solver.changeColsBounds(int(idx_b.size), idx_b, lb_b, ub_b)
            if status_bounds != HighsStatus.kOk:
                return None
            state.lb[bounds_diff] = lb_arr[bounds_diff]
            state.ub[bounds_diff] = ub_arr[bounds_diff]

        sense_target = ObjSense.kMaximize if sense == "max" else ObjSense.kMinimize
        if state.sense != sense_target:
            status_sense = state.solver.changeObjectiveSense(sense_target)
            if status_sense != HighsStatus.kOk:
                return None
            state.sense = sense_target

        run_status = state.solver.run()
        if run_status != HighsStatus.kOk:
            return LinearSolution(status="FAILED", success=False, x=np.zeros(state.n_cols), objective_value=float("nan"), message="HiGHS run failed")
        model_status = state.solver.getModelStatus()
        if model_status == HighsModelStatus.kOptimal:
            x = np.asarray(state.solver.getSolution().col_value, dtype=float)
            obj_val = float(state.solver.getObjectiveValue())
            return LinearSolution(
                status="OPTIMAL",
                success=True,
                x=x,
                objective_value=obj_val,
                message=state.solver.modelStatusToString(model_status),
            )
        return LinearSolution(
            status=_highs_model_status_to_matlab_style(model_status),
            success=False,
            x=np.zeros(state.n_cols, dtype=float),
            objective_value=float("nan"),
            message=state.solver.modelStatusToString(model_status),
        )
    except Exception:
        return None


def _build_reusable_highs_solver(
    A_eq: sparse.spmatrix | np.ndarray,
    b_eq: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> Highs | None:
    if not _HAS_HIGHSPY:
        return None
    try:
        A_csc = sparse.csc_matrix(A_eq)
        n_row, n_col = A_csc.shape
        lp = HighsLp()
        lp.num_col_ = int(n_col)
        lp.num_row_ = int(n_row)
        lp.col_cost_ = np.zeros(n_col, dtype=np.float64)
        lp.col_lower_ = np.asarray(lb, dtype=np.float64).reshape(-1)
        lp.col_upper_ = np.asarray(ub, dtype=np.float64).reshape(-1)
        lp.row_lower_ = np.asarray(b_eq, dtype=np.float64).reshape(-1)
        lp.row_upper_ = np.asarray(b_eq, dtype=np.float64).reshape(-1)
        lp.sense_ = ObjSense.kMinimize
        lp.offset_ = 0.0
        lp.a_matrix_.num_col_ = int(n_col)
        lp.a_matrix_.num_row_ = int(n_row)
        lp.a_matrix_.format_ = MatrixFormat.kColwise
        lp.a_matrix_.start_ = A_csc.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A_csc.indices.astype(np.int32)
        lp.a_matrix_.value_ = A_csc.data.astype(np.float64)

        solver = Highs()
        solver.setOptionValue("output_flag", False)
        solver.setOptionValue("presolve", "on")
        solver.setOptionValue("solver", "simplex")
        solver.setOptionValue("threads", 1)
        status = solver.passModel(lp)
        if status != HighsStatus.kOk:
            return None
        return solver
    except Exception:
        return None


def reaction_extrema(
    model: StoichiometricModel,
    reaction_indices: Sequence[int] | np.ndarray | None = None,
    *,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
    return_flux_vectors: bool = False,
    failure_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if reaction_indices is None:
        selected_indices = np.arange(model.n_rxns, dtype=int)
    else:
        selected_indices = np.asarray(reaction_indices, dtype=int).reshape(-1)
    n_selected = selected_indices.size
    n_rxns = model.n_rxns

    lb_eff = model.lb if lb is None else np.asarray(lb, dtype=float).reshape(-1)
    ub_eff = model.ub if ub is None else np.asarray(ub, dtype=float).reshape(-1)

    min_flux = np.full(n_selected, float(failure_value), dtype=float)
    max_flux = np.full(n_selected, float(failure_value), dtype=float)
    if return_flux_vectors:
        Vmin: np.ndarray | None = np.zeros((n_rxns, n_selected), dtype=float)
        Vmax: np.ndarray | None = np.zeros((n_rxns, n_selected), dtype=float)
    else:
        Vmin = None
        Vmax = None

    b_eq = np.zeros(model.n_mets, dtype=float)
    highs_solver = _build_reusable_highs_solver(model.S, b_eq, lb_eff, ub_eff)
    if highs_solver is not None:
        for col, rxn_num in enumerate(selected_indices):
            rxn_num_i = int(rxn_num)
            status_cost_on = highs_solver.changeColCost(rxn_num_i, 1.0)
            if status_cost_on != HighsStatus.kOk:
                continue

            highs_solver.changeObjectiveSense(ObjSense.kMaximize)
            max_run_status = highs_solver.run()
            if max_run_status == HighsStatus.kOk and highs_solver.getModelStatus() == HighsModelStatus.kOptimal:
                max_flux[col] = float(highs_solver.getObjectiveValue())
                if Vmax is not None:
                    Vmax[:, col] = np.asarray(highs_solver.getSolution().col_value, dtype=float)

            highs_solver.changeObjectiveSense(ObjSense.kMinimize)
            min_run_status = highs_solver.run()
            if min_run_status == HighsStatus.kOk and highs_solver.getModelStatus() == HighsModelStatus.kOptimal:
                min_flux[col] = float(highs_solver.getObjectiveValue())
                if Vmin is not None:
                    Vmin[:, col] = np.asarray(highs_solver.getSolution().col_value, dtype=float)
            highs_solver.changeColCost(rxn_num_i, 0.0)

        return min_flux, max_flux, Vmin, Vmax

    for col, rxn_num in enumerate(selected_indices):
        objective = np.zeros(n_rxns, dtype=float)
        objective[int(rxn_num)] = 1.0
        sol_max = solve_lp(objective, A_eq=model.S, b_eq=b_eq, lb=lb_eff, ub=ub_eff, sense="max")
        if sol_max.success:
            max_flux[col] = float(sol_max.x[int(rxn_num)])
            if Vmax is not None:
                Vmax[:, col] = sol_max.x
        sol_min = solve_lp(objective, A_eq=model.S, b_eq=b_eq, lb=lb_eff, ub=ub_eff, sense="min")
        if sol_min.success:
            min_flux[col] = float(sol_min.x[int(rxn_num)])
            if Vmin is not None:
                Vmin[:, col] = sol_min.x
    return min_flux, max_flux, Vmin, Vmax


def apply_target_bounds(
    model: StoichiometricModel,
    target_indices: Sequence[int] | np.ndarray,
    target_bounds: np.ndarray,
    *,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    lb_eff = model.lb.copy() if lb is None else np.asarray(lb, dtype=float).copy()
    ub_eff = model.ub.copy() if ub is None else np.asarray(ub, dtype=float).copy()
    if len(target_indices) == 0:
        return lb_eff, ub_eff
    target_indices = np.asarray(target_indices, dtype=int)
    target_bounds = np.asarray(target_bounds, dtype=float)
    if target_bounds.ndim == 1:
        target_bounds = target_bounds.reshape(1, 2)
    lb_eff[target_indices] = target_bounds[:, 0]
    ub_eff[target_indices] = target_bounds[:, 1]
    return lb_eff, ub_eff


def optimize_cb_model(
    model: StoichiometricModel,
    sense: str = "max",
    one_norm: bool = False,
    *,
    objective: np.ndarray | None = None,
    lb_override: np.ndarray | None = None,
    ub_override: np.ndarray | None = None,
) -> LinearSolution:
    obj = model.c if objective is None else np.asarray(objective, dtype=float)
    lb = model.lb if lb_override is None else lb_override
    ub = model.ub if ub_override is None else ub_override
    first = _solve_eq_lp_with_reuse(model, obj, lb, ub, sense)
    if first is None:
        first = solve_lp(obj, A_eq=model.S, b_eq=np.zeros(model.n_mets), lb=lb, ub=ub, sense=sense)
    if not first.success or not one_norm:
        return first
    return one_norm_minimization(model, obj, first.objective_value, lb=lb, ub=ub)


def one_norm_minimization(
    model: StoichiometricModel,
    objective_vector: np.ndarray,
    objective_value: float,
    *,
    lb: np.ndarray | None = None,
    ub: np.ndarray | None = None,
) -> LinearSolution:
    n = model.n_rxns
    lb_v = model.lb if lb is None else np.asarray(lb, dtype=float)
    ub_v = model.ub if ub is None else np.asarray(ub, dtype=float)

    A_eq = sparse.vstack(
        [
            sparse.hstack([model.S, sparse.csr_matrix((model.n_mets, n))], format="csr"),
            sparse.hstack([
                sparse.csr_matrix(objective_vector.reshape(1, -1)),
                sparse.csr_matrix((1, n)),
            ], format="csr"),
        ],
        format="csr",
    )
    b_eq = np.concatenate([np.zeros(model.n_mets), np.array([objective_value], dtype=float)])
    A_ub = sparse.vstack(
        [
            sparse.hstack([sparse.eye(n, format="csr"), -sparse.eye(n, format="csr")], format="csr"),
            sparse.hstack([-sparse.eye(n, format="csr"), -sparse.eye(n, format="csr")], format="csr"),
        ],
        format="csr",
    )
    b_ub = np.zeros(2 * n, dtype=float)
    c = np.concatenate([np.zeros(n, dtype=float), np.ones(n, dtype=float)])
    lb_all = np.concatenate([lb_v, np.zeros(n, dtype=float)])
    ub_all = np.concatenate([ub_v, np.full(n, np.inf, dtype=float)])
    sol = solve_lp(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, lb=lb_all, ub=ub_all, sense="min")
    if sol.success:
        return LinearSolution(
            status=sol.status,
            success=True,
            x=sol.x[:n],
            objective_value=float(objective_vector @ sol.x[:n]),
            message=sol.message,
        )
    return LinearSolution(status=sol.status, success=False, x=np.zeros(n), objective_value=float("nan"), message=sol.message)


def init_struct_lp(model: StoichiometricModel) -> tuple[dict, dict, dict]:
    obj_bm = np.zeros(model.n_rxns, dtype=float)
    obj_p = np.zeros(model.n_rxns, dtype=float)
    if model.bmRxnNum is not None:
        obj_bm[model.bmRxnNum] = 1.0
    if model.targetRxnNum is not None:
        obj_p[model.targetRxnNum] = 1.0
    gur_prob = {
        "A_eq": model.S,
        "b_eq": np.zeros(model.n_mets, dtype=float),
        "obj_BM": obj_bm,
        "obj_P": obj_p,
        "lb": model.lb.copy(),
        "ub": model.ub.copy(),
    }
    gur_prob_1norm = {
        "obj": np.ones(model.n_rxns, dtype=float),
        "lb": model.lb.copy(),
        "ub": model.ub.copy(),
    }
    gur_params = {"method": "highs"}
    return gur_prob, gur_prob_1norm, gur_params


def manual_fva(
    model: StoichiometricModel,
    selected_rxns: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if selected_rxns is None:
        selected_indices = np.arange(model.n_rxns, dtype=int)
    else:
        selected_indices = np.array([model.reaction_index(rxn) for rxn in selected_rxns], dtype=int)
    min_flux, max_flux, Vmin, Vmax = reaction_extrema(
        model,
        selected_indices,
        return_flux_vectors=True,
        failure_value=0.0,
    )
    if Vmin is None or Vmax is None:
        raise RuntimeError("manual_fva expected flux vectors from reaction_extrema")
    return min_flux, max_flux, Vmin, Vmax


def manual_fva_bounds(
    model: StoichiometricModel,
    selected_rxns: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if selected_rxns is None:
        selected_indices = np.arange(model.n_rxns, dtype=int)
    else:
        selected_indices = np.array([model.reaction_index(rxn) for rxn in selected_rxns], dtype=int)
    min_flux, max_flux, _, _ = reaction_extrema(
        model,
        selected_indices,
        return_flux_vectors=False,
        failure_value=0.0,
    )
    return min_flux, max_flux


@dataclass
class QuadraticFitResult:
    x: np.ndarray
    success: bool
    message: str
    objective_value: float


def minimize_quadratic_with_linear_constraints(
    H_diag: np.ndarray,
    target: np.ndarray,
    *,
    A_eq: sparse.spmatrix,
    b_eq: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x0: np.ndarray | None = None,
) -> QuadraticFitResult:
    n = lb.size
    H_diag = np.asarray(H_diag, dtype=float)
    target = np.asarray(target, dtype=float)
    if x0 is None:
        feas = solve_lp(np.zeros(n), A_eq=A_eq, b_eq=b_eq, lb=lb, ub=ub, sense="min")
        x0 = feas.x if feas.success else np.clip(np.zeros(n), lb, ub)

    def fun(v: np.ndarray) -> float:
        diff = v - target
        return float(np.sum(H_diag * diff * diff))

    def jac(v: np.ndarray) -> np.ndarray:
        diff = v - target
        return 2.0 * H_diag * diff

    def hessp(_v: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 2.0 * H_diag * p

    constraint = LinearConstraint(A_eq, lb=b_eq, ub=b_eq)
    bounds = Bounds(lb, ub)
    res = minimize(
        fun,
        x0=np.asarray(x0, dtype=float),
        method="trust-constr",
        jac=jac,
        hessp=hessp,
        constraints=[constraint],
        bounds=bounds,
        options={"verbose": 0, "maxiter": 300},
    )
    return QuadraticFitResult(
        x=np.asarray(res.x, dtype=float),
        success=bool(res.success),
        message=res.message,
        objective_value=float(res.fun),
    )


def mimbl(
    model: StoichiometricModel,
    flux_dist_wt: np.ndarray,
    irrev_flag: bool,
    excl_rxns: Iterable[int] | np.ndarray = (),
    *,
    lb_override: np.ndarray | None = None,
    ub_override: np.ndarray | None = None,
):
    if irrev_flag:
        model_irrev = model
        irrev_flux_dist_wt = np.asarray(flux_dist_wt, dtype=float).reshape(-1)
        excl_rxns_arr = as_numpy_int_array(excl_rxns)
        lb_irrev = model_irrev.lb if lb_override is None else np.asarray(lb_override, dtype=float).reshape(-1)
        ub_irrev = model_irrev.ub if ub_override is None else np.asarray(ub_override, dtype=float).reshape(-1)
    else:
        from gamo.core.model_utils import fd_rev2irr, rev2irr

        model_irrev = rev2irr(model)
        irrev_flux_dist_wt = fd_rev2irr(model, model_irrev, np.asarray(flux_dist_wt, dtype=float).reshape(-1))
        excl_rxns_arr = np.array([idx for ridx in as_numpy_int_array(excl_rxns) for idx in model_irrev.rev2irrev[ridx]], dtype=int)
        if lb_override is not None or ub_override is not None:
            raise ValueError("lb_override/ub_override are only supported with irrev_flag=True")
        lb_irrev = model_irrev.lb
        ub_irrev = model_irrev.ub

    n_mets, n_rxns = model_irrev.S.shape
    alpha = abs(model_irrev.S).toarray()
    if excl_rxns_arr.size:
        alpha[:, excl_rxns_arr] = 0.0
    t_wt = alpha @ irrev_flux_dist_wt

    # First optimization: minimize absolute metabolite balance deviations.
    # Variables: [m, v, a_m]
    A_eq_1 = sparse.vstack(
        [
            sparse.hstack([sparse.csr_matrix((n_mets, n_mets)), model_irrev.S, sparse.csr_matrix((n_mets, n_mets))], format="csr"),
            sparse.hstack([sparse.eye(n_mets, format="csr"), sparse.csr_matrix(alpha), sparse.csr_matrix((n_mets, n_mets))], format="csr"),
        ],
        format="csr",
    )
    b_eq_1 = np.concatenate([np.zeros(n_mets), t_wt])
    A_ub_1 = sparse.vstack(
        [
            sparse.hstack([sparse.eye(n_mets, format="csr"), sparse.csr_matrix((n_mets, n_rxns)), -sparse.eye(n_mets, format="csr")], format="csr"),
            sparse.hstack([-sparse.eye(n_mets, format="csr"), sparse.csr_matrix((n_mets, n_rxns)), -sparse.eye(n_mets, format="csr")], format="csr"),
        ],
        format="csr",
    )
    b_ub_1 = np.zeros(2 * n_mets, dtype=float)
    c_1 = np.concatenate([np.zeros(n_mets + n_rxns), np.ones(n_mets)])
    lb_1 = np.concatenate([np.full(n_mets, -np.inf), lb_irrev, np.zeros(n_mets)])
    ub_1 = np.concatenate([np.full(n_mets, np.inf), ub_irrev, np.full(n_mets, np.inf)])
    fo_sol = solve_lp(c_1, A_eq=A_eq_1, b_eq=b_eq_1, A_ub=A_ub_1, b_ub=b_ub_1, lb=lb_1, ub=ub_1, sense="min")
    if not fo_sol.success:
        return {
            "x": np.array([], dtype=float),
            "x_i": np.array([], dtype=float),
            "xWT": model_irrev.mapIrr2Rev @ irrev_flux_dist_wt if model_irrev.mapIrr2Rev is not None else irrev_flux_dist_wt,
            "firstOptSol": fo_sol,
            "secondOptSol": LinearSolution("INFEASIBLE", False, np.array([], dtype=float), float("nan"), "MiMBl first LP infeasible"),
            "rc_so": np.array([], dtype=float),
            "rc_fo": np.array([], dtype=float),
        }
    true_obj = fo_sol.objective_value

    irrev_flux_dist_wt = irrev_flux_dist_wt.copy()
    irrev_flux_dist_wt[abs(irrev_flux_dist_wt) < 1.0e-7] = 0.0
    obj = np.zeros(n_rxns, dtype=float)
    nz = abs(irrev_flux_dist_wt) > 0
    obj[nz] = 1.0 / np.maximum(abs(irrev_flux_dist_wt[nz]), TOL)
    if excl_rxns_arr.size:
        obj[excl_rxns_arr] = 0.0

    # Second optimization: weighted L1 distance in flux space with same metabolite balance objective.
    # Variables: [dv, m, v, a_v, a_m]
    A_eq_2 = sparse.vstack(
        [
            sparse.hstack([
                sparse.csr_matrix((n_mets, n_rxns)),
                sparse.csr_matrix((n_mets, n_mets)),
                model_irrev.S,
                sparse.csr_matrix((n_mets, n_rxns)),
                sparse.csr_matrix((n_mets, n_mets)),
            ], format="csr"),
            sparse.hstack([
                sparse.csr_matrix((1, n_rxns)),
                sparse.csr_matrix((1, n_mets)),
                sparse.csr_matrix((1, n_rxns)),
                sparse.csr_matrix((1, n_rxns)),
                sparse.csr_matrix(np.ones((1, n_mets))),
            ], format="csr"),
            sparse.hstack([
                sparse.eye(n_rxns, format="csr"),
                sparse.csr_matrix((n_rxns, n_mets)),
                sparse.eye(n_rxns, format="csr"),
                sparse.csr_matrix((n_rxns, n_rxns)),
                sparse.csr_matrix((n_rxns, n_mets)),
            ], format="csr"),
            sparse.hstack([
                sparse.csr_matrix((n_mets, n_rxns)),
                sparse.eye(n_mets, format="csr"),
                sparse.csr_matrix(alpha),
                sparse.csr_matrix((n_mets, n_rxns)),
                sparse.csr_matrix((n_mets, n_mets)),
            ], format="csr"),
        ],
        format="csr",
    )
    b_eq_2 = np.concatenate([
        np.zeros(n_mets),
        np.array([true_obj], dtype=float),
        irrev_flux_dist_wt,
        t_wt,
    ])
    A_ub_2 = sparse.vstack(
        [
            sparse.hstack([
                sparse.eye(n_rxns, format="csr"),
                sparse.csr_matrix((n_rxns, n_mets + n_rxns)),
                -sparse.eye(n_rxns, format="csr"),
                sparse.csr_matrix((n_rxns, n_mets)),
            ], format="csr"),
            sparse.hstack([
                -sparse.eye(n_rxns, format="csr"),
                sparse.csr_matrix((n_rxns, n_mets + n_rxns)),
                -sparse.eye(n_rxns, format="csr"),
                sparse.csr_matrix((n_rxns, n_mets)),
            ], format="csr"),
            sparse.hstack([
                sparse.csr_matrix((n_mets, n_rxns)),
                sparse.eye(n_mets, format="csr"),
                sparse.csr_matrix((n_mets, 2 * n_rxns)),
                -sparse.eye(n_mets, format="csr"),
            ], format="csr"),
            sparse.hstack([
                sparse.csr_matrix((n_mets, n_rxns)),
                -sparse.eye(n_mets, format="csr"),
                sparse.csr_matrix((n_mets, 2 * n_rxns)),
                -sparse.eye(n_mets, format="csr"),
            ], format="csr"),
        ],
        format="csr",
    )
    b_ub_2 = np.zeros((2 * n_rxns) + (2 * n_mets), dtype=float)
    c_2 = np.concatenate([np.zeros(n_rxns + n_mets + n_rxns), obj, np.zeros(n_mets)])
    lb_2 = np.concatenate([
        np.full(n_rxns, -np.inf),
        np.full(n_mets, -np.inf),
        lb_irrev,
        np.zeros(n_rxns),
        np.zeros(n_mets),
    ])
    ub_2 = np.concatenate([
        np.full(n_rxns, np.inf),
        np.full(n_mets, np.inf),
        ub_irrev,
        np.full(n_rxns, np.inf),
        np.full(n_mets, np.inf),
    ])
    so_sol = solve_lp(c_2, A_eq=A_eq_2, b_eq=b_eq_2, A_ub=A_ub_2, b_ub=b_ub_2, lb=lb_2, ub=ub_2, sense="min")
    if not so_sol.success:
        x_i = np.array([], dtype=float)
        x = np.array([], dtype=float)
    else:
        start = n_rxns + n_mets
        stop = start + n_rxns
        x_i = so_sol.x[start:stop]
        x = np.asarray(model_irrev.mapIrr2Rev @ x_i).reshape(-1) if model_irrev.mapIrr2Rev is not None else x_i.copy()
    return {
        "x": x,
        "x_i": x_i,
        "xWT": np.asarray(model_irrev.mapIrr2Rev @ irrev_flux_dist_wt).reshape(-1) if model_irrev.mapIrr2Rev is not None else irrev_flux_dist_wt,
        "firstOptSol": fo_sol,
        "secondOptSol": so_sol,
        "rc_so": np.array([], dtype=float),
        "rc_fo": np.array([], dtype=float),
    }


def fit_fun_mimbl(
    model: StoichiometricModel,
    model_i: StoichiometricModel,
    targets_i: Sequence[int] | np.ndarray,
    target_bounds_i: np.ndarray,
    opt_fit_fun: FitFunctionOptions,
) -> tuple[float, np.ndarray]:
    lb_i, ub_i = apply_target_bounds(model_i, targets_i, target_bounds_i)
    if model_i.subsRxnNum is not None:
        ub_i[model_i.subsRxnNum] = max(ub_i[model_i.subsRxnNum], 1000.0)
    sol = mimbl(
        model_i,
        model_i.fd_ref,
        True,
        opt_fit_fun.excl_rxns_i,
        lb_override=lb_i,
        ub_override=ub_i,
    )
    if sol["x"].size == 0:
        return -1.0, np.array([], dtype=float)
    flux_dist = np.asarray(sol["x"], dtype=float)
    mu = flux_dist[model.bmRxnNum] if model.bmRxnNum is not None else 0.0
    uptake = flux_dist[model.subsRxnNum] if model.subsRxnNum is not None else 0.0
    if mu < opt_fit_fun.minGrowth or abs(uptake) < TOL:
        return 0.0, flux_dist
    if opt_fit_fun.fitParam == 0:
        fit_val = (flux_dist[model.targetRxnNum] * mu) / -uptake
    elif opt_fit_fun.fitParam == 1:
        fit_val = flux_dist[model.targetRxnNum] / -uptake
    else:
        fit_val = flux_dist[model.targetRxnNum]
    return float(fit_val), flux_dist


def multi_obj_mimbl(
    model: StoichiometricModel,
    model_i: StoichiometricModel,
    opt_fit_fun: FitFunctionOptions,
    *,
    lb_i_override: np.ndarray | None = None,
    ub_i_override: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    sol = mimbl(
        model_i,
        model_i.fd_ref,
        True,
        opt_fit_fun.excl_rxns_i,
        lb_override=lb_i_override,
        ub_override=ub_i_override,
    )
    if sol["x"].size == 0:
        return -1.0, np.array([], dtype=float)
    flux_dist = np.asarray(sol["x"], dtype=float)
    mu = flux_dist[model.bmRxnNum] if model.bmRxnNum is not None else 0.0
    uptake = flux_dist[model.subsRxnNum] if model.subsRxnNum is not None else 0.0
    if mu < opt_fit_fun.minGrowth or abs(uptake) < TOL:
        return 0.0, flux_dist
    if opt_fit_fun.fitParam == 0:
        fit_val = (flux_dist[model.bmRxnNum] * flux_dist[model.targetRxnNum] / -flux_dist[model.subsRxnNum]) / max(opt_fit_fun.maxMiMBl, TOL)
    elif opt_fit_fun.fitParam == 1:
        fit_val = (flux_dist[model.targetRxnNum] / -flux_dist[model.subsRxnNum]) / max(opt_fit_fun.maxMiMBl, TOL)
    else:
        fit_val = flux_dist[model.targetRxnNum] / max(opt_fit_fun.maxMiMBl, TOL)
    return float(fit_val), flux_dist


def multi_obj_optknock(
    model: StoichiometricModel,
    opt_fit_fun: FitFunctionOptions,
    *,
    lb_override: np.ndarray | None = None,
    ub_override: np.ndarray | None = None,
) -> tuple[float, np.ndarray, float]:
    objective = np.zeros(model.n_rxns, dtype=float)
    objective[model.bmRxnNum] = 1.0
    lb_eff = model.lb if lb_override is None else np.asarray(lb_override, dtype=float)
    ub_eff = model.ub if ub_override is None else np.asarray(ub_override, dtype=float)
    sol = optimize_cb_model(model, sense="max", objective=objective, one_norm=True, lb_override=lb_eff, ub_override=ub_eff)
    if not sol.success:
        return -1.0, np.array([], dtype=float), -1.0
    flux_dist = sol.x
    max_growth = flux_dist[model.bmRxnNum]
    fit_val = flux_dist[model.targetRxnNum] / max(opt_fit_fun.maxOptKnock, TOL)
    return float(fit_val), flux_dist, float(max_growth)


def multi_obj_robustknock(
    model: StoichiometricModel,
    opt_fit_fun: FitFunctionOptions,
    max_growth: float,
    *,
    lb_override: np.ndarray | None = None,
    ub_override: np.ndarray | None = None,
) -> tuple[float, np.ndarray, float]:
    lb = (model.lb if lb_override is None else np.asarray(lb_override, dtype=float)).copy()
    ub = (model.ub if ub_override is None else np.asarray(ub_override, dtype=float)).copy()
    lb[model.bmRxnNum] = max_growth * 0.99
    ub[model.bmRxnNum] = max_growth
    objective = np.zeros(model.n_rxns, dtype=float)
    objective[model.targetRxnNum] = 1.0
    sol = optimize_cb_model(model, sense="min", objective=objective, lb_override=lb, ub_override=ub)
    if not sol.success:
        return -1.0, np.array([], dtype=float), -1.0
    flux_dist = sol.x
    min_target = flux_dist[model.targetRxnNum]
    fit_val = min_target / max(opt_fit_fun.maxRbstKnock, TOL)
    return float(fit_val), flux_dist, float(min_target)


def multi_obj_gcopt(
    model: StoichiometricModel,
    opt_fit_fun: FitFunctionOptions,
    max_growth: float,
    min_target: float,
    *,
    lb_override: np.ndarray | None = None,
    ub_override: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    growth_red = max_growth * 0.9
    lb = (model.lb if lb_override is None else np.asarray(lb_override, dtype=float)).copy()
    ub = (model.ub if ub_override is None else np.asarray(ub_override, dtype=float)).copy()
    lb[model.bmRxnNum] = growth_red
    ub[model.bmRxnNum] = growth_red
    objective = np.zeros(model.n_rxns, dtype=float)
    objective[model.targetRxnNum] = 1.0
    sol = optimize_cb_model(model, sense="min", objective=objective, lb_override=lb, ub_override=ub)
    if not sol.success:
        return -1.0, np.array([], dtype=float)
    flux_dist = sol.x
    denom = (1.0 / max(max_growth, TOL)) - (1.0 / max(growth_red, TOL))
    b = ((min_target / max(max_growth, TOL)) - (flux_dist[model.targetRxnNum] / max(growth_red, TOL))) / denom
    a = (min_target - b) / max(max_growth, TOL)
    if b <= 0:
        fit_val = (min_target * (max_growth - (-b / max(a, TOL)))) / 2.0
    else:
        fit_val = (max_growth * min_target) - (((min_target - b) * max_growth) / 2.0)
    fit_val += (min_target * (opt_fit_fun.maxMu - max_growth)) / 2.0
    fit_val /= max(opt_fit_fun.maxgcOpt, TOL)
    return float(fit_val), flux_dist


def fit_fun_multi_obj(
    model: StoichiometricModel,
    model_i: StoichiometricModel,
    targets_i: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    target_bounds_i: np.ndarray,
    target_bounds: np.ndarray,
    opt_fit_fun: FitFunctionOptions,
) -> tuple[float, np.ndarray, np.ndarray]:
    lb = model.lb.copy()
    ub = model.ub.copy()
    lb, ub = apply_target_bounds(model, targets, target_bounds, lb=lb, ub=ub)
    lb_i = model_i.lb.copy()
    ub_i = model_i.ub.copy()
    lb_i, ub_i = apply_target_bounds(model_i, targets_i, target_bounds_i, lb=lb_i, ub=ub_i)
    if model_i.subsRxnNum is not None:
        ub_i[model_i.subsRxnNum] = max(ub_i[model_i.subsRxnNum], 1000.0)

    obj_val = -np.ones(4, dtype=float)
    flux_dists: list[np.ndarray] = [np.array([], dtype=float) for _ in range(4)]

    is_obj = np.asarray(opt_fit_fun.isObj, dtype=int)
    if is_obj[0]:
        obj_val[0], flux_dists[0] = multi_obj_mimbl(model, model_i, opt_fit_fun, lb_i_override=lb_i, ub_i_override=ub_i)
    max_growth = -1.0
    if np.any(is_obj[1:]):
        obj_val[1], flux_dists[1], max_growth = multi_obj_optknock(model, opt_fit_fun, lb_override=lb, ub_override=ub)
    min_target = -1.0
    if is_obj[2] or is_obj[3]:
        if max_growth > 0:
            obj_val[2], flux_dists[2], min_target = multi_obj_robustknock(
                model,
                opt_fit_fun,
                max_growth,
                lb_override=lb,
                ub_override=ub,
            )
            if is_obj[3] and min_target > 0:
                obj_val[3], flux_dists[3] = multi_obj_gcopt(
                    model,
                    opt_fit_fun,
                    max_growth,
                    min_target,
                    lb_override=lb,
                    ub_override=ub,
                )

    obj_clean = obj_val.copy()
    obj_clean[(obj_clean == -1.0) | np.isnan(obj_clean)] = 0.0
    lead = int(opt_fit_fun.leadObj) - 1
    fit_val = float(np.dot(obj_clean, np.asarray(opt_fit_fun.weighting, dtype=float)))
    if obj_clean[lead] == 0.0:
        fit_val = 0.0
    flux_dist = flux_dists[lead]
    return fit_val, flux_dist, obj_val


def compute_multiobjective_normalization(model: StoichiometricModel, opt_fit_fun: FitFunctionOptions, is_obj: np.ndarray) -> FitFunctionOptions:
    if np.sum(is_obj) <= 1 and int(opt_fit_fun.leadObj) == 1:
        opt_fit_fun.maxMiMBl = 1.0
        opt_fit_fun.maxOptKnock = 1.0
        opt_fit_fun.maxRbstKnock = 1.0
        opt_fit_fun.maxgcOpt = 1.0
        return opt_fit_fun

    obj_bm = np.zeros(model.n_rxns, dtype=float)
    obj_bm[model.bmRxnNum] = 1.0
    sol_mu = optimize_cb_model(model, sense="max", objective=obj_bm)
    if sol_mu.success:
        max_mu = sol_mu.x[model.bmRxnNum]
    else:
        max_mu = 0.0
    opt_fit_fun.maxMu = float(max_mu)

    obj_prod = np.zeros(model.n_rxns, dtype=float)
    obj_prod[model.targetRxnNum] = 1.0
    lb_prod = model.lb.copy()
    ub_prod = model.ub.copy()
    lb_prod[model.bmRxnNum] = 0.0
    ub_prod[model.bmRxnNum] = 0.0
    sol_prod = optimize_cb_model(model, sense="max", objective=obj_prod, lb_override=lb_prod, ub_override=ub_prod)
    theo_prod = sol_prod.x[model.targetRxnNum] if sol_prod.success else 0.0
    opt_fit_fun.theoProd = float(theo_prod)
    opt_fit_fun.maxOptKnock = max(theo_prod, 1.0)
    opt_fit_fun.maxRbstKnock = max(theo_prod, 1.0)
    opt_fit_fun.maxgcOpt = max((theo_prod * max_mu) / 2.0, 1.0)

    if opt_fit_fun.fitParam == 0:
        grid = 50
        prod_rates = np.linspace(0.0, theo_prod, grid + 1)
        obj_vals = np.zeros(grid + 1, dtype=float)
        for idx, prod_rate in enumerate(prod_rates):
            lb = model.lb.copy()
            ub = model.ub.copy()
            lb[model.targetRxnNum] = prod_rate
            ub[model.targetRxnNum] = prod_rate
            sol = optimize_cb_model(model, sense="max", objective=obj_bm, lb_override=lb, ub_override=ub)
            if sol.success and abs(sol.x[model.subsRxnNum]) > TOL:
                obj_vals[idx] = prod_rate * sol.x[model.bmRxnNum] / -sol.x[model.subsRxnNum]
        opt_fit_fun.maxMiMBl = max(float(np.max(obj_vals)), 1.0)
    elif opt_fit_fun.fitParam == 1:
        if sol_prod.success and abs(sol_prod.x[model.subsRxnNum]) > TOL:
            opt_fit_fun.maxMiMBl = max(float(theo_prod / -sol_prod.x[model.subsRxnNum]), 1.0)
        else:
            opt_fit_fun.maxMiMBl = 1.0
    else:
        opt_fit_fun.maxMiMBl = max(float(theo_prod), 1.0)
    return opt_fit_fun


if __name__ == "__main__":
    print("solver_utils.py imported successfully!")
