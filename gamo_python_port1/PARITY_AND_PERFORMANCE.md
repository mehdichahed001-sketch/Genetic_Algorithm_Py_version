# MATLAB Parity And Performance Report

## Scope

This report summarizes a direct parity check between the MATLAB sources in `original_matlab/` and the Python port, and documents performance-focused changes applied to the Python implementation.

## Parity Verification

### File-level mapping

- MATLAB `.m` files found: `37`
- MATLAB files with Python counterpart wrapper/module: `37`
- Missing mappings: `0`

The mapping is implemented through:

- `gamo/gamo_files/` wrappers
- `gamo/additional_files/` wrappers
- `examples/GAMO_Example.py` (for `original_matlab/GAMO_Example.m`)

### Behavioral alignment updates

The following behavior was aligned with MATLAB logic in this update:

- `compressModel` null-reaction LP handling now treats non-optimal solves as `-1` (MATLAB behavior), not `0`.
  - This prevents false positives in null-reaction detection when LPs are non-optimal/unbounded.

## Performance Enhancements Implemented

### 1. Reusable HiGHS LP path for repeated objective sweeps

Added a reusable HiGHS model path in `gamo/core/solver_utils.py` (`reaction_extrema`) and used it in:

- `manual_fva`
- `manual_fva_bounds` (new)
- `compress_model` null-reaction detection path

The code auto-falls back to SciPy `linprog` when `highspy` is unavailable.

### 2. Red-flag FVA optimization path

`GAMO` now uses `manual_fva_bounds` (min/max only) during target-space reduction instead of computing/storing full flux matrices that are not consumed in that code path.

### 3. Robustness fix in GA loop

`genetic_algorithm` now handles `maxGen=0` safely without `UnboundLocalError` by initializing final population objective arrays before the evolution loop.

### 4. Compressed-model exclusion mapping fix

When compression changes reaction indexing, excluded reaction indices are now mapped through `comprMapMat` before irreversible-space conversion. This fixes out-of-range errors in compressed runs.

### 5. Second-pass solver reuse for larger models

Added reusable HiGHS state in `optimize_cb_model` for repeated `A_eq` LPs with changing objectives/bounds:

- cache keyed by model identity,
- objective and bounds are updated by diff only,
- solver warm-start behavior is preserved across calls.

Also removed avoidable model copies in fitness routines by using bound overrides directly:

- `fit_fun_mimbl` now passes `lb_override`/`ub_override` into `mimbl`,
- `fit_fun_multi_obj` now works on bound arrays instead of copying full model structs,
- multi-objective helpers accept optional bound overrides.

## Benchmark Snapshot (This Environment)

Measured on the current machine and dataset:

- `_free_flux_null_reactions(iAF1260Core)`:
  - before: `~0.53 s`
  - after: `~0.058 s`
- `_free_flux_null_reactions(iJO1366)`:
  - before: exceeded `120 s` timeout
  - after: `~217 s` (completed)
- `manual_fva_bounds(iJO1366)`:
  - before (`manual_fva` path): exceeded `120 s` timeout
  - after: `~172 s` (completed)

Additional second-pass benchmark:

- repeated `120` LP solves on `iJO1366` (changing objective and one reaction bound each call):
  - baseline `solve_lp` rebuild path: `~11.113 s`
  - reusable `optimize_cb_model` path: `~5.569 s`
  - speedup: `~2.00x`

## Notes

- The original `multiObj_ATPsc.m` in the public MATLAB source is empty; Python preserves this as an explicit placeholder wrapper by design.
- Numerical outputs remain solver-dependent; parity here is workflow/logic parity rather than bit-for-bit solver identity.
