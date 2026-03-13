# GAMO Python Port

This repository is a MATLAB-to-Python migration of the public **GAMO** project (**Genetic Algorithm for Metabolic Optimization**).

The port keeps the original file mapping while restructuring the executable logic into reusable Python modules:

- `gamo/core/` contains the real implementation.
- `gamo/gamo_files/` and `gamo/additional_files/` contain compatibility wrappers that mirror the original MATLAB filenames.
- `examples/GAMO_Example.py` is the Python equivalent of the original example script.
- `data/` contains the example `.mat` models and supporting datasets.
- `original_matlab/` preserves the original MATLAB source tree for reference.

## What was migrated

The port includes:

- metabolic model loading from MATLAB `.mat` files
- reversible-to-irreversible model conversion
- reference flux distribution generation
- target-space initialization for reaction and gene interventions
- GA operators: initialization, selection, crossover, mutation, fitness memoization
- MiMBl and the multi-objective fitness workflow
- result post-processing and target analysis
- a Python wrapper for every original MATLAB `.m` file in the public ZIP

## Main design choices

1. **Core implementation + compatibility wrappers**
   The original MATLAB project relied on dynamic `.m` file generation and direct MATLAB/COBRA conventions. In Python, the logic was moved into stable modules and the original filenames were preserved as thin wrappers.

2. **Python-native target evaluation**
   MATLAB generated `evalTargets.m` at runtime. This port replaces that with Python parsing/evaluation of gene rules.

3. **SciPy-based optimization backend**
   The original public code assumes MATLAB + COBRA Toolbox + Gurobi-style workflows. This port replaces those calls with Python implementations built around `scipy.optimize`, sparse matrices, and NumPy.

4. **Conservative compression fallback**
   If compression would collapse the model to an empty network, the port falls back to the uncompressed model instead of failing.

## Installation

Create a Python environment and install:

```bash
pip install -r requirements.txt
```

Optional performance acceleration (recommended for large models):

```bash
pip install highspy
```

If `highspy` is available, repeated LP routines used by compression and FVA run through a reusable HiGHS model path for faster execution. The code automatically falls back to SciPy if `highspy` is not installed.

The solver layer also reuses HiGHS state for repeated equality-constrained LP calls (`optimize_cb_model`), updating only objective and changed bounds between solves.

## Quick start

Run the example from the repository root:

```bash
python examples/GAMO_Example.py
```

Useful options:

```bash
python examples/GAMO_Example.py --reference-mode fba
python examples/GAMO_Example.py --reference-mode experimental
python examples/GAMO_Example.py --genome-scale
python examples/GAMO_Example.py --heterologous
```

The default example uses the faster `fba` reference-flux mode. The `experimental` mode reproduces the original Ishii-2007-style reference fitting path, but it is substantially heavier.

## Minimal API example

```python
from gamo import GAMO, GAMOOptions, FitFunctionOptions, load_matlab_model

model = load_matlab_model("data/iAF1260Core.mat")
# configure model.bmRxn, model.targetRxn, model.subsRxn, model.fd_ref ...

opt = GAMOOptions(popSize=8, maxGen=1, genSize=1, numInt=3, optFocus="gene", fitFun=0)
fit = FitFunctionOptions(minGrowth=0.0)
results, prob = GAMO(model, opt, fit, {})
```

## Project structure

```text
GAMO_python_port/
├── gamo/
│   ├── core/
│   ├── gamo_files/
│   └── additional_files/
├── examples/
├── data/
├── original_matlab/
├── CONVERSION_NOTES.md
├── VALIDATION.md
└── requirements.txt
```

## Compatibility notes

- Results are saved as **pickle** (`.pkl`) instead of MATLAB `.mat` files.
- The original MATLAB code used generated helper files and MATLAB toolbox solvers; the Python port replaces that with static modules and SciPy-based routines.
- Numerical results should be treated as a **best-effort migration**, not a bit-for-bit reproduction of MATLAB/Gurobi runs.
- The public `multiObj_ATPsc.m` file shipped empty in the original repository; the Python port preserves it as an explicit placeholder wrapper.
