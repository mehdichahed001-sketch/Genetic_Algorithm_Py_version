# Conversion Notes

This file maps the public MATLAB repository to the Python port.

| Original MATLAB file | Python destination | Notes |
|---|---|---|
| `GAMO_Example.m` | `examples/GAMO_Example.py` | Python example script |
| `GAMO Files/GAMO.m` | `gamo/core/ga_utils.py + gamo/gamo_files/GAMO.py` | Top-level optimization entry point |
| `GAMO Files/adaptMutRate.m` | `gamo/core/ga_utils.py + gamo/gamo_files/adaptMutRate.py` | Adaptive mutation-rate helper |
| `GAMO Files/analyzeGAMOresults.m` | `gamo/core/analysis_utils.py + gamo/gamo_files/analyzeGAMOresults.py` | Result analysis |
| `GAMO Files/analyzePropProg.m` | `gamo/core/ga_utils.py + gamo/gamo_files/analyzePropProg.py` | Population plateau analysis |
| `GAMO Files/compressModel.m` | `gamo/core/model_utils.py + gamo/gamo_files/compressModel.py` | Model compression |
| `GAMO Files/decode.m` | `gamo/core/target_utils.py + gamo/gamo_files/decode.py` | Binary chromosome decoding |
| `GAMO Files/deleteReactions.m` | `gamo/core/model_utils.py + gamo/gamo_files/deleteReactions.py` | Reaction deletion utility |
| `GAMO Files/encode.m` | `gamo/core/target_utils.py + gamo/gamo_files/encode.py` | Binary chromosome encoding |
| `GAMO Files/evalFitness.m` | `gamo/core/ga_utils.py + gamo/gamo_files/evalFitness.py` | Fitness evaluation |
| `GAMO Files/evalFitness_comb.m` | `gamo/core/ga_utils.py + gamo/gamo_files/evalFitness_comb.py` | Combination post-processing fitness |
| `GAMO Files/evalFitness_mem.m` | `gamo/core/ga_utils.py + gamo/gamo_files/evalFitness_mem.py` | Memoized fitness evaluation |
| `GAMO Files/fitFun_MiMBl.m` | `gamo/core/solver_utils.py + gamo/gamo_files/fitFun_MiMBl.py` | MiMBl fitness |
| `GAMO Files/geneticAlgorithm.m` | `gamo/core/ga_utils.py + gamo/gamo_files/geneticAlgorithm.py` | Genetic algorithm loop |
| `GAMO Files/initStructLP.m` | `gamo/core/solver_utils.py + gamo/gamo_files/initStructLP.py` | LP problem initialization |
| `GAMO Files/initializeFitFun.m` | `gamo/core/ga_utils.py + gamo/gamo_files/initializeFitFun.py` | Fitness-function setup |
| `GAMO Files/initializePopulation.m` | `gamo/core/target_utils.py + gamo/gamo_files/initializePopulation.py` | Population initialization |
| `GAMO Files/mating.m` | `gamo/core/ga_utils.py + gamo/gamo_files/mating.py` | Crossover/mating |
| `GAMO Files/multiObj_ATPsc.m` | `gamo/gamo_files/multiObj_ATPsc.py` | Placeholder preserved; original file was empty |
| `GAMO Files/multiObj_MiMBl.m` | `gamo/core/solver_utils.py + gamo/gamo_files/multiObj_MiMBl.py` | Multi-objective MiMBl |
| `GAMO Files/multiObj_OptKnock.m` | `gamo/core/solver_utils.py + gamo/gamo_files/multiObj_OptKnock.py` | OptKnock objective |
| `GAMO Files/multiObj_RobustKnock.m` | `gamo/core/solver_utils.py + gamo/gamo_files/multiObj_RobustKnock.py` | RobustKnock objective |
| `GAMO Files/multiObj_gcOpt.m` | `gamo/core/solver_utils.py + gamo/gamo_files/multiObj_gcOpt.py` | gcOpt objective |
| `GAMO Files/mutation.m` | `gamo/core/ga_utils.py + gamo/gamo_files/mutation.py` | Mutation operator |
| `GAMO Files/redTargetSpace.m` | `gamo/core/model_utils.py + gamo/gamo_files/redTargetSpace.py` | Target-space reduction |
| `GAMO Files/selection.m` | `gamo/core/ga_utils.py + gamo/gamo_files/selection.py` | Selection operator |
| `GAMO Files/transformFitness.m` | `gamo/core/ga_utils.py + gamo/gamo_files/transformFitness.py` | Intervention-count fitness transform |
| `GAMO Files/translatePop.m` | `gamo/core/target_utils.py + gamo/gamo_files/translatePop.py` | Translate chromosomes to interventions |
| `GAMO Files/writeLogGeneRules.m` | `gamo/core/target_utils.py + gamo/gamo_files/writeLogGeneRules.py` | Gene-rule preparation |
| `GAMO Files/writeRxnRules.m` | `gamo/core/target_utils.py + gamo/gamo_files/writeRxnRules.py` | Reaction-target helper |
| `Additional Files/MiMBl.m` | `gamo/core/solver_utils.py + gamo/additional_files/MiMBl.py` | Underlying MiMBl solver |
| `Additional Files/addNetworkBranches.m` | `gamo/core/model_utils.py + gamo/additional_files/addNetworkBranches.py` | Heterologous reaction insertion |
| `Additional Files/createRefFD.m` | `gamo/core/model_utils.py + gamo/additional_files/createRefFD.py` | Reference flux distribution generation |
| `Additional Files/fd_rev2irr.m` | `gamo/core/model_utils.py + gamo/additional_files/fd_rev2irr.py` | Reference-flux conversion to irreversible space |
| `Additional Files/globalSolverVariable.m` | `gamo/additional_files/globalSolverVariable.py` | Compatibility shim |
| `Additional Files/manualFVA.m` | `gamo/core/solver_utils.py + gamo/additional_files/manualFVA.py` | Flux variability analysis |
| `Additional Files/rev2irr.m` | `gamo/core/model_utils.py + gamo/additional_files/rev2irr.py` | Reversible-to-irreversible conversion |

## Data files preserved

- `data/iAF1260Core.mat`
- `data/iJO1366.mat`
- `data/fluxData_Ishii_2007.xlsx`
- `data/heterogeneous_rxn_list_iAF1260Core.mat`
- `data/heterogeneous_rxn_list_iJO1366.mat`

## Implementation notes

- Executable logic lives in `gamo/core/`.
- MATLAB-style file names are preserved as wrapper modules in `gamo/gamo_files/` and `gamo/additional_files/`.
- The original MATLAB source was copied into `original_matlab/` for side-by-side inspection.

## Verification note

See `PARITY_AND_PERFORMANCE.md` for the latest parity verification summary and performance-focused implementation updates.
