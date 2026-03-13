from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gamo import (  # noqa: E402
    GAMO,
    FitFunctionOptions,
    GAMOOptions,
    add_network_branches,
    change_rxn_bounds,
    create_ref_fd,
    load_mat_struct,
    load_matlab_model,
)


def build_example_model(use_genome_scale: bool = False, include_heterologous: bool = False, reference_mode: str = "fba"):
    data_dir = ROOT / "data"
    model_file = data_dir / ("iJO1366.mat" if use_genome_scale else "iAF1260Core.mat")
    model = load_matlab_model(model_file)

    model = change_rxn_bounds(model, "EX_glc__D_e", -13.34, "l")
    model = change_rxn_bounds(model, "EX_o2_e", -20.0, "l")

    model.targetRxn = "EX_succ_e"
    model.bmRxn = "BIOMASS_Ec_iJO1366_WT_53p95M" if use_genome_scale else "BIOMASS_Ecoli_core_w_GAM"
    model.subsRxn = "EX_glc__D_e"
    model.refresh_special_reaction_indices()

    if reference_mode == "experimental":
        model, _, _ = create_ref_fd(
            model,
            None,
            1,
            {"filename": str(data_dir / "fluxData_Ishii_2007.xlsx")},
        )
    else:
        model, _, _ = create_ref_fd(model, None, 0, {})

    if include_heterologous:
        rxn_list = load_mat_struct(
            data_dir / ("heterogeneous_rxn_list_iJO1366.mat" if use_genome_scale else "heterogeneous_rxn_list_iAF1260Core.mat"),
            "rxn_list_trunc",
        )
        model = add_network_branches(model, rxn_list)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Python GAMO example")
    parser.add_argument("--genome-scale", action="store_true", help="Use the iJO1366 model instead of the core model")
    parser.add_argument("--heterologous", action="store_true", help="Include heterologous reaction insertion candidates")
    parser.add_argument("--reference-mode", choices=["fba", "experimental"], default="fba", help="Reference flux distribution mode")
    parser.add_argument("--pop-size", type=int, default=8, help="Population size per island")
    parser.add_argument("--max-gen", type=int, default=1, help="Number of gene-drift iterations")
    parser.add_argument("--gen-size", type=int, default=1, help="Number of generations per drift")
    parser.add_argument("--threads", type=int, default=1, help="Number of sequential islands")
    args = parser.parse_args()

    model = build_example_model(use_genome_scale=args.genome_scale, include_heterologous=args.heterologous, reference_mode=args.reference_mode)

    opt = GAMOOptions(
        saveFile="GAMO_example_run",
        saveFolder=str(ROOT / "results"),
        threads=args.threads,
        redFlag=1,
        compress=1,
        numInt=3,
        optFocus="gene",
        memPop=1,
        popSize=args.pop_size,
        maxGen=args.max_gen,
        genSize=args.gen_size,
        slctRate=0.25,
        mutRate=0.05,
        elite=1,
        initPopType=0,
        slctPressure=2.0,
        fitFun=1,
        numInsertions=1 if args.heterologous else 0,
    )

    excl_rxns = np.where(np.asarray(model.rxnGeneMat.getnnz(axis=1)).reshape(-1) == 0)[0]
    diffusion = [i for i, name in enumerate(model.rxnNames) if "diffusion" in name.lower()]
    spontaneous = [i for i, name in enumerate(model.rxnNames) if "spontaneous" in name.lower()]
    excl_rxns = np.unique(np.concatenate([excl_rxns, np.asarray(diffusion + spontaneous, dtype=int)]))
    if model.hetRxnNum.size:
        excl_rxns = np.unique(np.concatenate([excl_rxns, model.hetRxnNum]))

    fit_opt = FitFunctionOptions(
        excl_rxns=excl_rxns,
        weighting=np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        isObj="OG",
        leadObj="M",
        fitParam=0,
        minGrowth=0.1,
        minInt=0,
        FIRF=0.1,
    )

    results, _ = GAMO(model, opt, fit_opt, {})
    print(f"Finished. Final population size: {len(results['res'])}")
    if results["res_best"]:
        best = results["res_best"][0]
        print("Best design fitness:", best["fitness"])
        print("Reaction KOs:", best["KO"])
        print("Gene KOs:", best["Gene_KO"])
        print("Insertions:", best["Ins"])
    if "save_path" in results:
        print("Saved results to:", results["save_path"])


if __name__ == "__main__":
    main()
