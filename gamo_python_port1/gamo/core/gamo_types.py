from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping

import numpy as np
from scipy import sparse


@dataclass
class LinearSolution:
    status: str
    success: bool
    x: np.ndarray
    objective_value: float
    message: str = ""
    reduced_costs: np.ndarray | None = None

    @property
    def objval(self) -> float:
        return self.objective_value

    @property
    def f(self) -> float:
        return self.objective_value

    @property
    def origStat(self) -> str:
        return self.status


@dataclass
class StoichiometricModel:
    rxns: list[str]
    mets: list[str]
    S: sparse.csr_matrix
    lb: np.ndarray
    ub: np.ndarray
    c: np.ndarray
    genes: list[str] = field(default_factory=list)
    grRules: list[str] = field(default_factory=list)
    rxnGeneMat: sparse.csr_matrix | np.ndarray | None = None
    csense: str | list[str] | None = None
    b: np.ndarray | None = None
    rev: np.ndarray | None = None
    description: str | None = None
    rxnNames: list[str] = field(default_factory=list)
    metNames: list[str] = field(default_factory=list)
    metFormulas: list[str] = field(default_factory=list)
    subSystems: list[str] = field(default_factory=list)
    bmRxn: str | None = None
    subsRxn: str | None = None
    targetRxn: str | None = None
    bmRxnNum: int | None = None
    subsRxnNum: int | None = None
    targetRxnNum: int | None = None
    fd_ref: np.ndarray | None = None
    hetRxnNum: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    hetRxns: list[list[Any]] = field(default_factory=list)
    matchRev: list[bool] | None = None
    rev2irrev: list[np.ndarray] | None = None
    irrev2rev: np.ndarray | None = None
    B: sparse.csr_matrix | None = None
    mapIrr2Rev: sparse.csr_matrix | None = None
    comprMapVec: np.ndarray | None = None
    comprMapMat: sparse.csr_matrix | None = None
    geneMatCompress: list[str] = field(default_factory=list)

    def copy(self) -> "StoichiometricModel":
        return StoichiometricModel(
            rxns=list(self.rxns),
            mets=list(self.mets),
            S=self.S.copy(),
            lb=self.lb.copy(),
            ub=self.ub.copy(),
            c=self.c.copy(),
            genes=list(self.genes),
            grRules=list(self.grRules),
            rxnGeneMat=self.rxnGeneMat.copy() if hasattr(self.rxnGeneMat, "copy") else self.rxnGeneMat,
            csense=list(self.csense) if isinstance(self.csense, list) else self.csense,
            b=None if self.b is None else self.b.copy(),
            rev=None if self.rev is None else self.rev.copy(),
            description=self.description,
            rxnNames=list(self.rxnNames),
            metNames=list(self.metNames),
            metFormulas=list(self.metFormulas),
            subSystems=list(self.subSystems),
            bmRxn=self.bmRxn,
            subsRxn=self.subsRxn,
            targetRxn=self.targetRxn,
            bmRxnNum=self.bmRxnNum,
            subsRxnNum=self.subsRxnNum,
            targetRxnNum=self.targetRxnNum,
            fd_ref=None if self.fd_ref is None else self.fd_ref.copy(),
            hetRxnNum=self.hetRxnNum.copy(),
            hetRxns=[list(row) for row in self.hetRxns],
            matchRev=None if self.matchRev is None else list(self.matchRev),
            rev2irrev=None if self.rev2irrev is None else [arr.copy() for arr in self.rev2irrev],
            irrev2rev=None if self.irrev2rev is None else self.irrev2rev.copy(),
            B=None if self.B is None else self.B.copy(),
            mapIrr2Rev=None if self.mapIrr2Rev is None else self.mapIrr2Rev.copy(),
            comprMapVec=None if self.comprMapVec is None else self.comprMapVec.copy(),
            comprMapMat=None if self.comprMapMat is None else self.comprMapMat.copy(),
            geneMatCompress=list(self.geneMatCompress),
        )

    @property
    def n_rxns(self) -> int:
        return len(self.rxns)

    @property
    def n_mets(self) -> int:
        return len(self.mets)

    def reaction_index(self, rxn: str | int) -> int:
        if isinstance(rxn, int):
            return rxn
        return self.rxns.index(rxn)

    def gene_index(self, gene: str | int) -> int:
        if isinstance(gene, int):
            return gene
        return self.genes.index(gene)

    def refresh_special_reaction_indices(self) -> None:
        if self.bmRxn in self.rxns:
            self.bmRxnNum = self.rxns.index(self.bmRxn)
        if self.subsRxn in self.rxns:
            self.subsRxnNum = self.rxns.index(self.subsRxn)
        if self.targetRxn in self.rxns:
            self.targetRxnNum = self.rxns.index(self.targetRxn)


@dataclass
class EncodingInfo:
    encodeVec: np.ndarray
    encodeVec_midPos: np.ndarray
    numBits: int
    numBits_hri: int
    posGene: np.ndarray
    K: int
    K_hri: int
    K_tot: int
    numBits_tot: int


@dataclass
class TargetSet:
    rxnNum: np.ndarray
    score: np.ndarray
    bound: np.ndarray
    map: np.ndarray
    rxnNum_hri: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    score_hri: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    bound_hri: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=float))
    Nt_hri: int = 0
    fd_ref: np.ndarray | None = None
    K: int = 0
    K_hri: int = 0
    Nt: int = 0
    Nt_tot: int = 0
    shift: int = 0
    rxnNum_KO: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    KDID: np.ndarray | None = None
    geneNum: np.ndarray | None = None
    genes: list[str] = field(default_factory=list)
    bound_gene: np.ndarray | None = None
    ub_gene_rxn_mat: np.ndarray | None = None
    lb_gene_rxn_mat: np.ndarray | None = None
    genes_rule: list[list[int]] = field(default_factory=list)
    ub_rule: list[np.ndarray] = field(default_factory=list)
    rxnNum_i: list[np.ndarray] = field(default_factory=list)
    bound_i: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    B_t: sparse.csr_matrix | None = None
    B_t_bound: sparse.csr_matrix | None = None
    rev2irrev: list[np.ndarray] | None = None
    gene_rule_cache: dict[str, Any] = field(default_factory=dict)


@dataclass
class GAMOOptions:
    saveFile: str | None = None
    saveFolder: str | None = None
    threads: int = 1
    redFlag: int = 0
    compress: int = 0
    numInt: int = 3
    optFocus: str = "rxns"
    memPop: int = 1
    popSize: int = 20
    maxGen: int = 100
    genSize: int = 5
    slctRate: float = 0.5
    mutRate: float = 0.2
    elite: int = 2
    initPopType: int = 1
    slctPressure: float = 2.0
    fitFun: int = 0
    noMatingIdent: int = 1
    numKntChr: int = 1
    crossType: int = 0
    nonTarget: list[str] = field(default_factory=list)
    numInsertions: int = 0
    hetRxnNum: np.ndarray | None = None
    modelType: int = 0
    rMMA_flag: int = 0
    typeInt: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "GAMOOptions") -> "GAMOOptions":
        if isinstance(value, GAMOOptions):
            return value
        if value is None:
            return cls()
        data = {k: v for k, v in dict(value).items() if k in cls.__dataclass_fields__}
        if "nonTarget" in data and data["nonTarget"] is None:
            data["nonTarget"] = []
        return cls(**data)


@dataclass
class FitFunctionOptions:
    excl_rxns: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    excl_rxns_i: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    isObj: str | np.ndarray = "M"
    leadObj: str | int = "M"
    weighting: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0, 1.0], dtype=float))
    fitParam: int = 0
    minGrowth: float = 0.0
    minInt: int = 0
    FIRF: float = 0.1
    optFocus: int = 0
    numIntMaxFit: int = 0
    maxMiMBl: float = 1.0
    maxOptKnock: float = 1.0
    maxRbstKnock: float = 1.0
    maxgcOpt: float = 1.0
    theoProd: float = 0.0
    maxMu: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "FitFunctionOptions") -> "FitFunctionOptions":
        if isinstance(value, FitFunctionOptions):
            return value
        if value is None:
            return cls()
        data = {k: v for k, v in dict(value).items() if k in cls.__dataclass_fields__}
        if "excl_rxns" in data and not isinstance(data["excl_rxns"], np.ndarray):
            data["excl_rxns"] = np.asarray(data["excl_rxns"], dtype=int)
        if "weighting" in data and not isinstance(data["weighting"], np.ndarray):
            data["weighting"] = np.asarray(data["weighting"], dtype=float)
        return cls(**data)


def as_numpy_int_array(values: Iterable[int] | np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.array([], dtype=int)
    if isinstance(values, np.ndarray):
        return values.astype(int, copy=False)
    return np.asarray(list(values), dtype=int)


def as_namespace_dict(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError(f"Unsupported struct-like object: {type(value)!r}")
