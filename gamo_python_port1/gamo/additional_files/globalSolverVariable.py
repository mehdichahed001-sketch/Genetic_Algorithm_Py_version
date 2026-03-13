def globalSolverVariable(solver):
    """Compatibility shim for the MATLAB COBRA global solver variable."""
    return str(solver)

__all__ = ["globalSolverVariable"]
