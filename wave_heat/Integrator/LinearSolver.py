"""Direct (LU) solver setup for the substep operators."""

import dolfinx as dfx
import dolfinx.fem.petsc as dfx_petsc
from petsc4py import PETSc


def get_linear_solver(
    form: dfx.fem.forms.Form,
    V: dfx.fem.function.FunctionSpace,
    bcs: list[dfx.fem.bcs.DirichletBC],
) -> PETSc.Mat | PETSc.KSP:
    """Assemble `form` with `bcs` and return a KSP that applies its LU factor.

    PREONLY + LU: factorise once, reuse for every time step, no solver
    tolerance entering the error measurements.
    """
    A = dfx_petsc.assemble_matrix(form, bcs=bcs)
    A.assemble()
    solver = PETSc.KSP().create(V.mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    return solver
