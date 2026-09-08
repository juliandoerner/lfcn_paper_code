
import dolfinx as dfx
import dolfinx.fem.petsc as dfx_petsc
from petsc4py import PETSc


def get_linear_solver(
    form: dfx.fem.forms.Form,
    V: dfx.fem.function.FunctionSpace,
    bcs: list[dfx.fem.bcs.DirichletBC],
) -> PETSc.Mat | PETSc.KSP:
    # FenicsX interfacec
    A = dfx_petsc.assemble_matrix(form, bcs=bcs)
    A.assemble()
    solver = PETSc.KSP().create(V.mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    # MUMPS is a parallel direct solver: identical (exact) result in serial and
    # under MPI, where PETSc's built-in LU would be sequential-only.
    pc.setFactorSolverType("mumps")
    return solver
