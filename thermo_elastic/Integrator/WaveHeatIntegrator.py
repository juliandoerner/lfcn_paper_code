"""LFCN time integrator for the 2D thermo-elastic wave-heat coupling.

The wave part `(v, sigma)` is discretised with DG_k (the upwind operator of
``Integrator.Forms``), the heat part `theta` with CG_l. ``LFCNIntegrator.step``
is the four-step Leapfrog--Crank--Nicolson scheme (paper eq. ``LFCN``):
(1) `v^{n+1/2}`, (2) `sigma^{n+1}`, (3) `theta^{n+1}` by Crank--Nicolson,
(4) `v^{n+1}`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import basix
import ufl
import dolfinx as dfx
import dolfinx.fem.petsc as dfx_petsc
from petsc4py import PETSc

from Integrator.LinearSolver import get_linear_solver
from Integrator.Forms import ThermoElasticForms, Material
from Util.TimeDependentUFLFunctor import TimeDependentUFLFunctor
from Util.Meshing import MeshGeometry, generate_square_mesh


@dataclass
class DiscretizationData:
    mesh_geometry: MeshGeometry
    pol_deg_wave: int = 1     # k : DG degree of v and sigma
    pol_deg_heat: int = 1     # l : CG degree of theta


@dataclass
class ProblemData:
    discretization_data: DiscretizationData
    material: Material
    # sources: (x_ufl, t) -> ufl  (momentum body force f [vector], heat source g [scalar])
    f_expr: Callable
    g_expr: Callable
    # initial data: numpy callables x[gdim, N] -> values
    #   v0     -> (gdim, N)
    #   sigma0 -> (gdim*gdim, N)  full row-major tensor (symmetric space accepts it)
    #   theta0 -> (N,)
    v0_expr: Callable
    sigma0_expr: Callable
    theta0_expr: Callable
    tau: float = 0.05
    # boundary conditions
    wave_dirichlet_tags: tuple | None = None   # None => all exterior facets clamped (v=0)
    heat_dirichlet_tags: tuple | None = None   # None => Neumann on all of d(Omega)
    heat_dirichlet_value: float = 0.0


class WaveHeatIntegrator(ABC):

    def __init__(self, problem_data: ProblemData):
        self.problem_data = problem_data
        dd = problem_data.discretization_data

        # mesh
        self.mesh_data = generate_square_mesh(dd.mesh_geometry)
        mesh = self.mesh_data.mesh
        gdim = mesh.geometry.dim

        # function spaces:  Y1 = v (DG vec), Y2 = sigma (DG sym tensor), V = theta (CG)
        self.Y1 = dfx.fem.functionspace(mesh, ("DG", dd.pol_deg_wave, (gdim,)))
        el_sig = basix.ufl.element(
            "DG", mesh.basix_cell(), dd.pol_deg_wave,
            shape=(gdim, gdim), symmetry=True)
        self.Y2 = dfx.fem.functionspace(mesh, el_sig)
        self.V = dfx.fem.functionspace(mesh, ("CG", dd.pol_deg_heat))

        # forms
        self.forms = ThermoElasticForms(
            mesh, problem_data.material,
            facet_tags=self.mesh_data.facet_tags,
            dirichlet_wave_tags=problem_data.wave_dirichlet_tags)

        # time-dependent sources
        self.functor_f_tn = TimeDependentUFLFunctor(
            lambda t: problem_data.f_expr(ufl.SpatialCoordinate(mesh), t), mesh)
        self.functor_f_tn_p = TimeDependentUFLFunctor(
            lambda t: problem_data.f_expr(ufl.SpatialCoordinate(mesh), t), mesh)
        self.functor_g_tnhalf = TimeDependentUFLFunctor(
            lambda t: problem_data.g_expr(ufl.SpatialCoordinate(mesh), t), mesh)

        self.f_tn = self.functor_f_tn(0.0)
        self.f_tn_p = self.functor_f_tn_p(problem_data.tau)
        self.g_tnhalf = self.functor_g_tnhalf(problem_data.tau / 2.0)

        # trial / test functions
        self.x1 = ufl.TrialFunction(self.Y1)     # v
        self.x2 = ufl.TrialFunction(self.Y2)     # sigma
        self.theta = ufl.TrialFunction(self.V)
        self.y1 = ufl.TestFunction(self.Y1)
        self.y2 = ufl.TestFunction(self.Y2)
        self.phi = ufl.TestFunction(self.V)

        self.tau_ufl = problem_data.tau
        self.tau_half_ufl = problem_data.tau / 2.0

        # state and update vectors
        self.x1n = dfx.fem.Function(self.Y1)
        self.x2n = dfx.fem.Function(self.Y2)
        self.thetan = dfx.fem.Function(self.V)
        self.x1nhalf = dfx.fem.Function(self.Y1)
        self.x1n_p = dfx.fem.Function(self.Y1)
        self.x2n_p = dfx.fem.Function(self.Y2)
        self.thetan_p = dfx.fem.Function(self.V)

        # initial data (interpolate from numpy callables)
        self.x1n.interpolate(problem_data.v0_expr)
        self.x2n.interpolate(problem_data.sigma0_expr)
        self.thetan.interpolate(problem_data.theta0_expr)

    @abstractmethod
    def _assemble_system_form(self):
        pass

    @abstractmethod
    def step(self, tn: float):
        pass


class LFCNIntegrator(WaveHeatIntegrator):

    def __init__(self, problem_data: ProblemData) -> None:
        super().__init__(problem_data)
        F = self.forms
        tau, tauh = self.tau_ufl, self.tau_half_ufl

        # ---- LHS (mass) forms ------------------------------------------------
        self.first_form = dfx.fem.form(F.p1(self.x1, self.y1))           # rho-mass on Y1
        self.second_form = dfx.fem.form(F.p2(self.x2, self.y2))          # C^{-1}-mass on Y2
        self.third_form = dfx.fem.form(                                  # CN heat operator
            F.p_tilda(self.theta, self.phi) + tauh * F.s_tilda(self.theta, self.phi))
        self.fourth_form = self.first_form

        # ---- RHS forms (LFCN eq. One/Two/Three/Four) -------------------------
        # (1) v^{n+1/2}
        self.first_rhs = dfx.fem.form(
            F.p1(self.x1n, self.y1)
            + tauh * F.s2(self.x2n, self.y1)
            - tauh * F.s_pos_1(self.x1n, self.y1)
            - tauh * F.b(self.thetan, self.y1)
            + tauh * F.p1(self.f_tn, self.y1))

        # (2) sigma^{n+1}
        self.second_rhs = dfx.fem.form(
            F.p2(self.x2n, self.y2)
            - tau * F.s1(self.x1nhalf, self.y2)
            - tau * F.s_pos_2(self.x2n, self.y2))

        # (3) theta^{n+1}  (Crank-Nicolson)
        self.third_rhs = dfx.fem.form(
            F.p_tilda(self.thetan, self.phi)
            - tauh * F.s_tilda(self.thetan, self.phi)
            + tau * F.b_ast(self.x1nhalf, self.phi)
            + tau * F.p_tilda(self.g_tnhalf, self.phi))

        # (4) v^{n+1}
        self.fourth_rhs = dfx.fem.form(
            F.p1(self.x1nhalf, self.y1)
            + tauh * F.s2(self.x2n_p, self.y1)
            - tauh * F.s_pos_1(self.x1n, self.y1)
            - tauh * F.b(self.thetan_p, self.y1)
            + tauh * F.p1(self.f_tn_p, self.y1))

        # rhs vectors
        self.first_rhs_vector = dfx_petsc.create_vector(self.Y1)
        self.second_rhs_vector = dfx_petsc.create_vector(self.Y2)
        self.thrid_rhs_vector = dfx_petsc.create_vector(self.V)
        self.fourth_rhs_vector = dfx_petsc.create_vector(self.Y1)

        # ---- boundary conditions --------------------------------------------
        # wave: Dirichlet enforced weakly via the flux -> no strong bcs
        self.bcs_Y1 = []
        self.bcs_Y2 = []
        # heat: strong Dirichlet if requested, else natural Neumann
        self.bcs_V = self._make_heat_bcs()

        self._assemble_system_form()

    def _make_heat_bcs(self):
        pd = self.problem_data
        if pd.heat_dirichlet_tags is None:
            return []
        ft = self.mesh_data.facet_tags
        mesh = self.mesh_data.mesh
        fdim = mesh.topology.dim - 1
        mesh.topology.create_connectivity(fdim, mesh.topology.dim)
        facets = np.concatenate([ft.find(int(t)) for t in pd.heat_dirichlet_tags])
        dofs = dfx.fem.locate_dofs_topological(self.V, fdim, facets)
        return [dfx.fem.dirichletbc(
            PETSc.ScalarType(pd.heat_dirichlet_value), dofs, self.V)]

    def _assemble_system_form(self) -> None:
        self.solver_first_form = get_linear_solver(self.first_form, self.Y1, bcs=self.bcs_Y1)
        self.solver_second_form = get_linear_solver(self.second_form, self.Y2, bcs=self.bcs_Y2)
        self.solver_third_form = get_linear_solver(self.third_form, self.V, bcs=self.bcs_V)
        self.solver_fourth_form = self.solver_first_form

    def destroy(self) -> None:
        """Free the PETSc solvers, their factored matrices and the RHS vectors,
        and reclaim the MPI communicators they duplicated.

        Each KSP/MUMPS factorisation duplicates a communicator that petsc4py only
        releases on ``garbage_cleanup``; without this, sweeping many problems in a
        single process (e.g. the convergence drivers) exhausts the communicator
        pool ("Too many communicators" / MUMPS ``INFOG(1)=-3``). The fourth solver
        is an alias of the first, so it is skipped.
        """
        for solver in (self.solver_first_form, self.solver_second_form,
                       self.solver_third_form):
            try:
                A = solver.getOperators()[0]
                solver.destroy()
                A.destroy()
            except Exception:
                pass
        for vec in (self.first_rhs_vector, self.second_rhs_vector,
                    self.thrid_rhs_vector, self.fourth_rhs_vector):
            try:
                vec.destroy()
            except Exception:
                pass
        PETSc.garbage_cleanup()

    @staticmethod
    def _solve_step(rhs_vector, rhs_form, lhs_form, bcs, solver, out):
        with rhs_vector.localForm() as loc_b:
            loc_b.set(0)
        dfx_petsc.assemble_vector(rhs_vector, rhs_form)
        if bcs:
            dfx_petsc.apply_lifting(rhs_vector, [lhs_form], bcs=[bcs])
        rhs_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        if bcs:
            dfx.fem.set_bc(rhs_vector, bcs)
        solver.solve(rhs_vector, out.x.petsc_vec)
        out.x.scatter_forward()

    def step(self, tn):
        tau = self.problem_data.tau
        self.functor_f_tn.t.value = tn
        self.functor_f_tn_p.t.value = tn + tau
        self.functor_g_tnhalf.t.value = tn + tau / 2.0

        # (1) v^{n+1/2}
        self._solve_step(self.first_rhs_vector, self.first_rhs, self.first_form,
                         self.bcs_Y1, self.solver_first_form, self.x1nhalf)
        # (2) sigma^{n+1}
        self._solve_step(self.second_rhs_vector, self.second_rhs, self.second_form,
                         self.bcs_Y2, self.solver_second_form, self.x2n_p)
        # (3) theta^{n+1}
        self._solve_step(self.thrid_rhs_vector, self.third_rhs, self.third_form,
                         self.bcs_V, self.solver_third_form, self.thetan_p)
        # (4) v^{n+1}
        self._solve_step(self.fourth_rhs_vector, self.fourth_rhs, self.fourth_form,
                         self.bcs_Y1, self.solver_fourth_form, self.x1n_p)

        # advance
        self.x1n.x.array[:] = self.x1n_p.x.array[:]
        self.x2n.x.array[:] = self.x2n_p.x.array[:]
        self.thetan.x.array[:] = self.thetan_p.x.array[:]
