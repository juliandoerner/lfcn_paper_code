"""LFCN Integrator"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import ufl
from math import ceil

from Integrator.LinearSolver import get_linear_solver
from Integrator import Forms
from Util.TimeDependentUFLFunctor import TimeDependentUFLFunctor
import dolfinx as dfx
import dolfinx.fem.petsc as dfx_petsc
from mpi4py import MPI
from petsc4py import PETSc

from Util.Meshing import MESH_ID, MeshGeometry, generate_mesh

@dataclass
class DiscretizationData:
    """Mesh plus the degrees of Y1, Y2 (wave) and V (heat)."""

    mesh_geometry: MeshGeometry
    pol_deg_Y1: int = 1
    pol_deg_Y2: int = 1
    pol_deg_V:  int = 1

@dataclass
class ProblemData:
    """Discretisation, problem data and coefficients.

    f_expr and g_expr are UFL callables (x, t); the initial data are numpy
    callables of x only.
    """

    discretization_data: DiscretizationData
    f_expr: Callable[[object, object], object]
    g_expr: Callable[[object, object], object]
    x1_0: Callable[[np.ndarray], float]
    x2_0: Callable[[np.ndarray], float]
    theta_0 : Callable[[np.ndarray], float]
    tau: float = 0.05
    kappa: float = 1.0
    alpha: float = 1.0
    coup_strengh_heat: float = 1.0
    coup_strengh_wave: float = 1.0

class WaveHeatIntegrator(ABC):
    """Mesh, spaces, RHS functors and initial data; subclasses add the scheme."""

    def __init__(self, problem_data: ProblemData):
        self.problem_data = problem_data

        # gmsh gives a tagged, optionally graded interval; the built-in uniform
        # interval is much cheaper to build and is what the sweeps use.
        if self.problem_data.discretization_data.mesh_geometry.gmsh:
            self.mesh_model = generate_mesh(
                problem_data.discretization_data.mesh_geometry)
            self.mesh_data = dfx.io.gmsh.model_to_mesh(
                self.mesh_model, MPI.COMM_WORLD, 0, 1)
        else:
            class _mesh_data:
                def __init__(self, mesh) -> None:
                    self.mesh = mesh

            self.mesh_data = _mesh_data(dfx.mesh.create_interval(
                MPI.COMM_WORLD, 
                ceil(
                    1./self.problem_data.discretization_data.mesh_geometry.lc_far
                ),
                [problem_data.discretization_data.mesh_geometry.left_point,
                problem_data.discretization_data.mesh_geometry.right_point]
                ))


        # setup function spaces
        self.Y1 = dfx.fem.functionspace(
            self.mesh_data.mesh, ("CG", problem_data.discretization_data.pol_deg_Y1))
        self.Y2 = dfx.fem.functionspace(
            self.mesh_data.mesh, ("CG", problem_data.discretization_data.pol_deg_Y2))
        self.V = dfx.fem.functionspace(
            self.mesh_data.mesh, ("CG", problem_data.discretization_data.pol_deg_V))

        # setup rhs functors
        self.functor_f_tn = TimeDependentUFLFunctor(
            lambda t: self.problem_data.f_expr(ufl.SpatialCoordinate(self.Y1.mesh), t),
            self.Y1.mesh)
        self.functor_f_tn_p = TimeDependentUFLFunctor(
            lambda t: self.problem_data.f_expr(ufl.SpatialCoordinate(self.Y1.mesh), t),
            self.Y1.mesh)
        self.functor_g_tnhalf = TimeDependentUFLFunctor(
            lambda t: self.problem_data.g_expr(ufl.SpatialCoordinate(self.V.mesh), t),
            self.V.mesh)

        self.f_tn = self.functor_f_tn(0.0)
        self.f_tn_p = self.functor_f_tn_p(self.problem_data.tau)
        self.g_tnhalf = self.functor_g_tnhalf(self.problem_data.tau / 2.)

        # defining ufl expressions
        #self.trail_funcs = ufl.TrialFunctions(self.fem_space)
        self.x1 = ufl.TrialFunction(self.Y1)
        self.x2 = ufl.TrialFunction(self.Y2)
        self.theta = ufl.TrialFunction(self.V)

        #self.test_funcs = ufl.TestFunctions(self.fem_space)
        self.y1 = ufl.TestFunction(self.Y1)
        self.y2 = ufl.TestFunction(self.Y2)
        self.phi = ufl.TestFunction(self.V)

        #self.tau_ufl = dfx.fem.Constant(self.mesh_data.mesh, problem_data.tau)
        #self.tau_half_ufl = dfx.fem.Constant(self.mesh_data.mesh, problem_data.tau / 2)

        self.tau_ufl = problem_data.tau
        self.tau_half_ufl = problem_data.tau / 2.

        # allocate solution vectors
        #self.soln = dfx.fem.Function(self.fem_space)
        self.x1n = dfx.fem.Function(self.Y1)
        self.x2n = dfx.fem.Function(self.Y2)
        self.thetan = dfx.fem.Function(self.V)

        self.x1nhalf = dfx.fem.Function(self.Y1)


        # allocate update vectors
        self.x1n_p = dfx.fem.Function(self.Y1)
        self.x2n_p = dfx.fem.Function(self.Y2)
        self.thetan_p = dfx.fem.Function(self.V)

        # interpolate initial data
        self.x1n.interpolate(self.problem_data.x1_0)
        self.x2n.interpolate(self.problem_data.x2_0)
        self.thetan.interpolate(self.problem_data.theta_0)

    @abstractmethod
    def _assemble_system_form(self):
        pass

    def set_tau(self, tau: float):
        """Switch to a new step size, refactorising only if tau actually changed."""
        # it is necessary to reassemble the form for a new tau!
        if not np.isclose(tau, self.problem_data.tau, atol=1e-7):
            print(f"Reassemble new_tau = {tau}, self.problem_data.tau = {self.problem_data.tau}")
            self.problem_data.tau = tau
            self.tau_ufl = self.problem_data.tau
            self._assemble_system_form()

    @abstractmethod
    def step(self, tn: float):
        pass

class LFCNIntegrator(WaveHeatIntegrator):
    """The four-substep scheme; fourth_form is first_form, same Y1 mass matrix."""

    def __init__(self, problem_data: ProblemData) -> None:
        super().__init__(problem_data)
        
        # defining forms
        # first step
        self.first_form = dfx.fem.form(Forms.p1(self.x1, self.y1, self.Y1))

        self.first_rhs = dfx.fem.form(
            Forms.p1(self.x1n, self.y1, self.Y1)
            + self.tau_half_ufl 
                * self.problem_data.kappa * Forms.s2(self.x2n, self.y1, self.Y2, self.Y1)
            - self.tau_half_ufl 
                * problem_data.coup_strengh_wave * Forms.b(self.thetan, self.y1, self.V, self.Y1)
            + self.tau_half_ufl 
                * Forms.p1(self.f_tn, self.y1, self.Y1))

        
        # second step
        self.second_form = dfx.fem.form(Forms.p2(self.x2, self.y2, self.Y2))

        self.second_rhs = dfx.fem.form(Forms.p2(self.x2n, self.y2, self.Y2) \
            - self.tau_ufl * Forms.s1(self.x1nhalf, self.y2, self.Y1, self.Y2)
        )
        
        #third step
        self.third_form = dfx.fem.form(
            Forms.p_tilda(self.theta, self.phi, self.V)
            + self.tau_half_ufl 
            * Forms.s_tilda(self.theta, self.phi, self.V, problem_data.alpha)
        )
        
        self.third_rhs = dfx.fem.form(
            Forms.p_tilda(self.thetan, self.phi, self.V)
            - self.tau_half_ufl * Forms.s_tilda(self.thetan, self.phi, self.V, problem_data.alpha)
            + self.tau_ufl * problem_data.coup_strengh_heat * Forms.b_ast(self.x1nhalf, self.phi, self.Y1, self.V)
            + self.tau_ufl * Forms.p_tilda(self.g_tnhalf, self.phi, self.V)
        )
        
        #fourth step
        #self.fourth_form = dfx.fem.form(Forms.p1(self.x1, self.y1))
        self.fourth_form = self.first_form

        self.fourth_rhs = dfx.fem.form(
            Forms.p1(self.x1nhalf, self.y1, self.Y1)
            + self.tau_half_ufl * self.problem_data.kappa * Forms.s2(self.x2n_p, self.y1, self.Y2, self.Y1)
            - self.tau_half_ufl * problem_data.coup_strengh_wave * Forms.b(self.thetan_p, self.y1, self.V, self.Y1)
            + self.tau_half_ufl * Forms.p1(self.f_tn_p, self.y1, self.Y1)
        )

        # create rhs_vector
        self.first_rhs_vector = dfx_petsc.create_vector(self.Y1)
        self.second_rhs_vector = dfx_petsc.create_vector(self.Y2)
        self.thrid_rhs_vector = dfx_petsc.create_vector(self.V)
        self.fourth_rhs_vector = dfx_petsc.create_vector(self.Y1)

        # define bcs
        if hasattr(self.mesh_data, "facet_tags"):
            locater = lambda V, id: dfx.fem.locate_dofs_topological(
                V, self.mesh_data.facet_tags.dim, self.mesh_data.facet_tags.find(id))
            bc_dofs_Y1 = np.concatenate([
                        locater(self.Y1, MESH_ID.LEFT_BC),
                        locater(self.Y1, MESH_ID.RIGHT_BC)])
            bc_dofs_Y2 = np.concatenate([
                        locater(self.Y2, MESH_ID.LEFT_BC),
                        locater(self.Y2, MESH_ID.RIGHT_BC)])
            bc_dofs_V = np.concatenate([
                        locater(self.V, MESH_ID.LEFT_BC),
                        locater(self.V, MESH_ID.RIGHT_BC)])
        else:
            lft_pt = self.problem_data.discretization_data.mesh_geometry.left_point
            rgt_pt = self.problem_data.discretization_data.mesh_geometry.right_point

            is_bc =lambda x: np.logical_or(
                np.isclose(x[0],lft_pt), np.isclose(x[0], rgt_pt))
            bc_facets = dfx.mesh.locate_entities_boundary(self.mesh_data.mesh, 0, is_bc)
            bc_dofs_Y1 = dfx.fem.locate_dofs_topological(self.Y1, 0, bc_facets)
            bc_dofs_Y2 = dfx.fem.locate_dofs_topological(self.Y2, 0, bc_facets)
            bc_dofs_V = dfx.fem.locate_dofs_topological(self.V, 0, bc_facets)

        self.bcs_Y1=[
            dfx.fem.dirichletbc(
                value=PETSc.ScalarType(0),
                dofs=bc_dofs_Y1,
                V=self.Y1
            )
        ]
        self.bcs_Y2=[
            dfx.fem.dirichletbc(
                value=PETSc.ScalarType(0),
                dofs=bc_dofs_Y2,
                V=self.Y2
            )
        ]
        self.bcs_V=[
            dfx.fem.dirichletbc(
                value=PETSc.ScalarType(0),
                dofs=bc_dofs_V,
                V=self.V
            )
        ]
        # self.bcs_V = []

        # assemble system form
        self._assemble_system_form()

    def _assemble_system_form(self) -> None:
        """Factorise the three substep operators for the current tau."""

        self.solver_first_form = get_linear_solver(
            self.first_form, self.Y1, bcs=self.bcs_Y1
        )
        
        self.solver_second_form = get_linear_solver(
            self.second_form, self.Y2, bcs=self.bcs_Y2
        )

        self.solver_third_form = get_linear_solver(
            self.third_form, self.V, bcs=self.bcs_V
        )

        self.solver_fourth_form = self.solver_first_form
    
    def step(self, tn):
        """Advance (x1n, x2n, thetan) from tn to tn + tau in place."""

        # f at both ends of the step, g at the midpoint (Crank-Nicolson heat).
        self.functor_f_tn.t.value = tn
        self.functor_f_tn_p.t.value = tn + self.problem_data.tau
        self.functor_g_tnhalf.t.value = tn + self.problem_data.tau / 2.
        
        # first step
        with self.first_rhs_vector.localForm() as loc_b:
            loc_b.set(0)
        dfx_petsc.assemble_vector(self.first_rhs_vector, self.first_rhs)
        dfx_petsc.apply_lifting(self.first_rhs_vector, [self.first_form], bcs=[self.bcs_Y1])
        self.first_rhs_vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        dfx.fem.set_bc(self.first_rhs_vector, self.bcs_Y1)
        self.solver_first_form.solve(self.first_rhs_vector, self.x1nhalf.x.petsc_vec)

        # second step
        with self.second_rhs_vector.localForm() as loc_b:
            loc_b.set(0)
        dfx_petsc.assemble_vector(self.second_rhs_vector, self.second_rhs)
        dfx_petsc.apply_lifting(self.second_rhs_vector, [self.second_form], bcs=[self.bcs_Y2])
        self.second_rhs_vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        dfx.fem.set_bc(self.second_rhs_vector, self.bcs_Y2)
        self.solver_first_form.solve(self.second_rhs_vector, self.x2n_p.x.petsc_vec)

        # third step
        with self.thrid_rhs_vector.localForm() as loc_b:
            loc_b.set(0)
        dfx_petsc.assemble_vector(self.thrid_rhs_vector, self.third_rhs)
        dfx_petsc.apply_lifting(self.thrid_rhs_vector, [self.third_form], bcs=[self.bcs_V])
        self.thrid_rhs_vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        dfx.fem.set_bc(self.thrid_rhs_vector, self.bcs_V)
        self.solver_third_form.solve(self.thrid_rhs_vector, self.thetan_p.x.petsc_vec)

        # fourth step
        with self.fourth_rhs_vector.localForm() as loc_b:
            loc_b.set(0)
        dfx_petsc.assemble_vector(self.fourth_rhs_vector, self.fourth_rhs)
        dfx_petsc.apply_lifting(self.fourth_rhs_vector, [self.fourth_form], bcs=[self.bcs_Y1])
        self.fourth_rhs_vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        dfx.fem.set_bc(self.fourth_rhs_vector, self.bcs_Y1)
        self.solver_fourth_form.solve(self.fourth_rhs_vector, self.x1n_p.x.petsc_vec)

        # update system
        self.x1n.x.array[:] = self.x1n_p.x.array[:]
        self.x2n.x.array[:] = self.x2n_p.x.array[:]
        self.thetan.x.array[:] = self.thetan_p.x.array[:]
        