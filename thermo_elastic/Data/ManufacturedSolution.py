"""Base class for the 2D thermo-elastic manufactured solutions.

A manufactured solution is built from spatial profiles ``V`` (velocity) and
``Th`` (temperature) and scalar time profiles ``a, A, c`` (with ``A' = a``):

    v(x,t)     = V(x)  * a(t)
    sigma(x,t) = C eps(V(x)) * A(t)     (=> d/dt sigma = C eps(v), zero source)
    theta(x,t) = Th(x) * c(t)

The momentum/heat sources f, g follow from the PDE. Two time regimes (selected
by ``mode``):

    "space" : a=1, A=t, c=t           -> low time-degree, LFCN temporally exact
                                         (isolates the spatial error; for the
                                         mesh-size experiment).
    "time"  : a=cos wt, A=sin(wt)/w,
              c=sin et                 -> oscillatory; nonzero temporal error
                                         (for the time-step / CFL experiment).

Concrete solutions (``TrigSolution``, ``PolySolution``) only supply the spatial
profiles and the cell type; everything else is shared here.
"""
from abc import ABC, abstractmethod
import math

import numpy as np
import ufl
import dolfinx as dfx
from mpi4py import MPI

from Integrator import WaveHeatIntegrator as I
from Integrator.Forms import Material, eps
from Util.Meshing import MeshGeometry
from Util.Util import TimeLoop, get_h_max

PI = np.pi


def C_iso(e, mu, lam):
    """Isotropic elasticity tensor C e = 2 mu e + lambda tr(e) I (2D)."""
    return 2.0 * mu * e + lam * ufl.tr(e) * ufl.Identity(2)


def l2_error(uh, u_ex_ufl, qdeg):
    """Global L2 error ||uh - u_ex|| via high-quadrature assembly."""
    mesh = uh.function_space.mesh
    diff = uh - u_ex_ufl
    form = dfx.fem.form(ufl.inner(diff, diff) * ufl.dx(metadata={"quadrature_degree": qdeg}))
    local = dfx.fem.assemble_scalar(form)
    return math.sqrt(mesh.comm.allreduce(local, op=MPI.SUM))


class ManufacturedSolution(ABC):
    #: mesh cell type this solution is designed for ("triangle" / "quadrilateral")
    cell_type = "triangle"

    def __init__(self, material: Material, mode: str = "space",
                 omega: float = 2 * PI, eta: float = 2 * PI):
        self.mat = material
        self.mode = mode
        if mode == "space":
            self.a = lambda t: 1.0 + 0.0 * t
            self.ap = lambda t: 0.0 * t
            self.A = lambda t: t
            self.c = lambda t: t
            self.cp = lambda t: 1.0 + 0.0 * t
        elif mode == "time":
            self.a = lambda t: ufl.cos(omega * t)
            self.ap = lambda t: -omega * ufl.sin(omega * t)
            self.A = lambda t: ufl.sin(omega * t) / omega
            self.c = lambda t: ufl.sin(eta * t)
            self.cp = lambda t: eta * ufl.cos(eta * t)
        else:
            raise ValueError(f"unknown mode {mode!r}")

    # ---- spatial profiles (supplied by concrete solutions) ------------------
    @abstractmethod
    def V_ufl(self, x):
        """UFL velocity profile V(x) (vector); must vanish on the boundary."""

    @abstractmethod
    def Th_ufl(self, x):
        """UFL temperature profile Th(x) (scalar); must vanish on the boundary."""

    @abstractmethod
    def v0_np(self, x):
        """Numpy velocity IC V(x), shape (gdim, N) (since a(0)=1)."""

    # sigma0, theta0 vanish because A(0)=0, c(0)=0
    def sigma0_np(self, x):
        return np.zeros((4, x.shape[1]))

    def theta0_np(self, x):
        return np.zeros(x.shape[1])

    # ---- sources (from the PDE) ---------------------------------------------
    def f_expr(self, x, t):
        mu, lam, rho, alpha = self.mat.mu, self.mat.lam, self.mat.rho, self.mat.alpha
        divCeV = ufl.div(C_iso(eps(self.V_ufl(x)), mu, lam))
        return (self.V_ufl(x) * self.ap(t)
                - (1.0 / rho) * divCeV * self.A(t)
                - (alpha / rho) * ufl.grad(self.Th_ufl(x)) * self.c(t))

    def g_expr(self, x, t):
        kappa, alpha = self.mat.kappa, self.mat.alpha
        lapTh = ufl.div(ufl.grad(self.Th_ufl(x)))
        return (self.Th_ufl(x) * self.cp(t)
                - kappa * lapTh * self.c(t)
                - alpha * ufl.div(self.V_ufl(x)) * self.a(t))

    # ---- exact fields -------------------------------------------------------
    def v_exact(self, x, t):
        return self.V_ufl(x) * self.a(t)

    def sigma_exact(self, x, t):
        return C_iso(eps(self.V_ufl(x)), self.mat.mu, self.mat.lam) * self.A(t)

    def theta_exact(self, x, t):
        return self.Th_ufl(x) * self.c(t)

    # ---- run one experiment and return the L2 errors ------------------------
    def compute_errors(self, n, tau, k, l, T,
                       divergence_cap=1e6, check_every=25):
        """Run to ``T`` on an ``n x n`` unit-square mesh and return
        ``(h, err_v, err_sigma, err_theta)``. If the run diverges (CFL
        violation), the errors are returned as ``inf``.
        """
        geom = MeshGeometry(x0=0.0, y0=0.0, x1=1.0, y1=1.0,
                            n=n, cell_type=self.cell_type)
        pd = I.ProblemData(
            discretization_data=I.DiscretizationData(
                geom, pol_deg_wave=k, pol_deg_heat=l),
            material=self.mat,
            f_expr=self.f_expr, g_expr=self.g_expr,
            v0_expr=self.v0_np, sigma0_expr=self.sigma0_np, theta0_expr=self.theta0_np,
            tau=tau, wave_dirichlet_tags=None, heat_dirichlet_tags=(10, 20, 30, 40))
        integ = I.LFCNIntegrator(pd)
        try:
            mesh = integ.mesh_data.mesh
            h = get_h_max(mesh)

            tn = 0.0
            for step_idx, tn in TimeLoop(tau, 0.0, T):
                integ.step(tn)
                if step_idx % check_every == 0:
                    arr = integ.x1n.x.array
                    local = float(np.max(np.abs(arr))) if arr.size else 0.0
                    vmax = mesh.comm.allreduce(local, op=MPI.MAX)
                    if not np.isfinite(vmax) or vmax > divergence_cap:
                        return h, math.inf, math.inf, math.inf
            te = tn + tau

            x = ufl.SpatialCoordinate(mesh)
            qdeg = 2 * (max(k, l) + 3)
            ev = l2_error(integ.x1n, self.v_exact(x, te), qdeg)
            es = l2_error(integ.x2n, self.sigma_exact(x, te), qdeg)
            et = l2_error(integ.thetan, self.theta_exact(x, te), qdeg)
            return h, ev, es, et
        finally:
            # release PETSc solvers/comms so a long sweep does not exhaust the
            # MPI communicator pool (see LFCNIntegrator.destroy)
            integ.destroy()
