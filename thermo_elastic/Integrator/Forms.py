"""Bilinear forms for the 2D thermo-elastic wave-heat coupling.

State (paper Section `sec:ana_thermo_wave`):
    wave part  x = (x1, x2) = (v, sigma)   v : velocity (vector),
                                           sigma : symmetric stress (tensor)
    heat part  theta                       temperature (scalar)

Discretisation: DG_k for the wave part `(v, sigma)` (the upwind operator of the
paper Appendix `appendix:Elastic`), CG_l for the heat part `theta`.

Abstract <-> code dictionary (cf. the paper):
    p1        ipwaveOne      <rho v, w>
    p2        ipwaveTwo      <C^{-1} sigma, lambda>
    s1        sbilwaveOne    DG form of  -<eps(v), lambda>        (operator A_1)
    s2        sbilwaveTwo    DG form of   <div sigma, w>          (operator A_2)
    s_pos_1   B_+,1          upwind stabilisation acting on v
    s_pos_2   B_+,2          upwind stabilisation acting on sigma
    p_tilda   ipheat         <theta, phi>
    s_tilda   sbilheat       <kappa grad theta, grad phi>
    b         coupopform     -<alpha grad theta, w>               (heat -> wave)
    b_ast     coupopform*    -<alpha grad phi,   v>               (wave -> heat)
"""

from dataclasses import dataclass

import numpy as np
import ufl
import dolfinx as dfx


@dataclass
class Material:
    """Homogeneous isotropic thermo-elastic material parameters."""
    rho: float = 1.0        # density
    mu: float = 1.0         # Lame mu  (> 0)
    lam: float = 1.0        # Lame lambda  (>= 0)
    alpha: float = 1.0      # thermo-elastic coupling
    kappa: float = 1.0      # thermal diffusivity

    @property
    def c_P(self) -> float:
        """Pressure-wave speed (standard isotropic: sqrt((lambda+2mu)/rho))."""
        return float(np.sqrt((self.lam + 2.0 * self.mu) / self.rho))

    @property
    def c_S(self) -> float:
        """Shear-wave speed sqrt(mu/rho)."""
        return float(np.sqrt(self.mu / self.rho))


def eps(w):
    """Symmetric gradient eps(w) = sym(grad w)."""
    return ufl.sym(ufl.grad(w))


class ThermoElasticForms:
    """Builds the UFL forms for the thermo-elastic LFCN scheme.

    Parameters
    ----------
    mesh : the (shared) computational mesh.
    mat  : :class:`Material`.
    facet_tags : MeshTags on facets (edge markers); may be ``None``.
    dirichlet_wave_tags : iterable of facet markers carrying the wave Dirichlet
        (clamped) flux.  If ``None`` and ``facet_tags`` is ``None`` *all* exterior
        facets are treated as Dirichlet.
    """

    def __init__(self, mesh, mat: Material, facet_tags=None,
                 dirichlet_wave_tags=None):
        self.mesh = mesh
        self.mat = mat
        self.facet_tags = facet_tags
        self.dim = mesh.topology.dim
        self.I = ufl.Identity(self.dim)
        self._n = ufl.FacetNormal(mesh)

        # measures
        self.dx = ufl.Measure("dx", domain=mesh)
        self.dS = ufl.Measure("dS", domain=mesh)               # interior facets
        if facet_tags is not None:
            self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
        else:
            self.ds = ufl.Measure("ds", domain=mesh)

        # which exterior facets get the Dirichlet wave flux
        self._dir_tags = dirichlet_wave_tags

    # ---- exterior measure restricted to the Dirichlet wave boundary ----------
    def _ds_dir(self):
        if self._dir_tags is None:
            return self.ds                       # all exterior facets
        tags = tuple(int(t) for t in self._dir_tags)
        meas = self.ds(tags[0])
        for t in tags[1:]:
            meas = meas + self.ds(t)
        return meas

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _tangent(n):
        return ufl.as_vector((-n[1], n[0]))

    # ---- material operators --------------------------------------------------
    def Cinv(self, s):
        """Compliance  C^{-1} s = (1/2mu)(s - lambda/(2mu + d lambda) tr(s) I)."""
        mu, lam, d = self.mat.mu, self.mat.lam, self.dim
        return (1.0 / (2.0 * mu)) * (
            s - (lam / (2.0 * mu + d * lam)) * ufl.tr(s) * self.I
        )

    # ---- mass forms (ipwave) -------------------------------------------------
    def p1(self, v, w):
        """ipwaveOne(v, w) = <rho v, w>."""
        return self.mat.rho * ufl.inner(v, w) * self.dx

    def p2(self, s, l):
        """ipwaveTwo(sigma, lambda) = <C^{-1} sigma, lambda>."""
        return ufl.inner(self.Cinv(s), l) * self.dx

    # ---- skew wave forms (A_h^skew, eq. SkewFormGlobal) ----------------------
    def s2(self, s, w):
        """sbilwaveTwo(sigma, w) = DG form of <div sigma, w> (operator A_2).

        bulk:           + int_K div(sigma) . w
        interior faces: + <n.([[sigma]] n), n.{{w}}> + <t.([[sigma]] n), t.{{w}}>
        (no boundary term)
        """
        n = self._n("+")
        t = self._tangent(n)
        pj_s = s("-") - s("+")                       # [[sigma]]_F (paper)
        sn = ufl.dot(pj_s, n)                         # [[sigma]] n   (vector)
        face = (ufl.dot(n, sn) * ufl.dot(n, ufl.avg(w))
                + ufl.dot(t, sn) * ufl.dot(t, ufl.avg(w))) * self.dS
        bulk = ufl.inner(ufl.div(s), w) * self.dx
        return bulk + face

    def s1(self, v, l):
        """sbilwaveOne(v, lambda) = DG form of -<eps(v), lambda> (operator A_1).

        bulk:           - int_K eps(v) : lambda
        interior faces: - <n.[[v]], n.({{lambda}} n)> - <t.[[v]], t.({{lambda}} n)>
        Dirichlet bnd:  + <v, lambda n>            (clamped, weak v = 0)
        """
        n = self._n("+")
        t = self._tangent(n)
        pj_v = v("-") - v("+")                        # [[v]]_F (paper)
        ln = ufl.dot(ufl.avg(l), n)                   # {{lambda}} n  (vector)
        face = -(ufl.dot(n, pj_v) * ufl.dot(n, ln)
                 + ufl.dot(t, pj_v) * ufl.dot(t, ln)) * self.dS
        bulk = -ufl.inner(eps(v), l) * self.dx
        nb = self._n
        bnd = ufl.dot(v, ufl.dot(l, nb)) * self._ds_dir()
        return bulk + face + bnd

    # ---- positive wave stabilisation (A_h^pos, eq. PosFormGlobal) ------------
    def s_pos_2(self, s, l):
        """B_+,2(sigma, lambda): upwind stabilisation on sigma (sigma-lambda jumps)."""
        n = self._n("+")
        t = self._tangent(n)
        a1 = 1.0 / (2.0 * self.mat.rho * self.mat.c_P)
        a5 = 1.0 / (2.0 * self.mat.rho * self.mat.c_S)
        jsn = ufl.dot(ufl.jump(s), n)
        jln = ufl.dot(ufl.jump(l), n)
        return (a1 * ufl.dot(n, jsn) * ufl.dot(n, jln)
                + a5 * ufl.dot(t, jsn) * ufl.dot(t, jln)) * self.dS

    def s_pos_1(self, v, w):
        """B_+,1(v, w): upwind stabilisation on v (v-w jumps + Dirichlet bnd)."""
        n = self._n("+")
        t = self._tangent(n)
        a4 = 0.5 * self.mat.rho * self.mat.c_P
        a8 = 0.5 * self.mat.rho * self.mat.c_S
        jv, jw = ufl.jump(v), ufl.jump(w)
        face = (a4 * ufl.dot(n, jv) * ufl.dot(n, jw)
                + a8 * ufl.dot(t, jv) * ufl.dot(t, jw)) * self.dS
        nb = self._n
        tb = self._tangent(nb)
        rcP = self.mat.rho * self.mat.c_P
        rcS = self.mat.rho * self.mat.c_S
        bnd = (rcP * ufl.dot(nb, v) * ufl.dot(nb, w)
               + rcS * ufl.dot(tb, v) * ufl.dot(tb, w)) * self._ds_dir()
        return face + bnd

    # ---- heat forms ----------------------------------------------------------
    def p_tilda(self, theta, phi):
        """ipheat(theta, phi) = <theta, phi>."""
        return ufl.inner(theta, phi) * self.dx

    def s_tilda(self, theta, phi):
        """sbilheat(theta, phi) = <kappa grad theta, grad phi>."""
        return self.mat.kappa * ufl.inner(ufl.grad(theta), ufl.grad(phi)) * self.dx

    # ---- coupling forms ------------------------------------------------------
    def b(self, theta, w):
        """coupopform(theta, w) = -<alpha grad theta, w>   (heat -> wave)."""
        return -self.mat.alpha * ufl.inner(ufl.grad(theta), w) * self.dx

    def b_ast(self, v, phi):
        """coupopform*(v, phi) = -<alpha grad phi, v>      (wave -> heat)."""
        return -self.mat.alpha * ufl.inner(ufl.grad(phi), v) * self.dx
