"""Trigonometric manufactured solution (for the mesh-size experiment).

    V(x)  = (sin k pi x sin k pi y, sin k pi x sin k pi y)
    Th(x) =  sin k pi x sin k pi y

Both vanish on the boundary of the unit square. The profile is not contained in
the finite-element space, so refining the mesh reveals the genuine spatial order.
The wave number ``k`` must be a positive integer, so that sin(k pi) = 0 and the
profile still vanishes on the right/top edge. Designed for triangular meshes.
"""
import numpy as np
import ufl

from Data.ManufacturedSolution import ManufacturedSolution, PI


class TrigSolution(ManufacturedSolution):
    cell_type = "triangle"

    def __init__(self, material, mode="space", k: int = 1, **kwargs):
        super().__init__(material, mode=mode, **kwargs)
        if int(k) != k or k < 1:
            raise ValueError(
                f"wave number k must be a positive integer (so the profile "
                f"vanishes on the boundary), got {k!r}")
        self.k = int(k)

    def V_ufl(self, x):
        s = ufl.sin(self.k * PI * x[0]) * ufl.sin(self.k * PI * x[1])
        return ufl.as_vector((s, s))

    def Th_ufl(self, x):
        return ufl.sin(self.k * PI * x[0]) * ufl.sin(self.k * PI * x[1])

    def v0_np(self, x):
        s = np.sin(self.k * PI * x[0]) * np.sin(self.k * PI * x[1])
        return np.vstack((s, s))
