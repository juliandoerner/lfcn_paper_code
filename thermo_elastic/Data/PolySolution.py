"""Polynomial manufactured solution (for the time-step / CFL experiment).

    P(x)  = x(1-x) y(1-y)
    V(x)  = (P, P),   Th(x) = P

Both vanish on the boundary of the unit square. P lies in Q_2, so on a
quadrilateral mesh with degree-2 elements the solution is represented exactly in
space and the measured error is the pure temporal error. Use
``cell_type='quadrilateral'`` and ``k = l = 2``.
"""
import numpy as np
import ufl

from Data.ManufacturedSolution import ManufacturedSolution


def _P_ufl(x):
    return x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])


class PolySolution(ManufacturedSolution):
    cell_type = "quadrilateral"

    def V_ufl(self, x):
        P = _P_ufl(x)
        return ufl.as_vector((P, P))

    def Th_ufl(self, x):
        return _P_ufl(x)

    def v0_np(self, x):
        P = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
        return np.vstack((P, P))
