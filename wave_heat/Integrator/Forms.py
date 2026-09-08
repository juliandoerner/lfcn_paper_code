"""Bilinear forms of the 1D wave/heat system, named after the blocks of the
scheme: p* mass, s* stiffness, b/b_ast the two coupling terms."""

from enum import Enum
from typing import Union
from collections.abc import Callable

import ufl.constant
import ufl.form

import dolfinx as dfx
import ufl

# Trail functions (x_1, x_2, theta)
# Test function (y_1, y_2, phi)
# spaces (Y1, Y2, V)


def p1(x1: Union[ufl.TrialFunction, dfx.fem.Function],
        y1: ufl.TestFunction,
        Y1: dfx.fem.FunctionSpace) -> ufl.Form:
    """Mass form on Y1."""
    
    dx = ufl.Measure("dx", Y1.mesh)
    return ufl.inner(x1, y1) * dx
    

def p2(x2: Union[ufl.TrialFunction, dfx.fem.Function],
       y2: ufl.TestFunction,
       Y2: dfx.fem.FunctionSpace) -> ufl.Form:
    """Mass form on Y2."""
    
    dx = ufl.Measure("dx", Y2.mesh)
    return ufl.inner(x2, y2) * dx


def s1(x1: Union[ufl.TrialFunction, dfx.fem.Function],
       y2: ufl.TestFunction,
       Y1: dfx.fem.FunctionSpace,
       Y2: dfx.fem.FunctionSpace) -> ufl.Form:
    """Wave block coupling x1 into the x2 equation (carries the minus sign)."""
    
    assert(Y1.mesh == Y2.mesh)
    
    dx = ufl.Measure("dx", Y1.mesh)
    return dfx.fem.Constant(Y2.mesh, -1.) * ufl.inner(x1, y2) * dx

def s2(x2: Union[ufl.TrialFunction, dfx.fem.Function],
       y1: ufl.TestFunction,
       Y2: dfx.fem.FunctionSpace,
       Y1: dfx.fem.FunctionSpace) -> ufl.Form: 
    """Wave stiffness block -(grad x2, grad y1)."""
    
    assert(Y2.mesh == Y1.mesh)

    dx = ufl.Measure("dx", Y2.mesh)
    return dfx.fem.Constant(Y2.mesh, -1.) \
        * ufl.inner(ufl.grad(x2)[0], ufl.grad(y1)[0]) * dx

def p_tilda(theta: Union[ufl.TrialFunction, dfx.fem.Function],
            phi: ufl.TestFunction,
            V: dfx.fem.FunctionSpace) -> ufl.Form:
    """Mass form on V."""
    
    dx = ufl.Measure("dx", V.mesh)

    return ufl.inner(theta, phi) * dx

def s_tilda(theta: Union[ufl.TrialFunction, dfx.fem.Function],
            phi: ufl.TestFunction,
            V: dfx.fem.FunctionSpace,
            alpha_dif: float) -> ufl.Form:
    """Heat stiffness alpha_dif * (grad theta, grad phi)."""
    dx = ufl.Measure("dx", V.mesh)

    return dfx.fem.Constant(V.mesh,alpha_dif) \
        * ufl.inner(ufl.grad(theta)[0], ufl.grad(phi)[0]) * dx

def b(theta : Union[ufl.TrialFunction, dfx.fem.Function],
      y1: ufl.TestFunction,
      V: dfx.fem.FunctionSpace,
      Y1: dfx.fem.FunctionSpace) -> ufl.Form:
    """Coupling of theta into the wave equation, weak form of +theta_x."""
    
    assert(V.mesh == Y1.mesh)

    dx = ufl.Measure("dx", V.mesh)
    return dfx.fem.Constant(V.mesh, -1.) \
        * ufl.inner(theta, ufl.grad(y1)[0]) * dx

def b_ast(x1: Union[ufl.TrialFunction, dfx.fem.Function],
          phi: ufl.TestFunction,
          Y1: dfx.fem.FunctionSpace,
          V: dfx.fem.FunctionSpace) -> ufl.Form:
    """Coupling of x1 into the heat equation, weak form of -x1_x; adjoint of b."""
    
    assert(Y1.mesh == V.mesh)
    
    dx = ufl.Measure("dx", Y1.mesh)
    return ufl.inner(x1, ufl.grad(phi)[0]) * dx