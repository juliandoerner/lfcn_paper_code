#!/usr/bin/env python3
"""Stability boundary: bisects for the largest stable tau on each mesh size h.

Stability is judged by comparing against a one-cell reference run rather than an
exact solution, so the bisection only needs a yes/no answer.

Usage::

    python3 TraceCFL.py     # writes output_trace_cfl/
"""

import os

from matplotlib.ticker import StrMethodFormatter
import numpy as np

from Util.Util import TimeLoop
import Integrator.WaveHeatIntegrator as Integrator
from Util import Norms
from Util.Meshing import MeshGeometry
from Data import Solutions
import matplotlib.pyplot as plt


def run(mesh_size: float, tau: float, deg: int):
    """Integrate to T = 1 and return the energy-norm distance to a one-cell run."""

    mesh_geometry = MeshGeometry(
        left_point = 0.0,
        right_point = 1.0,
        mid_point = 0.5,
        lc_near = mesh_size,
        lc_far = mesh_size,
        gmsh = False
    )
    mesh_geometry_ref = MeshGeometry(
        left_point = 0.0,
        right_point = 1.0,
        mid_point = 0.5,
        lc_near = 1.0,
        lc_far = 1.0,
        gmsh = False
    )

    discretization_data = Integrator.DiscretizationData(
        mesh_geometry = mesh_geometry,
        pol_deg_Y1  = deg,
        pol_deg_Y2  = deg,
        pol_deg_V   = deg,
    )
    discretization_data_ref = Integrator.DiscretizationData(
        mesh_geometry = mesh_geometry_ref,
        pol_deg_Y1  = 2,
        pol_deg_Y2  = 2,
        pol_deg_V   = 2,
    )

    #ex_solution = Solutions.OscSolution(3.73*np.pi, 2.3*np.pi)
    ex_solution = Solutions.ExpSolution(3.73*np.pi, 2.3*np.pi)
    #ex_solution = Solutions.ExpExpSolution(2.*np.pi, 10, 2.3*np.pi, 7)

    f_expr = ex_solution.f_expr
    g_expr = ex_solution.g_expr

    x1_0 = lambda x: ex_solution.x1_expr(x,0.0)
    x2_0 = lambda x: ex_solution.x2_expr(x,0.0)
    theta_0 = lambda x: ex_solution.theta_expr(x,0.0)


    problem_data = Integrator.ProblemData(
        discretization_data = discretization_data,
        f_expr = f_expr,
        g_expr = g_expr,
        x1_0 = x1_0,
        x2_0 = x2_0,
        theta_0 = theta_0,
        tau = tau,
        kappa = 1.0,
        alpha = 1.0,
        coup_strengh_heat = 1.0,
        coup_strengh_wave = 1.0
    )
    problem_data_ref = Integrator.ProblemData(
        discretization_data = discretization_data_ref,
        f_expr = f_expr,
        g_expr = g_expr,
        x1_0 = x1_0,
        x2_0 = x2_0,
        theta_0 = theta_0,
        tau = tau,
        kappa = 1.0,
        alpha = 1.0,
        coup_strengh_heat = 1.0,
        coup_strengh_wave = 1.0
    )

    integrator = Integrator.LFCNIntegrator(problem_data=problem_data)
    integrator_ref = Integrator.LFCNIntegrator(problem_data=problem_data_ref)

    for _, tn in TimeLoop(problem_data.tau, 0.0, 1.):
        
        integrator.set_tau(tau)
        integrator.step(tn)

        integrator_ref.set_tau(tau)
        integrator_ref.step(tn)
    

    e1 = Norms.error_norm_ref(integrator.x1n, integrator_ref.x1n, "L2")
    e2 = Norms.error_norm_ref(integrator.x2n, integrator_ref.x2n, "H10")
    e3 = Norms.error_norm_ref(integrator.thetan, integrator_ref.thetan, "L2")
    
    #print(np.sqrt(e1**2 + e2**2 + e3**2))
    return np.sqrt(e1**2 + e2**2 + e3**2)

def trace(
        mesh_size:float, 
        tau_high: float, 
        tau_low: float,
        deg: int, 
        confidence: float = 1e-5,
        eps: float = 1e-1
        ) -> tuple[float, tuple[float, float]] | None:
    """Bisect for the critical tau, to a bracket of width `confidence`.

    `tau_low` must be stable and `tau_high` unstable against the threshold
    `eps`; the assert below fires if the bracket is not of that form.
    """

    assert tau_high > tau_low

    res_tau_high = run(mesh_size, tau_high, deg)
    res_tau_low = run(mesh_size, tau_low, deg)
    is_tau_high = True if  res_tau_high < eps else False
    is_tau_low = True if res_tau_low < eps else False

    #print(is_tau_high)
    #print(is_tau_low)
    if not (is_tau_low and not is_tau_high):
        print(f"is_tau_low = {is_tau_low}, res = {res_tau_low}")
        print(f"is_tau_high = {is_tau_high}, res = {res_tau_high}")
        assert False
    
    while tau_high - tau_low > confidence:

        tau_mid = (tau_high - tau_low)*0.5 + tau_low
        is_tau_mid = True if run(mesh_size, tau_mid, deg) < eps else False

        if is_tau_mid:
            tau_low = tau_mid
            is_tau_low = is_tau_mid
        else:
            tau_high = tau_mid
            is_tau_high = is_tau_mid
    

    return tau_mid, (tau_low, tau_high)
    

def main() -> None:
    """Trace the critical tau across the h range and write the CSV and figure."""
    os.makedirs("output_trace_cfl", exist_ok=True)
    
    hs = np.linspace(0.01,0.001, num=20)
    deg = 1

    tau_cfls = []
    for h in hs:
        print(f"Starting trace for h = {h}")
        # tau = h is expected unstable, tau = h/10 stable.
        tau_high = h
        tau_low = h / 10.

        tau_cfl, _ = trace(h, tau_high, tau_low, deg)
        tau_cfls.append(tau_cfl)

        print(f"Found tau_cfl = {tau_cfl}")

    tau_cfls = np.array(tau_cfls)
    
    plt.plot(hs, tau_cfls, marker="*")
    #plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("output_trace_cfl/cfl_plot.png")
    np.savetxt(
        f"output_trace_cfl/cfl_data_deg_{deg}.txt", 
        np.array([hs, tau_cfls]).transpose(), 
        delimiter=",",
        header="h,tau",
        comments="")

    print(f"EOC = {Norms.EOC(hs, tau_cfls)}")
    slopes = np.diff(tau_cfls) / np.diff(hs)
    mean_slope = slopes.mean()
    print(f"Slope = {mean_slope}")


if __name__ == "__main__":
    main()