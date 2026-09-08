#!/usr/bin/env python3
"""Convergence in time: energy-norm error vs step size tau, on eight meshes.

Runs that went unstable are filtered out with an `err < 10` mask before the
order is reported.

Usage::

    python3 TauPlot.py      # writes output_tau_plot/
"""

import os

from matplotlib.ticker import StrMethodFormatter
import numpy as np

import Integrator.WaveHeatIntegrator as Integrator
from Util import Norms
from Util.Util import TimeLoop
from Util.Meshing import MeshGeometry
from Data import Solutions
import matplotlib.pyplot as plt


def run(mesh_size: float, tau: float):
    """Integrate to T = 1 and return the largest energy-norm error over all steps."""

    mesh_geometry = MeshGeometry(
        left_point = 0.0,
        right_point = 1.0,
        mid_point = 0.5,
        lc_near = mesh_size,
        lc_far = mesh_size,
        gmsh = False
    )

    discretization_data = Integrator.DiscretizationData(
        mesh_geometry = mesh_geometry,
        pol_deg_Y1 = 2,
        pol_deg_Y2 = 2,
        pol_deg_V = 2,
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


    integrator = Integrator.LFCNIntegrator(problem_data=problem_data)

    #print(f"target h: {mesh_size}, actual h: {get_h_max(integrator.mesh_data.mesh)}")
    
    errs = []
    tn = None
    
    for _, tn in TimeLoop(problem_data.tau, 0.0, 1.):
    #for _, tn, tau in EndTimeLoop(problem_data.tau, 0.0, 1.):
        e1 = Norms.error_norm(integrator.x1n, lambda x: ex_solution.x1_expr(x, tn), "L2")
        e2 = Norms.error_norm(integrator.x2n, lambda x: ex_solution.x2_expr(x, tn), "H10")
        e3 = Norms.error_norm(integrator.thetan, lambda x: ex_solution.theta_expr(x, tn), "L2")
        
        errs.append(np.sqrt(e1**2 + e2**2 + e3**2))
        
        integrator.set_tau(tau)
        integrator.step(tn)

    #print(f"End tn = {tn}")

    errs = np.array(errs)
    return np.max(errs)

def main() -> None:
    """Sweep tau on each mesh, write the CSVs, the figure and the orders."""
    os.makedirs("output_tau_plot", exist_ok=True)
    
    #hs = [0.5, 0.1, 0.05, 0.01, 0.05]
    hs = np.geomspace(0.5, 0.005, num=8)
    taus = np.geomspace(0.1, 0.001, num=40)
    #hs = [0.5, 0.1]
    #taus = np.geomspace(0.01,0.0001, num=10)

    errs_hs = []
    for h in hs:
        errs_hs.append([])
        for tau in taus:
            print(f"run: h = {h}, tau = {tau}")
            errs_hs[-1].append(run(h, tau))

    errs_hs = np.array(errs_hs)

    
    for idx, h in enumerate(hs):

        # The `< 10` mask drops the runs that went unstable.
        print(errs_hs[idx])
        plt.loglog(taus[errs_hs[idx] < 10], errs_hs[idx][errs_hs[idx] < 10], marker="*")
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
        plt.tight_layout()
        plt.savefig("output_tau_plot/error_plot.png")
        plt.grid(True)
        np.savetxt(
            f"output_tau_plot/{h}_data.txt", 
            np.array([taus,errs_hs[idx]]).transpose(), 
            delimiter=",",
            header="tau,err",
            comments="")

        print(f"EOC = {Norms.EOC(taus[errs_hs[idx] < 10], errs_hs[idx][errs_hs[idx] < 10])}")


if __name__ == "__main__":
    main()