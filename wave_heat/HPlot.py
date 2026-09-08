#!/usr/bin/env python3
"""Convergence in space: energy-norm error vs mesh size h at a fixed tau = 1e-3.

Two sweeps: equal degrees in the wave and heat spaces, and a `miss_matched_`
one with the heat space a degree below the wave spaces.

Usage::

    python3 HPlot.py        # writes output_h_plot/
"""

import os

import numpy as np

import Integrator.WaveHeatIntegrator as Integrator
from Util import Norms
from Util.Meshing import MeshGeometry
from Util.Util import get_h_min
from Data import Solutions
# from Data import GenSolutions
import matplotlib.pyplot as plt

from Util.Util import TimeLoop


def run(mesh_size: float, degree_wave: int, degree_heat: int, tau: float):
    """Integrate to T = 1 and return `(h, energy-norm error at T)`."""
    

    mesh_geometry = MeshGeometry(
        left_point = 0.0,
        right_point = 1.0,
        mid_point = 0.5,
        lc_near = mesh_size,
        lc_far = mesh_size,
    )
    

    discretization_data = Integrator.DiscretizationData(
        mesh_geometry = mesh_geometry,
        pol_deg_Y1 = degree_wave,
        pol_deg_Y2 = degree_wave,
        pol_deg_V = degree_heat,
    )



    #ex_solution = Solutions.OscSolution(3.73*np.pi, 2.3*np.pi)
    #ex_solution = Solutions.ExpSolution(3.73*np.pi, 2.3*np.pi)
    #ex_solution = Solutions.ExpExpSolution(2.*np.pi, 21, 2.3*np.pi, 13)
    #ex_solution = Solutions.PolyExpSolution(2*np.pi,3*np.pi)
    ex_solution = Solutions.MonExpSolution(3*np.pi,4*np.pi)


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

    mesh_size_actual = get_h_min(integrator.mesh_data.mesh)
    print(f"mesh_size_actual = {mesh_size_actual}, mesh_size = {mesh_size}")
    
    tn = None
    for _, tn in TimeLoop(problem_data.tau, 0.0, 1.):

        integrator.set_tau(tau)
        integrator.step(tn)

    # TimeLoop yields the time before each step, so the state is one tau later.
    tn += tau

    #print(f"End tn = {tn}\n")
    e1 = Norms.error_norm(integrator.x1n, lambda x: ex_solution.x1_expr(x, tn), "L2")
    e2 = Norms.error_norm(integrator.x2n, lambda x: ex_solution.x2_expr(x, tn), "H10")
    e3 = Norms.error_norm(integrator.thetan, lambda x: ex_solution.theta_expr(x, tn), "L2")

    return mesh_size_actual, np.sqrt(e1**2 + e2**2 + e3**2)

def sweep(hs, degs, tau: float, heat_offset: int, prefix: str) -> None:
    """Sweep h for every degree in `degs` and write the CSVs and the figure.

    The heat space runs `heat_offset` degrees below the wave spaces.  Output
    file names are prefixed with `prefix`.
    """

    plt.figure()

    for deg in degs:
        degree_wave = deg
        degree_heat = deg - heat_offset

        errs = []
        for h in hs:
            _, err = run(h, degree_wave=degree_wave, degree_heat=degree_heat, tau=tau)
            errs.append(err)

        errs = np.array(errs)

        plt.loglog(hs, errs, marker="*", label=f"{deg}")
        np.savetxt(
            f"output_h_plot/{prefix}deg_{deg}_data.txt", 
            np.array([hs,errs]).transpose(),
            header="h,err",
            comments="",
            delimiter=',')

    #plt.ylim(1e-10,1.)
    plt.legend()
    plt.grid()
    plt.savefig(f"output_h_plot/{prefix}error.png")

def main() -> None:
    """Run both h sweeps."""
    os.makedirs("output_h_plot", exist_ok=True)
    
    tau = 0.001
    hs = np.geomspace(0.1,0.01, 40)

    # equal degrees in the wave and heat spaces
    sweep(hs, [1,2,3,4], tau, heat_offset=0, prefix="")
    # heat space one degree below the wave spaces
    sweep(hs, [2,3,4], tau, heat_offset=1, prefix="miss_matched_")


if __name__ == "__main__":
    main()