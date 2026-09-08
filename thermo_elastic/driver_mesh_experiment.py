#!/usr/bin/env python3
"""Convergence in space: total L2 error at T against mesh size h, at a fixed tau.

One curve per (wave degree, heat degree, wave number) combination; see README.md
for the parameters and the time-step choice.

Usage::

    mpirun -n 4 python3 driver_mesh_experiment.py --n_lo 4 --n_hi 64
"""
import argparse
import csv
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI

from Integrator.Forms import Material
from Data.TrigSolution import TrigSolution
from Util.Meshing import MeshGeometry, generate_square_mesh
from Util.Util import get_h_min


def _degree_pair(s):
    """Parse a 'k,l' string into a (wave_degree, heat_degree) integer tuple."""
    k, l = s.split(",")
    return (int(k), int(l))


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # mesh-resolution interval (log-spaced)
    ap.add_argument("--n_lo", type=int, default=4,
                    help="coarsest mesh resolution (cells per direction)")
    ap.add_argument("--n_hi", type=int, default=64,
                    help="finest mesh resolution (cells per direction)")
    ap.add_argument("--n_num", type=int, default=5,
                    help="number of meshes, log-spaced over [n_lo, n_hi]")
    ap.add_argument("--wave-degrees", type=int, nargs="+", default=[1, 2],
                    help="polynomial degrees k for the wave part (v, sigma) to test")
    ap.add_argument("--heat-degrees", type=int, nargs="+", default=None,
                    help="polynomial degrees l for the heat part (theta); "
                         "defaults to the wave degrees (k=l)")
    ap.add_argument("--degree-pairs", type=_degree_pair, nargs="+", default=None,
                    help="explicit (wave,heat) degree pairs, e.g. '2,1 3,2'; overrides "
                         "--wave-degrees/--heat-degrees (no cross product)")
    ap.add_argument("--wavenumbers", type=int, nargs="+", default=[1],
                    help="spatial wave numbers (positive integers); one curve per value")
    # time step (single fixed value for all runs)
    ap.add_argument("--cfl", type=float, default=0.8,
                    help="fraction of the measured CFL stability limit to use for "
                         "the fixed time step tau = cfl * nu_lim(p_max) * h_min / c_P "
                         "(cfl=1 is at the limit; default 0.8 leaves a 20%% margin)")
    ap.add_argument("--tau", type=float, default=None,
                    help="explicit fixed time step (overrides the CFL formula)")
    ap.add_argument("-T", type=float, default=1.0, help="final time")
    # material parameters
    ap.add_argument("--rho", type=float, default=1.0, help="density")
    ap.add_argument("--mu", type=float, default=1.0, help="Lame mu")
    ap.add_argument("--lam", type=float, default=1.0, help="Lame lambda")
    ap.add_argument("--kappa", type=float, default=1.0, help="thermal diffusivity")
    ap.add_argument("--alpha", type=float, default=1.0, help="thermo-elastic coupling")
    ap.add_argument("-o", "--outdir", default="output_mesh_experiment")
    return ap.parse_args()


def log_spaced_resolutions(n_lo, n_hi, n_num):
    """Distinct integer resolutions, log-spaced over [n_lo, n_hi]."""
    raw = np.geomspace(n_lo, n_hi, n_num)
    ns = sorted(set(int(round(x)) for x in raw))
    return ns


# Maximum stable Courant number  nu = tau * c_P / h_max  of the leapfrog wave step,
# measured by bisection on the crossed-triangle mesh (see README.md).
NU_LIMIT = {1: 0.118, 2: 0.062, 3: 0.040}


def courant_limit(p):
    """Max stable Courant number for degree ``p`` (measured table, ~0.12/p beyond)."""
    return NU_LIMIT.get(p, 0.12 / p)


def fmt_dur(seconds):
    """Human-readable duration."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def work_weight(deg, n):
    """Cost proxy for one run, cells (~n^2) times dofs per cell (~(deg+1)^2).
    Weights the ETA estimate; the step count is the same for every run."""
    return float(n) ** 2 * float(deg + 1) ** 2


def main():
    args = parse_args()
    rank0 = MPI.COMM_WORLD.rank == 0
    mat = Material(rho=args.rho, mu=args.mu, lam=args.lam,
                   alpha=args.alpha, kappa=args.kappa)

    ns = log_spaced_resolutions(args.n_lo, args.n_hi, args.n_num)
    if args.degree_pairs is not None:
        pairs = list(args.degree_pairs)            # explicit (wave, heat) pairs
    else:
        heat_degrees = (args.heat_degrees if args.heat_degrees is not None
                        else args.wave_degrees)
        pairs = [(k, l) for k in args.wave_degrees for l in heat_degrees]
    # the CFL limit is set by the wave part -> size tau from the largest wave degree
    p_max = max(k for k, _ in pairs)

    # single fixed tau, sized from the smallest element of the finest mesh and the
    # highest wave degree (the binding CFL constraint)
    fine_mesh = generate_square_mesh(
        MeshGeometry(n=ns[-1], cell_type=TrigSolution.cell_type)).mesh
    h_cfl = get_h_min(fine_mesh)
    nu_lim = courant_limit(p_max)
    if args.tau is not None:
        tau = args.tau
    else:
        tau = args.cfl * nu_lim * h_cfl / mat.c_P

    if rank0:
        print(f"resolutions (log-spaced): {ns}")
        print(f"(wave, heat) degree pairs: {pairs}")
        print(f"material: rho={mat.rho} mu={mat.mu} lam={mat.lam} "
              f"kappa={mat.kappa} alpha={mat.alpha}  (c_P={mat.c_P:.4f})")
        print(f"fixed tau = {tau:.4e}  "
              f"(h_cfl={h_cfl:.4e}, p_max={p_max}, nu_lim={nu_lim:.3f}, "
              f"cfl={args.cfl} of limit, {int(np.ceil(args.T/tau))} steps/run)\n")

    # one curve per (wave degree, heat degree, wave number) combination
    curves = [(k, l, kw) for (k, l) in pairs for kw in args.wavenumbers]
    vary_k = len({k for k, _ in pairs}) > 1
    vary_l = len({l for _, l in pairs}) > 1
    vary_kw = len(args.wavenumbers) > 1

    # progress / ETA bookkeeping (weighted by per-experiment cost)
    total_count = len(curves) * len(ns)
    total_weight = sum(work_weight(max(k, l), n) for k, l, kw in curves for n in ns)
    done_count = 0
    done_weight = 0.0
    t_start = time.perf_counter()

    results = {}   # (k, l, kw) -> (hs, totals)
    for k, l, kw in curves:
        sol = TrigSolution(mat, mode="space", k=kw)   # temporally exact -> pure spatial error
        hs, totals = [], []
        for n in ns:
            h, ev, es, et = sol.compute_errors(n, tau, k, l, args.T)
            tot = float(np.sqrt(ev**2 + es**2 + et**2))
            hs.append(h)
            totals.append(tot)
            done_count += 1
            done_weight += work_weight(max(k, l), n)
            if rank0:
                elapsed = time.perf_counter() - t_start
                remaining = elapsed * (total_weight - done_weight) / done_weight
                finish = datetime.now() + timedelta(seconds=remaining)
                print(f"k={k} l={l} wave={kw} n={n:4d} h={h:.4f} tau={tau:.2e}  "
                      f"err_v={ev:.3e} err_sig={es:.3e} err_th={et:.3e}  total={tot:.3e}")
                print(f"    [{done_count}/{total_count}  {100*done_weight/total_weight:5.1f}%  "
                      f"elapsed {fmt_dur(elapsed)}  ETA {fmt_dur(remaining)}  "
                      f"finish ~{finish:%H:%M:%S}]", flush=True)
        results[(k, l, kw)] = (np.array(hs), np.array(totals))

    if not rank0:
        return
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # record the exact invocation for reproducibility
    (outdir / "call_string.txt").write_text("mpirun python3 " + " ".join(sys.argv) + "\n")

    def col_name(k, l, kw):
        return f"total_k{k}_l{l}_w{kw}"

    def label(k, l, kw, eoc):
        parts = []
        if vary_k or vary_l or not vary_kw:
            if k == l:
                parts.append(rf"$k=\ell={k}$")
            else:
                parts.append(rf"$k={k},\ \ell={l}$")
        if vary_kw or not (vary_k or vary_l):
            parts.append(rf"wave no. ${kw}$")
        return ", ".join(parts) + rf"  (EOC$\approx{eoc:.2f}$)"

    # ---- CSV: h, one column per (wave degree, heat degree, wave number) ------
    h_ref = results[curves[0]][0]
    with open(outdir / "data.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["h"] + [col_name(k, l, kw) for k, l, kw in curves])
        for i in range(len(h_ref)):
            row = [f"{h_ref[i]:.6e}"]
            for k, l, kw in curves:
                v = results[(k, l, kw)][1][i]
                row.append(f"{v:.6e}" if np.isfinite(v) else "nan")
            w.writerow(row)

    # ---- plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for k, l, kw in curves:
        hs, totals = results[(k, l, kw)]
        m = np.isfinite(totals)
        hs, totals = hs[m], totals[m]
        # least-squares slope of log(error) vs log(h)
        eoc = float(np.polyfit(np.log(hs), np.log(totals), 1)[0]) if len(hs) > 1 else float("nan")
        ax.loglog(hs, totals, marker="o", label=label(k, l, kw, eoc))
        order = min(k, l) + 1   # total error converges at order min(k,l)+1
        ref = totals[-1] * (hs / hs[-1]) ** order   # h^{min(k,l)+1} guide
        ax.loglog(hs, ref, ls="--", color="0.6", lw=1)
    ax.set_xlabel("$h$")
    ax.set_ylabel(r"$\sqrt{\|e_v\|_{L^2}^2+\|e_\sigma\|_{L^2}^2+\|e_\theta\|_{L^2}^2}$")
    ax.set_title("Mesh-size experiment (trigonometric solution)")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "plot.png", dpi=150)
    print(f"\nwrote {outdir/'plot.png'} and {outdir/'data.csv'} "
          f"(dashed grey lines are h^(min(k,l)+1) guides)")


if __name__ == "__main__":
    main()
