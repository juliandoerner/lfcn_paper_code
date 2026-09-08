#!/usr/bin/env python3
"""Convergence in time and CFL threshold: total L2 error at T against the step
size tau, one curve per mesh.

Unstable runs are kept and drawn above the clipped y-axis, so the CFL threshold
shows up as a vertical rise; see README.md.

Usage::

    mpirun -n 4 python3 driver_timestep_experiment.py --n_lo 8 --n_hi 48
"""
import argparse
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI

from Integrator.Forms import Material
from Data.PolySolution import PolySolution

PI = np.pi


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # mesh-resolution interval (log-spaced); one curve per mesh
    ap.add_argument("--n_lo", type=int, default=8,
                    help="coarsest mesh resolution (cells per direction)")
    ap.add_argument("--n_hi", type=int, default=48,
                    help="finest mesh resolution (cells per direction)")
    ap.add_argument("--n_num", type=int, default=4,
                    help="number of meshes, log-spaced over [n_lo, n_hi]")
    ap.add_argument("--omega", type=float, default=2 * PI,
                    help="temporal frequency of the wave profile cos(omega t)")
    ap.add_argument("--eta", type=float, default=None,
                    help="heat-profile frequency sin(eta t) (default: eta=omega)")
    ap.add_argument("--taus", type=float, nargs="+", default=None,
                    help="explicit tau values (overrides the geometric sweep)")
    ap.add_argument("--ntau", type=int, default=22, help="number of tau values")
    ap.add_argument("--tau_lo", type=float, default=4e-4)
    ap.add_argument("--tau_hi", type=float, default=4e-2)
    ap.add_argument("-k", type=int, default=2, help="degree (k=l), must be >=2 (Q2 exact)")
    ap.add_argument("-T", type=float, default=0.6, help="final time")
    # material parameters
    ap.add_argument("--rho", type=float, default=1.0, help="density")
    ap.add_argument("--mu", type=float, default=1.0, help="Lame mu")
    ap.add_argument("--lam", type=float, default=1.0, help="Lame lambda")
    ap.add_argument("--kappa", type=float, default=1.0, help="thermal diffusivity")
    ap.add_argument("--alpha", type=float, default=1.0, help="thermo-elastic coupling")
    ap.add_argument("--ycut", type=float, default=1.0,
                    help="y-axis top (10^0 by default); an error >= ycut (or a "
                         "diverged run) counts as unstable -> written as 10*ycut in "
                         "csv, line rises off the top in the plot")
    ap.add_argument("--max_unstable_streak", type=int, default=3,
                    help="after this many consecutive unstable taus, mark all "
                         "remaining (larger) taus for the mesh unstable without "
                         "computing them (CFL stability is monotonic in tau)")
    ap.add_argument("-o", "--outdir", default="output_timestep_experiment")
    return ap.parse_args()


def log_spaced_resolutions(n_lo, n_hi, n_num):
    """Distinct integer resolutions, log-spaced over [n_lo, n_hi]."""
    return sorted(set(int(round(x)) for x in np.geomspace(n_lo, n_hi, n_num)))


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


def work_weight(n, tau, k, T):
    """Cost proxy for one run: cells (~n^2) * dofs/cell (~(k+1)^2) * steps (~T/tau).
    Diverging runs stop early, so the ETA built from it is conservative."""
    return float(n) ** 2 * float(k + 1) ** 2 * (T / tau)


def main():
    args = parse_args()
    rank0 = MPI.COMM_WORLD.rank == 0
    taus = (np.array(args.taus, dtype=float) if args.taus
            else np.geomspace(args.tau_lo, args.tau_hi, args.ntau))
    taus = np.sort(taus)                      # ascending tau for clean plotting
    ns = log_spaced_resolutions(args.n_lo, args.n_hi, args.n_num)
    eta = args.eta if args.eta is not None else args.omega
    mat = Material(rho=args.rho, mu=args.mu, lam=args.lam,
                   alpha=args.alpha, kappa=args.kappa)
    sol = PolySolution(mat, mode="time", omega=args.omega, eta=eta)

    if rank0:
        print(f"resolutions (log-spaced): {ns}")
        print(f"material: rho={mat.rho} mu={mat.mu} lam={mat.lam} "
              f"kappa={mat.kappa} alpha={mat.alpha}  (c_P={mat.c_P:.4f})")
        print(f"profile: omega={args.omega:.4f}  eta={eta:.4f}  T={args.T}  k=l={args.k}")
        print(f"tau sweep: {len(taus)} values in [{taus.min():.3e}, {taus.max():.3e}]\n")

    # progress / ETA bookkeeping (weighted by per-run cost ~ n^2 (k+1)^2 / tau)
    total_count = len(ns) * len(taus)
    total_weight = sum(work_weight(n, tau, args.k, args.T) for n in ns for tau in taus)
    done_count = 0
    done_weight = 0.0
    t_start = time.perf_counter()

    results = {}   # n -> (h, totals[len(taus)])  (totals = inf where diverged/skipped)
    for n in ns:
        h = None
        totals = []
        streak = 0
        skipping = False                 # set once the CFL limit is confidently passed
        for j, tau in enumerate(taus):
            done_count += 1
            done_weight += work_weight(n, tau, args.k, args.T)
            if skipping:
                totals.append(np.inf)    # remaining larger taus are unstable, not run
                continue
            h, ev, es, et = sol.compute_errors(n, tau, args.k, args.k, args.T)
            tot = float(np.sqrt(ev**2 + es**2 + et**2)) if np.isfinite(ev) else np.inf
            totals.append(tot)
            unstable = (not np.isfinite(tot)) or (tot >= args.ycut)
            streak = streak + 1 if unstable else 0
            if rank0:
                elapsed = time.perf_counter() - t_start
                remaining = elapsed * (total_weight - done_weight) / done_weight
                finish = datetime.now() + timedelta(seconds=remaining)
                print(f"n={n:3d} h={h:.4f} tau={tau:.3e}  total={tot:.3e}  "
                      f"[{'unstable' if unstable else 'ok'}]"
                      f"   [{done_count}/{total_count}  {100*done_weight/total_weight:5.1f}%"
                      f"  elapsed {fmt_dur(elapsed)}  ETA {fmt_dur(remaining)}"
                      f"  finish ~{finish:%H:%M:%S}]", flush=True)
            if streak >= args.max_unstable_streak:
                skipping = True
                n_left = len(taus) - (j + 1)
                if rank0 and n_left:
                    print(f"    -> {streak} unstable in a row; marking remaining "
                          f"{n_left} larger tau(s) unstable, moving to next mesh",
                          flush=True)
        results[n] = (h, np.array(totals))

    if not rank0:
        return
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sentinel = 10.0 * args.ycut          # value written/plotted for unstable taus

    def is_stable(v):
        return np.isfinite(v) and v < args.ycut

    def csv_value(v):
        return f"{v:.6e}" if is_stable(v) else f"{sentinel:.6e}"

    # ---- CSV: tau, one column per mesh (unstable -> 10*ycut) ---------------
    with open(outdir / "data.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tau"] + [f"n{n}" for n in ns])
        for j, tau in enumerate(taus):
            w.writerow([f"{tau:.6e}"] + [csv_value(results[n][1][j]) for n in ns])
    with open(outdir / "curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["column", "n", "h", "omega", "eta"])
        for n in ns:
            w.writerow([f"n{n}", n, f"{results[n][0]:.6e}", f"{args.omega:.6e}", f"{eta:.6e}"])

    # ---- plot --------------------------------------------------------------
    # unstable taus are drawn at ``sentinel`` (above ``ycut``), so the curve rises
    # off the top of the clipped axis instead of being dropped
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    stable_min = np.inf
    all_stable = []
    for n in ns:
        h, totals = results[n]
        stable = np.array([is_stable(v) for v in totals])
        y = np.where(stable, totals, sentinel)
        ax.loglog(taus, y, marker="o", label=rf"$h\approx{h:.3f}$ ($n={n}$)")
        if stable.any():
            stable_min = min(stable_min, float(totals[stable].min()))
            all_stable += [(taus[j], totals[j]) for j in range(len(taus)) if stable[j]]
    # tau^2 reference, anchored at the most-converged stable point
    if all_stable:
        ta = np.array([t for t, _ in all_stable])
        ea = np.array([e for _, e in all_stable])
        i0 = int(np.argmin(ea))
        C = ea[i0] / ta[i0] ** 2
        tt = np.array([taus.min(), taus.max()])
        ax.loglog(tt, C * tt ** 2, ls="--", color="k", lw=1, label=r"$\propto\tau^2$")
        ax.set_ylim(bottom=stable_min / 3.0, top=args.ycut)
    else:
        ax.set_ylim(top=args.ycut)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\sqrt{\|e_v\|_{L^2}^2+\|e_\sigma\|_{L^2}^2+\|e_\theta\|_{L^2}^2}$")
    ax.set_title(r"Time-step / CFL experiment (polynomial solution, $k=\ell=%d$)" % args.k)
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "plot.png", dpi=150)
    print(f"\nwrote {outdir/'plot.png'}, {outdir/'data.csv'} and {outdir/'curves.csv'}")


if __name__ == "__main__":
    main()
