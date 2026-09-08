#!/usr/bin/env bash
# Temporal convergence / CFL experiment of the paper (thermo-elastic wave coupling).
#
# Polynomial profile x1(1-x1)x2(1-x2) with a(t) = cos(omega t), c(t) = sin(eta t),
# resolved exactly by the second-order ansatz spaces, on 10 meshes with target
# mesh sizes from 0.2 to 0.02. The paper does not fix omega and eta; both are
# 2 pi here.
#
#   ./run_tau_plot.sh            # 4 ranks
#   NP=8 ./run_tau_plot.sh       # 8 ranks
#   ./run_tau_plot.sh --ntau 8   # extra arguments override the ones below
set -eu

mpirun -n "${NP:-4}" python3 driver_timestep_experiment.py \
    --n_lo 5 --n_hi 50 --n_num 10 \
    --ntau 20 --tau_lo 1e-4 --tau_hi 3e-2 \
    -k 2 -T 1.0 \
    --omega 6.283185307179586 --eta 6.283185307179586 \
    --rho 1.1 --mu 2.32 --lam 1.42 --kappa 4.2 --alpha 1.0 \
    -o output_tau_plot "$@"
