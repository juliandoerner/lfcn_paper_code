#!/usr/bin/env bash
# Spatial convergence experiment of the paper (thermo-elastic wave coupling).
#
# Cavity profile sin(2 pi x1) sin(2 pi x2) with a(t) = 1, c(t) = t, on 15 meshes
# with target mesh sizes from 0.2 to 0.02. Two degree sets: k = l = p for
# p in {1, 2, 3}, and k = l + 1 = p for p in {2, 3}.
#
#   ./run_h_plot.sh              # 4 ranks
#   NP=8 ./run_h_plot.sh         # 8 ranks
#   ./run_h_plot.sh --n_num 5    # extra arguments override the ones below
set -eu

mpirun -n "${NP:-4}" python3 driver_mesh_experiment.py \
    --n_lo 5 --n_hi 50 --n_num 15 \
    --degree-pairs 1,1 2,2 3,3 2,1 3,2 \
    --wavenumbers 2 \
    -T 1.0 \
    --rho 1.1 --mu 2.32 --lam 1.42 --kappa 4.2 --alpha 1.0 \
    -o output_h_plot "$@"
