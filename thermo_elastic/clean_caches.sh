#!/usr/bin/env bash
# Clean the two caches that cause trouble between/after MPI runs:
#   1. the FEniCSx (FFCx) JIT cache -- a stale/partial entry from an interrupted
#      compile makes the next run fail with "JIT compilation timed out, probably
#      due to a failed previous compile".
#   2. stale MPICH shared-memory segments in /dev/shm left by killed `mpirun`s --
#      they fill the (small) /dev/shm and cause SIGBUS (bus error) on later runs.
#
# Safe to run any time *no* MPI job is currently running.
set -u

cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/fenics"

echo "==> removing FFCx JIT cache: $cache_dir"
rm -rf "$cache_dir"

echo "==> removing stale MPICH shared-memory segments: /dev/shm/mpich_shm_*"
rm -f /dev/shm/mpich_shm_* 2>/dev/null

echo "done."
df -h /dev/shm 2>/dev/null | tail -1
