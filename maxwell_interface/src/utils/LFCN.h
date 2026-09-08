#ifndef TIME_INTEGRATION_LFCN_H_
#define TIME_INTEGRATION_LFCN_H_

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>

#include "TimeIntegrator.h"
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_matrix_ez.h>
#include <deal.II/lac/block_sparse_matrix_ez.h>
#include <deal.II/lac/sparse_direct.h>

#include "ODESystem.h"
#include "ODESystemOperators.h"

using namespace dealii;

// Leapfrog with central fluxes for the H/E system, coupled to the interface
// ODE assembled by ODESystemOperators.
// NOTE: This is not the most efficient implementation.
// A more efficient implementation might be archived by reducing the ODE systems with a block ordering
// of the individual aux fields. See dissertation for details
template <typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype = Vector<double>>
class LFCN
{
public:
  LFCN(
      const MassMatrixtype &MH_inv,
      const MassMatrixtype &ME_inv,
      const CurlMatrixtype &CH,
      const CurlMatrixtype &CE,
      const ODESystem &ode_system,
      const dealii::BlockSparseMatrixEZ<double> &ode_mass_matrix,
      const dealii::BlockSparseMatrixEZ<double> &ode_inv_mass_matrix,
      const dealii::BlockSparseMatrixEZ<double> &maxwell_ode_coupling_matrix,
      const dealii::BlockSparseMatrixEZ<double> &ode_maxwell_coupling_matrix,
      double timestep,
      ODESystemOperators &ode_system_operators);

  // One full step: half-step E, full step H, coupled ODE step, half-step E.
  void integrate_step(
	  Vectortype &H,
	  Vectortype &E,
	  dealii::BlockVector<double> &ode_solution,
	  Vectortype &j_current_0,
	  Vectortype &j_current_1,
	  dealii::BlockVector<double> &g_0,
	  dealii::BlockVector<double> &g_1);

  void set_timestep(double timestep);

  // Discards the carried-over E half-step so it is rebuilt on the next call.
  void reset_initial_step();

private:
  const MassMatrixtype *MH_inverse;
  const MassMatrixtype *ME_inverse;
  const CurlMatrixtype *CH;
  const CurlMatrixtype *CE;

  const ODESystem &ode_system;

  const dealii::BlockSparseMatrixEZ<double> &ode_mass_matrix;
  const dealii::BlockSparseMatrixEZ<double> &ode_inv_mass_matrix;
  const dealii::BlockSparseMatrixEZ<double> &maxwell_ode_coupling_matrix;
  const dealii::BlockSparseMatrixEZ<double> &ode_maxwell_coupling_matrix;

  double timestep;

  dealii::BlockSparseMatrixEZ<double> R_minus_op;
  dealii::BlockSparseMatrixEZ<double> R_plus_op;

  // UMFPACK cannot factorize a BlockSparseMatrixEZ, so R_plus_op is flattened
  // into a plain SparseMatrixEZ before the solve.
  dealii::SparseMatrixEZ<double> R_plus_flat;

  SparseDirectUMFPACK direct;

  Vectortype E_half;
  Vectortype tmp_H;
  Vectortype tmp_E;
  dealii::BlockVector<double> tmp_maxwell_block;
  dealii::BlockVector<double> tmp_coupling;

  bool initial_step;

  ODESystemOperators &ode_system_operators;
};

#endif // TIME_INTEGRATION_LFCN_H_
