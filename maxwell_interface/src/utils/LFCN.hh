#ifndef TIME_INTEGRATION_LFCN_HH_
#define TIME_INTEGRATION_LFCN_HH_

#include "LFCN.h"

template <typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
LFCN<MassMatrixtype, CurlMatrixtype, Vectortype>::LFCN(
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
	ODESystemOperators &ode_system_operators)
	: MH_inverse(&MH_inv),
	  ME_inverse(&ME_inv),
	  CH(&CH),
	  CE(&CE),
	  ode_system(ode_system),
	  ode_mass_matrix(ode_mass_matrix),
	  ode_inv_mass_matrix(ode_inv_mass_matrix),
	  maxwell_ode_coupling_matrix(maxwell_ode_coupling_matrix),
	  ode_maxwell_coupling_matrix(ode_maxwell_coupling_matrix),
	  timestep(timestep),
	  initial_step(true),
	  ode_system_operators(ode_system_operators){
	  }

template <typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LFCN<MassMatrixtype, CurlMatrixtype, Vectortype>::integrate_step(
	Vectortype &H,
	Vectortype &E,
	dealii::BlockVector<double> &ode_solution,
	Vectortype &j_current_0,
	Vectortype &j_current_1,
	dealii::BlockVector<double> &g_0,
	dealii::BlockVector<double> &g_1)
{
	if (initial_step)
	{
		R_minus_op = 0;
		R_plus_op = 0;

		ode_system_operators.setup_prop_matrices(R_minus_op, R_plus_op);
		ode_system_operators.assemble_prop_matrices(R_minus_op, R_plus_op, timestep);

		// Block ordering here matches the BlockVector concatenation.
		{
			const unsigned int n_br = R_plus_op.n_block_rows();
			const unsigned int n_bc = R_plus_op.n_block_cols();
			std::vector<unsigned int> row_off(n_br + 1, 0), col_off(n_bc + 1, 0);
			for (unsigned int i = 0; i < n_br; ++i)
				row_off[i + 1] = row_off[i] + R_plus_op.block(i, 0).m();
			for (unsigned int j = 0; j < n_bc; ++j)
				col_off[j + 1] = col_off[j] + R_plus_op.block(0, j).n();

			R_plus_flat.clear();
			R_plus_flat.reinit(row_off[n_br], col_off[n_bc]);
			for (unsigned int i = 0; i < n_br; ++i)
				for (unsigned int j = 0; j < n_bc; ++j)
				{
					const auto &blk = R_plus_op.block(i, j);
					for (unsigned int r = 0; r < blk.m(); ++r)
						for (auto it = blk.begin(r); it != blk.end(r); ++it)
							R_plus_flat.set(row_off[i] + r, col_off[j] + it->column(),
								it->value());
				}
		}

		direct.initialize(R_plus_flat);

		tmp_maxwell_block.reinit(2);
		tmp_maxwell_block.block(0).reinit(H);
		tmp_maxwell_block.block(1).reinit(E);
		tmp_maxwell_block.collect_sizes();

		tmp_H.reinit(E);
		tmp_E.reinit(H);
		E_half.reinit(E);
		
		tmp_coupling.reinit(ode_solution);

		// Staggered start: build the first E half-step.
		(*CH).vmult(tmp_H, H);
		ode_maxwell_coupling_matrix.vmult(tmp_maxwell_block, ode_solution);

		tmp_H.sadd(timestep * 0.5, -timestep * 0.5, 
			tmp_maxwell_block.block(1));
		tmp_H.add(timestep * 0.5, 
			j_current_0);

		(*ME_inverse).vmult_add(E_half, tmp_H);
		E_half += E;

		initial_step = false;
	}
	else
		// Reuse the staggered value from the previous step: E_half = 2E - E_half.
		E_half.sadd(-1., 2., E);

	// Full step for H.
	(*CE).vmult(tmp_E, E_half);
	tmp_E *= -1.0 * timestep;
	// ode_maxwell_coupling_matrix.vmult(tmp_maxwell_block, ode_solution);
	// tmp_E += tmp_maxwell_block.block(0);
	(*MH_inverse).vmult_add(H, tmp_E);

	// Coupled ODE step, driven by the E half-step.
	tmp_maxwell_block.block(0) = 0;
	// Borrow E_half as the E block of the coupling input.
	tmp_maxwell_block.block(1).swap(E_half);

	maxwell_ode_coupling_matrix.vmult(tmp_coupling, tmp_maxwell_block);
	tmp_coupling *= timestep;
	tmp_coupling.add(0.5*timestep, g_0);
	tmp_coupling.add(0.5*timestep, g_1);
	R_minus_op.vmult_add(tmp_coupling, ode_solution);
	direct.vmult(ode_solution, tmp_coupling);

	tmp_maxwell_block.block(1).swap(E_half);

	// Second E half-step.
	(*CH).vmult(tmp_H, H);
	ode_maxwell_coupling_matrix.vmult(tmp_maxwell_block, ode_solution);

	tmp_H.sadd(timestep * 0.5, -timestep * 0.5, tmp_maxwell_block.block(1));
	tmp_H.add(timestep * 0.5, j_current_1);

	(*ME_inverse).vmult(E, tmp_H);
	E += E_half;

}

template <typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LFCN<MassMatrixtype, CurlMatrixtype, Vectortype>::set_timestep(double timestep)
{
	this->timestep = timestep;

	initial_step = true;
}

template <typename MassMatrixtype, typename CurlMatrixtype, typename Vectortype>
void LFCN<MassMatrixtype, CurlMatrixtype, Vectortype>::reset_initial_step()
{
	initial_step = true;
}

#endif // TIME_INTEGRATION_LFCN_HH_
