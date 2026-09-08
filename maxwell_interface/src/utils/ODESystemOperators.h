#ifndef ODE_SYSTEM_OPERATORS_H_
#define ODE_SYSTEM_OPERATORS_H_

#include "deal.II/base/function.h"

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include "deal.II/fe/fe_values.h"
#include <deal.II/fe/fe_dgq.h>
#include "deal.II/fe/fe_values_extractors.h"
#include "deal.II/dofs/dof_handler.h"

#include "deal.II/lac/block_vector.h"
#include "deal.II/lac/block_sparsity_pattern.h"
#include "deal.II/lac/block_sparse_matrix.h"
#include "deal.II/lac/sparse_matrix.h"
#include "deal.II/lac/sparse_matrix_ez.h"
#include "deal.II/lac/block_sparse_matrix_ez.h"
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include "ODESystem.h"

// Assembles the interface ODE operators (mass, Maxwell<->ODE coupling and
// the propagation matrices R_-, R_+) from the codim-1 interface mesh and the
// volume DoFHandler.
class ODESystemOperators
{
private:

    dealii::FESystem<1,2> &int_fe;
    const dealii::Quadrature<1> &int_quadrature;
    dealii::DoFHandler<1,2> &int_dof_handler;

    dealii::FESystem<2> &vol_fe;
    dealii::DoFHandler<2> &vol_dof_handler;

    using FACE_TO_VOLUME_FACE_TYPE = 
        std::map<
            typename dealii::Triangulation<1, 2>::cell_iterator,
            typename dealii::Triangulation<2, 2>::face_iterator>;
    
    const FACE_TO_VOLUME_FACE_TYPE &interface_cell_to_face_map;

    using VOLUME_FACE_TO_CELLS_TYPE =
        std::map<
            dealii::Triangulation<2>::face_iterator,
            std::pair<
                dealii::Triangulation<2>::active_cell_iterator,
                dealii::Triangulation<2>::active_cell_iterator>>;

    const VOLUME_FACE_TO_CELLS_TYPE &volume_face_to_volume_cells_map;

    const ODESystem &ode_system;

    dealii::FEValues<1, 2> int_fe_v;
    dealii::FEFaceValues<2> vol_fe_v_face;
    dealii::FEFaceValues<2> vol_fe_v_face_neighbor;

public:
    ODESystemOperators(
        dealii::FESystem<1,2> &int_fe,
        const dealii::Quadrature<1> &int_quadrature,
        dealii::DoFHandler<1,2> &int_dof_handler,
        dealii::FESystem<2> &vol_fe,
        dealii::DoFHandler<2> &vol_dof_handler,
        const FACE_TO_VOLUME_FACE_TYPE &map1,
        const VOLUME_FACE_TO_CELLS_TYPE &map2,
        const ODESystem &ode_system)
        :
        int_fe{int_fe},
        int_quadrature{int_quadrature},
        int_dof_handler{int_dof_handler},
        vol_fe{vol_fe},
        vol_dof_handler{vol_dof_handler},
        interface_cell_to_face_map{map1},
        volume_face_to_volume_cells_map{map2},
        ode_system{ode_system},
        int_fe_v{
            int_fe, 
            int_quadrature, 
            dealii::UpdateFlags::update_quadrature_points
                | dealii::UpdateFlags::update_transformation_values
                | dealii::UpdateFlags::update_JxW_values
                | dealii::UpdateFlags::update_values},
        vol_fe_v_face{
            vol_fe,
            int_quadrature,
            dealii::UpdateFlags::update_values
                | dealii::UpdateFlags::update_normal_vectors
                | dealii::UpdateFlags::update_quadrature_points},
        vol_fe_v_face_neighbor{
            vol_fe,
            int_quadrature,
            dealii::UpdateFlags::update_values
                | dealii::UpdateFlags::update_quadrature_points} {};
    
    void setup_matrices(
        dealii::BlockSparseMatrixEZ<double> &ode_mass_matrix,
        dealii::BlockSparseMatrixEZ<double> &ode_inv_mass_matrix,
        dealii::BlockSparseMatrixEZ<double> &maxwell_ode_coupling_matrix,
        dealii::BlockSparseMatrixEZ<double> &ode_maxwell_coupling_matrix);

    void setup_prop_matrices(
        dealii::BlockSparseMatrixEZ<double> &R_minus_op,
        dealii::BlockSparseMatrixEZ<double> &R_plus_op
    );

    void assemble_ode_mass_matrix(
        dealii::BlockSparseMatrixEZ<double> &ode_mass_matrix,
        dealii::BlockSparseMatrixEZ<double> &ode_inv_mass_matrix);

    void assemble_maxwell_to_ode_coupling_matrix(
        dealii::BlockSparseMatrixEZ<double> &maxwell_ode_coupling_matrix);

    void assemble_ode_to_maxwell_coupling_matrix(
        dealii::BlockSparseMatrixEZ<double> &ode_maxwell_coupling_matrix);

    void assemble_prop_matrices(
        dealii::BlockSparseMatrixEZ<double> &R_minus_op,
        dealii::BlockSparseMatrixEZ<double> &R_plus_op,
        double timestep_size
    );
};

#endif // ODE_SYSTEM_OPERATORS_H_