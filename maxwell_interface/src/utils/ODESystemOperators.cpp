#include "ODESystemOperators.h"

#include <vector>

#include "deal.II/dofs/dof_tools.h"

#include <deal.II/dofs/dof_accessor.h>

void check_quadrature_point_correspondence(
    const dealii::FEValues<1, 2>     &int_fe_v,
    const dealii::FEFaceValues<2>    &vol_fe_v_face,
    const dealii::FEFaceValues<2>    &vol_fe_v_face_neighbor,
    const double                      tol = 1e-12)
{
    const auto &int_qp        = int_fe_v.get_quadrature_points();
    const auto &face_qp       = vol_fe_v_face.get_quadrature_points();
    const auto &face_qp_neigh = vol_fe_v_face_neighbor.get_quadrature_points();

    AssertThrow(int_qp.size() == face_qp.size()
                    && int_qp.size() == face_qp_neigh.size(),
                dealii::ExcMessage(
                    "Mismatch in number of quadrature points between "
                    "interface cell and volume faces."));

    for (unsigned int q = 0; q < int_qp.size(); ++q)
    {
        AssertThrow(int_qp[q].distance(face_qp[q]) < tol,
                    dealii::ExcMessage(
                        "Interface and volume-face quadrature points do not "
                        "coincide (own cell). Check interface mesh "
                        "orientation / parametrization."));
        AssertThrow(int_qp[q].distance(face_qp_neigh[q]) < tol,
                    dealii::ExcMessage(
                        "Interface and volume-face quadrature points do not "
                        "coincide (neighbor cell). Check face orientation."));
    }
}

void ODESystemOperators::setup_matrices(
        dealii::BlockSparseMatrixEZ<double> &ode_mass_matrix,
        dealii::BlockSparseMatrixEZ<double> &ode_inv_mass_matrix,
        dealii::BlockSparseMatrixEZ<double> &maxwell_ode_coupling_matrix,
        dealii::BlockSparseMatrixEZ<double> &ode_maxwell_coupling_matrix)
{
    // Maxwell DoFs
    std::vector<dealii::types::global_dof_index> dofs_per_block =
        dealii::DoFTools::count_dofs_per_fe_block(vol_dof_handler, {0, 1});
    
    const unsigned int n_dofs_H = dofs_per_block[0];
    const unsigned int n_dofs_E = dofs_per_block[1];

    // Interface DoFs
    auto ell = ode_system.system_size;
    std::vector<unsigned int> interface_blocks(ell);
    for (unsigned int i = 0; i < ell; ++i)
        interface_blocks[i] = i;

    std::vector<dealii::types::global_dof_index>
        dofs_per_interface_block
         = dealii::DoFTools::count_dofs_per_fe_block(
            int_dof_handler, interface_blocks);
    
    Assert(dofs_per_interface_block.size() 
        == ell, dealii::ExcImpossibleInDim(dofs_per_interface_block.size()));
    const unsigned int interface_dofs = dofs_per_interface_block[0];

    ode_mass_matrix.reinit(ell, ell);
    ode_inv_mass_matrix.reinit(ell, ell);
    for (unsigned int i = 0; i < ell; ++i)
    {
        for (unsigned int j = 0; j < ell; ++j)
        {
            ode_mass_matrix.block(i,j)
                .reinit(interface_dofs, interface_dofs);
            ode_inv_mass_matrix.block(i,j)
                .reinit(interface_dofs, interface_dofs);
        }
    }
    ode_mass_matrix.collect_sizes();
    ode_inv_mass_matrix.collect_sizes();

    // setup block
    maxwell_ode_coupling_matrix.reinit(ell,2);
    for (unsigned int i = 0; i < ell; ++i)
    {
        maxwell_ode_coupling_matrix.block(i,0)
            .reinit(interface_dofs, n_dofs_H);
        maxwell_ode_coupling_matrix.block(i,1)
            .reinit(interface_dofs, n_dofs_E);

    }
    maxwell_ode_coupling_matrix.collect_sizes();

    ode_maxwell_coupling_matrix.reinit(2,ell);
    for (unsigned int i = 0; i < ell; ++i)
    {
        ode_maxwell_coupling_matrix.block(0,i)
            .reinit(n_dofs_H, interface_dofs);
        ode_maxwell_coupling_matrix.block(1,i)
            .reinit(n_dofs_E, interface_dofs);
    }
    ode_maxwell_coupling_matrix.collect_sizes();

}

void ODESystemOperators::setup_prop_matrices(
        dealii::BlockSparseMatrixEZ<double> &R_minus_op,
        dealii::BlockSparseMatrixEZ<double> &R_plus_op)
{

    // Interface DoFs
    auto ell = ode_system.system_size;
    std::vector<unsigned int> interface_blocks(ell);
    for (unsigned int i = 0; i < ell; ++i)
        interface_blocks[i] = i;

    std::vector<dealii::types::global_dof_index>
        dofs_per_interface_block
         = dealii::DoFTools::count_dofs_per_fe_block(
            int_dof_handler, interface_blocks);
    
    Assert(dofs_per_interface_block.size() 
        == ell, dealii::ExcImpossibleInDim(dofs_per_interface_block.size()));
    const unsigned int interface_dofs = dofs_per_interface_block[0];

    R_minus_op.reinit(ell, ell);
    R_plus_op.reinit(ell, ell);
    for (unsigned int i = 0; i < ell; ++i)
    {
        for (unsigned int j = 0; j < ell; ++j)
        {
            R_minus_op.block(i,j)
                .reinit(interface_dofs, interface_dofs);
            R_plus_op.block(i,j)
                .reinit(interface_dofs, interface_dofs);
        }
    }
    R_minus_op.collect_sizes();
    R_plus_op.collect_sizes();

}

void ODESystemOperators::assemble_ode_mass_matrix(
        dealii::BlockSparseMatrixEZ<double> &ode_mass_matrix,
        dealii::BlockSparseMatrixEZ<double> &ode_inv_mass_matrix)
{   

    ode_mass_matrix = 0;
    ode_inv_mass_matrix = 0;

    auto dofs_per_cell = int_dof_handler.get_fe().n_dofs_per_cell();

    dealii::FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    auto cell = int_dof_handler.begin_active();
    auto endc = int_dof_handler.end();

    std::vector<dealii::types::global_dof_index> 
        local_dof_indices(dofs_per_cell);

    for(; cell != endc; ++cell)
    {
        cell_matrix = 0;

        int_fe_v.reinit(cell);

        const std::vector<double> &JxW = int_fe_v.get_JxW_values();

        auto n_quad_points = int_fe_v.n_quadrature_points;
        for (unsigned int q_point = 0; q_point < n_quad_points; ++q_point)
        {
            for (const unsigned int i : int_fe_v.dof_indices())
            {
                for (const unsigned int j : int_fe_v.dof_indices())
                {
                    const unsigned int comp_i 
                        = int_fe_v.get_fe().system_to_component_index(i).first;
                    const unsigned int comp_j 
                        = int_fe_v.get_fe().system_to_component_index(j).first;

                    if(comp_i != comp_j) continue;

                    const auto phi_i 
                        = int_fe_v.shape_value(i, q_point);
                    const auto phi_j 
                        = int_fe_v.shape_value(j, q_point);
                    
                    cell_matrix(i, j) += phi_i * phi_j * JxW[q_point];
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);

        for (const unsigned int i : int_fe_v.dof_indices())
        {
            for (const unsigned int j : int_fe_v.dof_indices())
            {
                ode_mass_matrix.add(
                    local_dof_indices[i],
                    local_dof_indices[j], 
                    cell_matrix(i,j));
            }
        }
        
        cell_matrix.gauss_jordan();

        for (const unsigned int i : int_fe_v.dof_indices())
            for (const unsigned int j : int_fe_v.dof_indices())
               ode_inv_mass_matrix.add(
                local_dof_indices[i],
                local_dof_indices[j], 
                cell_matrix(i,j));

    }
}

void ODESystemOperators::assemble_maxwell_to_ode_coupling_matrix(
        dealii::BlockSparseMatrixEZ<double> &maxwell_ode_coupling_matrix)
{   

    const dealii::FEValuesExtractors::Scalar H(0);
    const dealii::FEValuesExtractors::Vector E(1);

    maxwell_ode_coupling_matrix = 0;
    
    auto dofs_per_int_cell = int_dof_handler.get_fe().dofs_per_cell;
    auto dofs_per_cell = vol_fe.dofs_per_cell;

    dealii::FullMatrix<double> cell_matrix_int(
        dofs_per_int_cell, dofs_per_cell);
    dealii::FullMatrix<double> cell_matrix_ext(
        dofs_per_int_cell, dofs_per_cell);

    std::vector<dealii::types::global_dof_index> 
        local_dof_indices(dofs_per_int_cell);
    std::vector<dealii::types::global_cell_index>
        local_dof_indices_int(dofs_per_cell);
    std::vector<dealii::types::global_cell_index>
        local_dof_indices_ext(dofs_per_cell);

    for(const auto &cell : int_dof_handler.active_cell_iterators())
    {
        cell_matrix_int = 0;
        cell_matrix_ext = 0;

        // Interface cell -> volume face -> the two adjacent volume cells.
        auto it = interface_cell_to_face_map.find(cell);
        if (it == interface_cell_to_face_map.end())
            throw dealii::ExcInvalidState();
        const auto &volume_face = it->second;
        auto it_2 = volume_face_to_volume_cells_map.find(volume_face);
        if (it_2 == volume_face_to_volume_cells_map.end())
            throw dealii::ExcInvalidState();

        const auto &volume_cell = it_2->second.first;
        const auto &volume_cell_neighbor = it_2->second.second;

        typename dealii::DoFCellAccessor<2,2,false> dof_volume_cell(
            &vol_dof_handler.get_triangulation(),
            volume_cell->level(),
            volume_cell->index(),
            &vol_dof_handler
        );

        typename dealii::DoFCellAccessor<2,2,false> dof_volume_cell_neighbor(
            &vol_dof_handler.get_triangulation(),
            volume_cell_neighbor->level(),
            volume_cell_neighbor->index(),
            &vol_dof_handler
        );

        // get face_no's
        unsigned int face_no = dealii::numbers::invalid_unsigned_int;
        for (unsigned int f = 0; 
                f < dealii::GeometryInfo<2>::faces_per_cell; 
                ++f)
        {
            if (volume_cell->face(f) == volume_face)
            {
                face_no = f;
                break;
            }
        }

        // get neighbor_face_no's
        unsigned int neighbor_face_no = dealii::numbers::invalid_unsigned_int;
        for (unsigned int f = 0; 
                f < dealii::GeometryInfo<2>::faces_per_cell; 
                ++f)
        {
            if (volume_cell_neighbor->face(f) == volume_face)
            {
                neighbor_face_no = f;
                break;
            }
        }

        vol_fe_v_face.reinit(
            volume_cell, face_no);
        vol_fe_v_face_neighbor.reinit(
            volume_cell_neighbor, neighbor_face_no);

        int_fe_v.reinit(cell);

        check_quadrature_point_correspondence(
            int_fe_v, vol_fe_v_face, vol_fe_v_face_neighbor);

        const std::vector<double> &JxW = int_fe_v.get_JxW_values();
        const auto normals = vol_fe_v_face.get_normal_vectors();

        for (const unsigned int q_point : int_fe_v.quadrature_point_indices())
        {
            for (const unsigned int i : int_fe_v.dof_indices())
            {
                // TE mode: the tangential space on F is 1D, so J_F is a
                // scalar and the only tangential field is E_2. The 0.5 below
                // is the central-flux average of the two sides.
                const unsigned int block_index
                = int_fe_v.get_fe().system_to_component_index(i).first;

                const auto phi_i = int_fe_v.shape_value(i, q_point);
                
                const auto coupling_vector_entry = 
                    ode_system.coupling_vector(block_index);

                const auto coupling_times_phi_i = phi_i * coupling_vector_entry;

                for (const unsigned int j : vol_fe_v_face.dof_indices())
                {
                    const auto E_j_parallel = 
                        vol_fe_v_face[E].value(j, q_point)[1];
                    
                    cell_matrix_int(i, j) += 
                        coupling_times_phi_i * 0.5 * E_j_parallel * JxW[q_point];
                }

                for (const unsigned int j : vol_fe_v_face_neighbor.dof_indices())
                {
                    const auto E_j_parallel_neighbor =
                        vol_fe_v_face_neighbor[E].value(j, q_point)[1];

                    cell_matrix_ext(i, j) +=
                        coupling_times_phi_i * 0.5 * E_j_parallel_neighbor * JxW[q_point];
                }

            }
        }

        cell->get_dof_indices(local_dof_indices);
        dof_volume_cell.get_dof_indices(local_dof_indices_int);
        dof_volume_cell_neighbor.get_dof_indices(local_dof_indices_ext);

        for (const unsigned int i : int_fe_v.dof_indices())
        {
            for (const unsigned int j : vol_fe_v_face.dof_indices())
            {
                maxwell_ode_coupling_matrix.add(
                    local_dof_indices[i],
                    local_dof_indices_int[j], 
                    cell_matrix_int(i,j));

            }

            for (const unsigned int j : vol_fe_v_face_neighbor.dof_indices())
            {
                maxwell_ode_coupling_matrix.add(
                    local_dof_indices[i],
                    local_dof_indices_ext[j], 
                    cell_matrix_ext(i,j));
                
            }

        }

    }
}

void ODESystemOperators::assemble_ode_to_maxwell_coupling_matrix(
        dealii::BlockSparseMatrixEZ<double> &ode_maxwell_coupling_matrix)
{   

    const dealii::FEValuesExtractors::Scalar H(0);
    const dealii::FEValuesExtractors::Vector E(1);

    ode_maxwell_coupling_matrix = 0;
    
    auto dofs_per_int_cell = int_dof_handler.get_fe().dofs_per_cell;
    const auto dofs_per_cell = vol_fe.dofs_per_cell;

    dealii::FullMatrix<double> cell_matrix_int(
        dofs_per_cell, dofs_per_int_cell);
    dealii::FullMatrix<double> cell_matrix_ext(
        dofs_per_cell, dofs_per_int_cell);

    auto cell = int_dof_handler.begin_active();
    auto endc = int_dof_handler.end();

    std::vector<dealii::types::global_dof_index> 
        local_dof_indices(dofs_per_int_cell);
    std::vector<dealii::types::global_cell_index>
        local_dof_indices_int(dofs_per_cell);
    std::vector<dealii::types::global_cell_index>
        local_dof_indices_ext(dofs_per_cell);

    for(; cell != endc; ++cell)
    {
        cell_matrix_int = 0;
        cell_matrix_ext = 0;

        // Interface cell -> volume face -> the two adjacent volume cells.
        auto it = interface_cell_to_face_map.find(cell);
        if (it == interface_cell_to_face_map.end())
            throw dealii::ExcInvalidState();
        const auto &volume_face = it->second;
        auto it_2 = volume_face_to_volume_cells_map.find(volume_face);
        if (it_2 == volume_face_to_volume_cells_map.end())
            throw dealii::ExcInvalidState();

        const auto &volume_cell = it_2->second.first;
        const auto &volume_cell_neighbor = it_2->second.second;

        typename dealii::DoFCellAccessor<2,2,false> dof_volume_cell(
            &vol_dof_handler.get_triangulation(),
            volume_cell->level(),
            volume_cell->index(),
            &vol_dof_handler
        );

        typename dealii::DoFCellAccessor<2,2,false> dof_volume_cell_neighbor(
            &vol_dof_handler.get_triangulation(),
            volume_cell_neighbor->level(),
            volume_cell_neighbor->index(),
            &vol_dof_handler
        );

        // get face_no's
        unsigned int face_no = dealii::numbers::invalid_unsigned_int;
        for (unsigned int f = 0; 
                f < dealii::GeometryInfo<2>::faces_per_cell; 
                ++f)
        {
            if (volume_cell->face(f) == volume_face)
            {
                face_no = f;
                break;
            }
        }

        // get neighbor_face_no's
        unsigned int neighbor_face_no = dealii::numbers::invalid_unsigned_int;
        for (unsigned int f = 0; 
                f < dealii::GeometryInfo<2>::faces_per_cell; 
                ++f)
        {
            if (volume_cell_neighbor->face(f) == volume_face)
            {
                neighbor_face_no = f;
                break;
            }
        }

        vol_fe_v_face.reinit(
            volume_cell, face_no);
        vol_fe_v_face_neighbor.reinit(
            volume_cell_neighbor, neighbor_face_no);

        int_fe_v.reinit(cell);

        check_quadrature_point_correspondence(
            int_fe_v, vol_fe_v_face, vol_fe_v_face_neighbor);

        const std::vector<double> &JxW = int_fe_v.get_JxW_values();
        const auto normals = vol_fe_v_face.get_normal_vectors();

        for (const unsigned int q_point : int_fe_v.quadrature_point_indices())
        {
            for (const unsigned int j : int_fe_v.dof_indices())
            {
                // Transpose of the E -> ODE coupling: J_F enters the E
                // update as a lifted surface current.
                const unsigned int block_index
                = int_fe_v.get_fe().system_to_component_index(j).first;

                const auto phi_j = int_fe_v.shape_value(j, q_point);

                const auto coupling_times_phi_j =
                    ode_system.coupling_vector(block_index) * phi_j;

                for (const unsigned int i : vol_fe_v_face.dof_indices())
                {
                    const auto E_i_parallel =
                        vol_fe_v_face[E].value(i, q_point)[1];

                    cell_matrix_int(i, j) +=
                        coupling_times_phi_j * 0.5 * E_i_parallel * JxW[q_point];
                }

                for (const unsigned int i : vol_fe_v_face_neighbor.dof_indices())
                {
                    const auto E_i_parallel_neighbor =
                        vol_fe_v_face_neighbor[E].value(i, q_point)[1];

                    cell_matrix_ext(i, j) +=
                        coupling_times_phi_j * 0.5 * E_i_parallel_neighbor * JxW[q_point];
                }

            }
        }

        cell->get_dof_indices(local_dof_indices);
        dof_volume_cell.get_dof_indices(local_dof_indices_int);
        dof_volume_cell_neighbor.get_dof_indices(local_dof_indices_ext);

        for (const unsigned int j : int_fe_v.dof_indices())
        {
            for (const unsigned int i : vol_fe_v_face.dof_indices())
            {
                ode_maxwell_coupling_matrix.add(
                    local_dof_indices_int[i],
                    local_dof_indices[j],
                    cell_matrix_int(i,j));
                
            }

            for (const unsigned int i : vol_fe_v_face_neighbor.dof_indices())
            {
                ode_maxwell_coupling_matrix.add(
                    local_dof_indices_ext[i],
                    local_dof_indices[j], 
                    cell_matrix_ext(i,j));
                
            }

        }

    }
}

void ODESystemOperators::assemble_prop_matrices(
        dealii::BlockSparseMatrixEZ<double> &R_minus_op,
        dealii::BlockSparseMatrixEZ<double> &R_plus_op,
        double timestep_size)
{   

    R_minus_op = 0;
    R_plus_op = 0;

    auto dofs_per_cell = int_dof_handler.get_fe().n_dofs_per_cell();

    dealii::FullMatrix<double> cell_matrix_plus(dofs_per_cell, dofs_per_cell);
    dealii::FullMatrix<double> cell_matrix_minus(dofs_per_cell, dofs_per_cell);

    auto cell = int_dof_handler.begin_active();
    auto endc = int_dof_handler.end();

    std::vector<dealii::types::global_dof_index> 
        local_dof_indices(dofs_per_cell);

    for(; cell != endc; ++cell)
    {
        cell_matrix_plus = 0;
        cell_matrix_minus = 0;

        int_fe_v.reinit(cell);

        const std::vector<double> &JxW = int_fe_v.get_JxW_values();

        auto n_quad_points = int_fe_v.n_quadrature_points;
        for (unsigned int q_point = 0; q_point < n_quad_points; ++q_point)
        {
            for (const unsigned int i : int_fe_v.dof_indices())
            {
                for (const unsigned int j : int_fe_v.dof_indices())
                {

                            const unsigned int k
                            = int_fe_v.get_fe().system_to_component_index(i).first;
                            const unsigned int p
                            = int_fe_v.get_fe().system_to_component_index(j).first;

                            const auto phi_i
                                = int_fe_v.shape_value(i, q_point);
                            const auto phi_j 
                                = int_fe_v.shape_value(j, q_point);

                            double value_plus = 
                                timestep_size * 0.5 * ode_system.system_matrix(k,p);
                            double value_minus = 
                                - timestep_size * 0.5 * ode_system.system_matrix(k,p);
                            if (k == p)
                            {
                                value_plus += 1.0;
                                value_minus += 1.0;
                            }

                            const double product = phi_i * phi_j;
                            cell_matrix_plus(i, j) += value_plus * product * JxW[q_point];
                            cell_matrix_minus(i,j) += value_minus * product * JxW[q_point];

                }
            }
        }

        cell->get_dof_indices(local_dof_indices);

        for (const unsigned int i : int_fe_v.dof_indices())
        {
            for (const unsigned int j : int_fe_v.dof_indices())
            {
                R_plus_op.add(
                    local_dof_indices[i],
                    local_dof_indices[j], 
                    cell_matrix_plus(i,j));
                R_minus_op.add(
                    local_dof_indices[i],
                    local_dof_indices[j], 
                    cell_matrix_minus(i,j));
            }
        }

        // for (const unsigned int i : int_fe_v.dof_indices())
        //     for (const unsigned int j : int_fe_v.dof_indices())

    }
}

