// Time convergence study for the LFCN scheme. The manufactured solution of
// Exact_Solution_Polynomial.tex is piecewise polynomial in space, so a DG space
// of degree >= 2 on an affine mesh represents it exactly and only the time
// discretisation error is left. The mesh is therefore fixed and dt is swept.
// Unlike the trigonometric solution this one has nonzero bulk currents, defined
// by the two electric equations. Coarse dt violates the CFL condition and the
// run blows up; that is part of what the sweep shows.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_sparse_matrix_ez.h>
#include <deal.II/lac/block_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "AssemblerTE.h"
#include "IsotropicConstant.h"
#include "NumberSpaces.h"

#include "ExtractInterface.h"
#include "GenerateDomain.h"
#include "LFCN.h"
#include "LFCN.hh"
#include "ODESystem.h"
#include "ODESystemOperators.h"
#include "PolynomialInterfaceSolution.h"

using namespace dealii;

class InterfaceSolutionPolynomial_Prototype
{
public:
  InterfaceSolutionPolynomial_Prototype(int degree,
                                        double mesh_size,
                                        int refinements,
                                        int total_timesteps,
                                        double start_time,
                                        double end_time,
                                        double nu,
                                        double gamma,
                                        double omega,
                                        const std::filesystem::path &output);

  void assemble();
  void run();
  void set_total_timesteps(int new_total_timesteps);
  void set_vtu_output(bool output) { vtu_output = output; }
  void set_exact_output(bool output) { exact_output = output; }
  void print_mesh_info(std::ostream &stream) const;

private:
  void make_volume_mesh();
  void build_interface_mesh();
  void setup_system();
  void assemble_system();

  void assemble_volume_current_space();

  void assemble_ode_forcing(double time, BlockVector<double> &g) const;
  void set_ode_initial_value(BlockVector<double> &u) const;

  double compute_field_error() const;

  void compute_interface_error(double &K_error, double &J_error) const;

  void output_error(int timestep_number, double field_error, double K_error,
                    double J_error, bool unstable) const;
  void output_step(int timestep_number);
  void output_exact_step(int timestep_number);

  // parameters
  const int degree;
  const double mesh_size;    // fixed for this experiment
  const int refinements;     // extra interface (edge) refinements
  int total_timesteps;
  const double start_time;
  const double end_time;

  // manufactured solution / oscillator
  const double nu;
  const double gamma;
  const double omega;

  // domain Omega = (-1,1) x (0,1), interface at x = 0
  const double x_left{-1.0};
  const double x_mid{0.0};
  const double x_right{1.0};
  const double y_bottom{0.0};
  const double y_top{1.0};

  const std::filesystem::path output_path;
  bool vtu_output = false;
  bool exact_output = false;

  double time = 0.0;
  double timestep_width;

  // discrete spaces
  Triangulation<2> volume_tria;
  DoFHandler<2> dof_handler;
  FESystem<2> fe;
  MappingQ1<2> mapping;

  Triangulation<1, 2> interface_tria;
  DoFHandler<1, 2> dof_handler_interface;
  FESystem<1, 2> fe_interface;

  const QGauss<2> cell_quadrature;
  const QGauss<1> face_quadrature;
  const QGauss<1> interface_quadrature;

  MaxwellProblem::Data::IsotropicConstant<2> mu;
  MaxwellProblem::Data::IsotropicConstant<2> eps;

  MaxwellProblem::Assembling::AssemblerTE assembler;

  // surface-current ODE  u = (K, J),  d_t u + L u = c E_|| + g
  ODESystem ode_system;

  // Maxwell matrices
  BlockSparsityPattern mass_pattern;
  BlockSparseMatrix<double> mass;
  BlockSparsityPattern inv_mass_pattern;
  BlockSparseMatrix<double> inv_mass;
  BlockSparsityPattern curl_pattern;
  BlockSparseMatrix<double> curl;

  // interface ODE matrices
  BlockSparseMatrixEZ<double> ode_mass_matrix;
  BlockSparseMatrixEZ<double> ode_inv_mass_matrix;
  BlockSparseMatrixEZ<double> maxwell_ode_coupling_matrix;
  BlockSparseMatrixEZ<double> ode_maxwell_coupling_matrix;

  // kept alive because the operators hold references to them
  std::map<typename Triangulation<1, 2>::cell_iterator,
           typename Triangulation<2, 2>::face_iterator>
    interface_cell_to_face_map;
  std::map<Triangulation<2>::face_iterator,
           std::pair<Triangulation<2>::active_cell_iterator,
                     Triangulation<2>::active_cell_iterator>>
    volume_face_to_cells_map;

  std::unique_ptr<ODESystemOperators> ode_operators;

  // solution vectors
  types::global_dof_index n_H = 0;
  types::global_dof_index n_E = 0;

  BlockVector<double> solution;     // Maxwell: block 0 = H, block 1 = E
  BlockVector<double> ode_solution; // interface ODE: ell blocks (K, J)
  BlockVector<double> g_0;          // ODE forcing at the start of a step
  BlockVector<double> g_1;          // ODE forcing at the end of a step

  // Spatial part of the bulk current; the full current is cos(nu t) times this.
  Vector<double> j_current_space;
  Vector<double> j_current_0; // bulk current at the step start (half time)
  Vector<double> j_current_1; // bulk current at the step end   (half time)

  std::vector<std::pair<double, std::string>> times_and_names;
  std::vector<std::pair<double, std::string>> exact_times_and_names;
};

InterfaceSolutionPolynomial_Prototype::InterfaceSolutionPolynomial_Prototype(
  int degree,
  double mesh_size,
  int refinements,
  int total_timesteps,
  double start_time,
  double end_time,
  double nu,
  double gamma,
  double omega,
  const std::filesystem::path &output)
  : degree(degree),
    mesh_size(mesh_size),
    refinements(refinements),
    total_timesteps(total_timesteps),
    start_time(start_time),
    end_time(end_time),
    nu(nu),
    gamma(gamma),
    omega(omega),
    output_path(output),
    dof_handler(volume_tria),
    fe(FESystem<2>(FE_DGQ<2>(degree), 1), 1,
       FESystem<2>(FE_DGQ<2>(degree), 2), 1),
    dof_handler_interface(interface_tria),
    fe_interface(FE_DGQ<1, 2>(degree), /*ell=*/2),
    cell_quadrature(2 * degree + 2),
    face_quadrature(2 * degree + 1),
    interface_quadrature(3 * degree + 2),
    mu(1.0),
    eps(1.0),
    assembler(fe, mapping, cell_quadrature, face_quadrature, dof_handler, mu, eps),
    ode_system(/*system_size=*/2)
{
  timestep_width = (end_time - start_time) / total_timesteps;

  // u = (K, J) with K = omega * int J, so d_t u + L u = c E_par + g.
  ode_system.system_matrix = 0;
  ode_system.system_matrix.set(0, 0, 0.0);
  ode_system.system_matrix.set(0, 1, -omega);
  ode_system.system_matrix.set(1, 0, omega);
  ode_system.system_matrix.set(1, 1, gamma);

  ode_system.coupling_vector = 0;
  ode_system.coupling_vector(0) = 0.0; // K row does not couple to E
  ode_system.coupling_vector(1) = 1.0; // J row couples to E_{2,||}
}

void InterfaceSolutionPolynomial_Prototype::set_total_timesteps(
  int new_total_timesteps)
{
  total_timesteps = new_total_timesteps;
  timestep_width = (end_time - start_time) / total_timesteps;
}

void InterfaceSolutionPolynomial_Prototype::assemble()
{
  make_volume_mesh();
  build_interface_mesh();
  setup_system();
  assemble_system();
}

void InterfaceSolutionPolynomial_Prototype::make_volume_mesh()
{
  // The mesh has to stay affine (no distortion, no random refinement) or the
  // spatial error stops being zero and the time error is no longer isolated.
  const Point<2> p0(x_left, y_bottom);  // (-1, 0)
  const Point<2> p1(x_mid, y_top);      // ( 0, 1)  interface top
  const Point<2> p2(x_right, y_top);    // ( 1, 1)
  assemble_grid_xy_distortion(volume_tria, p0, p1, p2, mesh_size, 0.0);

  mark_cells(volume_tria, 0, 1, x_mid);

  refine_interface(volume_tria, refinements, 0, 1, x_mid);

  dof_handler.distribute_dofs(fe);
  const std::vector<unsigned int> block_components = {0, 1, 1};
  DoFRenumbering::component_wise(dof_handler, block_components);
}

void InterfaceSolutionPolynomial_Prototype::build_interface_mesh()
{
  interface_cell_to_face_map =
    extract_interface_mesh_2d(volume_tria, interface_tria);
  volume_face_to_cells_map = build_face_to_cells_sorted_by_material(
    volume_tria, interface_cell_to_face_map);

  dof_handler_interface.distribute_dofs(fe_interface);

  const auto ell = ode_system.system_size;
  std::vector<unsigned int> interface_blocks(ell);
  for (unsigned int i = 0; i < ell; ++i)
    interface_blocks[i] = i;
  DoFRenumbering::component_wise(dof_handler_interface, interface_blocks);
}

void InterfaceSolutionPolynomial_Prototype::setup_system()
{
  // Maxwell matrices
  assembler.generate_mass_pattern(mass, mass_pattern);
  assembler.generate_mass_pattern(inv_mass, inv_mass_pattern);
  assembler.generate_curl_pattern(curl, curl_pattern);

  // Maxwell vectors
  {
    const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, {0, 1});
    n_H = dofs_per_block[0];
    n_E = dofs_per_block[1];

    solution.reinit(2);
    solution.block(0).reinit(n_H);
    solution.block(1).reinit(n_E);
    solution.collect_sizes();

    j_current_space.reinit(n_E);
    j_current_0.reinit(n_E);
    j_current_1.reinit(n_E);
  }

  // interface ODE operators
  ode_operators = std::make_unique<ODESystemOperators>(
    fe_interface, interface_quadrature, dof_handler_interface, fe,
    dof_handler, interface_cell_to_face_map, volume_face_to_cells_map,
    ode_system);

  ode_operators->setup_matrices(ode_mass_matrix, ode_inv_mass_matrix,
                                maxwell_ode_coupling_matrix,
                                ode_maxwell_coupling_matrix);

  // interface ODE vectors (sized to match the assembled blocks)
  {
    const auto ell = ode_system.system_size;
    ode_solution.reinit(ell);
    for (unsigned int i = 0; i < ell; ++i)
      ode_solution.block(i).reinit(ode_mass_matrix.block(i, i).m());
    ode_solution.collect_sizes();

    g_0.reinit(ode_solution);
    g_1.reinit(ode_solution);
  }
}

void InterfaceSolutionPolynomial_Prototype::assemble_system()
{
  assembler.assemble_mass_matrix_parallel(mass, inv_mass);
  assembler.assemble_curl_matrix_parallel(curl);
  curl.block(0, 1).operator*=(-1.0);

  ode_operators->assemble_ode_mass_matrix(ode_mass_matrix, ode_inv_mass_matrix);
  ode_operators->assemble_maxwell_to_ode_coupling_matrix(
    maxwell_ode_coupling_matrix);
  ode_operators->assemble_ode_to_maxwell_coupling_matrix(
    ode_maxwell_coupling_matrix);

  assemble_volume_current_space();
}

// Assembles \int J^space . phi_E once; the time factor is applied per step.
// The current is side dependent, hence the per-cell choice by material id.
void InterfaceSolutionPolynomial_Prototype::assemble_volume_current_space()
{
  j_current_space = 0;

  // at t = 0 the time factor is 1, so this is the spatial part alone
  lfcn_poly::VolumeCurrent j_minus(nu, 0.0, lfcn_poly::Side::minus);
  lfcn_poly::VolumeCurrent j_plus(nu, 0.0, lfcn_poly::Side::plus);

  FEValues<2> fe_v(mapping, fe, cell_quadrature,
                   update_values | update_quadrature_points |
                     update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_v.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    const lfcn_poly::VolumeCurrent &j =
      (cell->material_id() == 0) ? j_minus : j_plus;

    for (const unsigned int q : fe_v.quadrature_point_indices())
    {
      const Point<2> &x = fe_v.quadrature_point(q);
      const double JxW = fe_v.get_JxW_values()[q];

      for (const unsigned int i : fe_v.dof_indices())
      {
        const unsigned int comp = fe.system_to_component_index(i).first;
        if (comp == 0) // H slot carries no current
          continue;

        const double Jc = j.value(x, comp); // comp 1 -> J_1, comp 2 -> J_2
        const auto gi = local_dof_indices[i]; // H block comes first
        j_current_space(gi - n_H) += fe_v.shape_value(i, q) * Jc * JxW;
      }
    }
  }
}

// LFCN adds 0.5*dt*(g_0 + g_1) straight onto the mass-weighted ODE right-hand
// side, so this has to be the integrated \int phi G, not nodal values. The
// forcing only enters the J row.
void InterfaceSolutionPolynomial_Prototype::assemble_ode_forcing(
  double time, BlockVector<double> &g) const
{
  g = 0;

  lfcn_poly::IntegratedForcing interface_forcing(nu, gamma, omega, time);

  FEValues<1, 2> fe_v(fe_interface, interface_quadrature,
                      update_values | update_quadrature_points |
                        update_JxW_values);

  const unsigned int dofs_per_cell = fe_interface.n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // global block offsets of g (block i starts at this global interface dof)
  std::vector<types::global_dof_index> block_start(ode_system.system_size + 1, 0);
  for (unsigned int b = 0; b < ode_system.system_size; ++b)
    block_start[b + 1] = block_start[b] + g.block(b).size();

  for (const auto &cell : dof_handler_interface.active_cell_iterators())
  {
    fe_v.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    for (const unsigned int q : fe_v.quadrature_point_indices())
    {
      const double f = interface_forcing.value(fe_v.quadrature_point(q));

      for (const unsigned int i : fe_v.dof_indices())
      {
        const unsigned int ode_block =
          fe_interface.system_to_component_index(i).first; // K=0, J=1

        // forcing enters the J equation
        if (ode_block != 1)
          continue;

        const auto gi = local_dof_indices[i];
        Assert(gi >= block_start[ode_block] && gi < block_start[ode_block + 1],
               ExcInternalError());

        g.block(ode_block)(gi - block_start[ode_block]) +=
          fe_v.shape_value(i, q) * f * fe_v.get_JxW_values()[q];
      }
    }
  }
}

// u(0) = (0, J_F(0)), L2-projected onto the interface space.
void InterfaceSolutionPolynomial_Prototype::set_ode_initial_value(
  BlockVector<double> &u) const
{
  u = 0;

  lfcn_poly::SurfaceCurrent surface_current(nu, start_time);

  BlockVector<double> rhs(u);
  rhs = 0;

  FEValues<1, 2> fe_v(fe_interface, interface_quadrature,
                      update_values | update_quadrature_points |
                        update_JxW_values);

  const unsigned int dofs_per_cell = fe_interface.n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<types::global_dof_index> block_start(ode_system.system_size + 1, 0);
  for (unsigned int b = 0; b < ode_system.system_size; ++b)
    block_start[b + 1] = block_start[b] + u.block(b).size();

  for (const auto &cell : dof_handler_interface.active_cell_iterators())
  {
    fe_v.reinit(cell);
    cell->get_dof_indices(local_dof_indices);

    for (const unsigned int q : fe_v.quadrature_point_indices())
    {
      const double j0 = surface_current.value(fe_v.quadrature_point(q));

      for (const unsigned int i : fe_v.dof_indices())
      {
        const unsigned int ode_block =
          fe_interface.system_to_component_index(i).first;
        if (ode_block != 1)
          continue;

        const auto gi = local_dof_indices[i];
        rhs.block(ode_block)(gi - block_start[ode_block]) +=
          fe_v.shape_value(i, q) * j0 * fe_v.get_JxW_values()[q];
      }
    }
  }

  ode_inv_mass_matrix.vmult(u, rhs);
}

double InterfaceSolutionPolynomial_Prototype::compute_field_error() const
{
  lfcn_poly::ExactSolution exact(nu, time);

  Vector<double> local_errors(volume_tria.n_active_cells());
  const QGauss<2> quad(2 * degree + 4);

  VectorTools::integrate_difference(mapping, dof_handler, solution, exact,
                                    local_errors, quad,
                                    VectorTools::L2_norm);

  return VectorTools::compute_global_error(volume_tria, local_errors,
                                           VectorTools::L2_norm);
}

// K and J are reported separately, hence the component masks.
void InterfaceSolutionPolynomial_Prototype::compute_interface_error(
  double &K_error, double &J_error) const
{
  lfcn_poly::AuxiliaryVariables exact(nu, omega, time);

  Vector<double> local_errors(interface_tria.n_active_cells());
  const QGauss<1> quad(2 * degree + 4);

  const ComponentSelectFunction<2> K_mask(0, 2);
  const ComponentSelectFunction<2> J_mask(1, 2);

  VectorTools::integrate_difference(dof_handler_interface, ode_solution, exact,
                                    local_errors, quad, VectorTools::L2_norm,
                                    &K_mask);
  K_error = VectorTools::compute_global_error(interface_tria, local_errors,
                                              VectorTools::L2_norm);

  VectorTools::integrate_difference(dof_handler_interface, ode_solution, exact,
                                    local_errors, quad, VectorTools::L2_norm,
                                    &J_mask);
  J_error = VectorTools::compute_global_error(interface_tria, local_errors,
                                              VectorTools::L2_norm);
}

void InterfaceSolutionPolynomial_Prototype::output_error(int timestep_number,
                                                         double field_error,
                                                         double K_error,
                                                         double J_error,
                                                         bool unstable) const
{
  std::stringstream dir_name;
  dir_name << "degree_" << degree << "/mesh_size_" << mesh_size << "/steps_"
           << total_timesteps;
  const auto dir_path = output_path / dir_name.str();
  std::filesystem::create_directories(dir_path);

  const auto error_file_path = dir_path / "error_steps.txt";
  const bool exists = std::filesystem::exists(error_file_path);
  std::ofstream error_file(error_file_path, std::ios_base::app);

  const double combined_error =
    std::sqrt(K_error * K_error + J_error * J_error);

  if (!exists)
    error_file << "time_step_number\ttime\tL2_field_error\tL2_K_error\t"
                  "L2_J_error\tL2_combined_error\tdt\tmin_h\tmax_h\tunstable\n";

  error_file << timestep_number << "\t" << std::setw(12) << time << "\t"
             << std::setw(12) << field_error << "\t" << std::setw(12) << K_error
             << "\t" << std::setw(12) << J_error << "\t" << std::setw(12)
             << combined_error << "\t" << std::setw(12) << timestep_width
             << "\t" << std::setw(12)
             << GridTools::minimal_cell_diameter(volume_tria) << "\t"
             << std::setw(12) << GridTools::maximal_cell_diameter(volume_tria)
             << "\t" << (unstable ? 1 : 0) << std::endl;
}

void InterfaceSolutionPolynomial_Prototype::output_step(int timestep_number)
{
  std::stringstream dir_name;
  dir_name << "degree_" << degree << "/mesh_size_" << mesh_size << "/steps_"
           << total_timesteps << "/solution";
  const auto dir_path = output_path / dir_name.str();
  std::filesystem::create_directories(dir_path);

  std::stringstream filename;
  filename << "solution_" << timestep_number << ".vtu";
  const auto solution_path = dir_path / filename.str();
  const auto pair_path = dir_path / "solution.pvd";

  std::ofstream solution_stream(solution_path);
  std::ofstream pair_stream(pair_path);

  const std::vector<std::string> names = {"H", "E", "E"};
  const std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation = {DataComponentInterpretation::component_is_scalar,
                      DataComponentInterpretation::component_is_part_of_vector,
                      DataComponentInterpretation::component_is_part_of_vector};

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(dof_handler, solution, names, interpretation);
  data_out.build_patches(degree + 1);
  data_out.write_vtu(solution_stream);

  times_and_names.emplace_back(time, filename.str());
  DataOutBase::write_pvd_record(pair_stream, times_and_names);
}

void InterfaceSolutionPolynomial_Prototype::output_exact_step(
  int timestep_number)
{
  std::stringstream dir_name;
  dir_name << "degree_" << degree << "/mesh_size_" << mesh_size << "/steps_"
           << total_timesteps << "/solution";
  const auto dir_path = output_path / dir_name.str();
  std::filesystem::create_directories(dir_path);

  std::stringstream filename;
  filename << "exact_" << timestep_number << ".vtu";
  const auto solution_path = dir_path / filename.str();
  const auto pair_path = dir_path / "exact.pvd";

  std::ofstream solution_stream(solution_path);
  std::ofstream pair_stream(pair_path);

  lfcn_poly::ExactSolution exact_minus(nu, time, lfcn_poly::Side::minus);
  lfcn_poly::ExactSolution exact_plus(nu, time, lfcn_poly::Side::plus);
  const std::map<types::material_id, const Function<2, double> *> exact_map{
    {0, &exact_minus}, {1, &exact_plus}};
  BlockVector<double> exact_field(solution);
  VectorTools::interpolate_based_on_material_id(mapping, dof_handler, exact_map,
                                                exact_field);

  const std::vector<std::string> names = {"H_exact", "E_exact", "E_exact"};
  const std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation = {DataComponentInterpretation::component_is_scalar,
                      DataComponentInterpretation::component_is_part_of_vector,
                      DataComponentInterpretation::component_is_part_of_vector};

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(dof_handler, exact_field, names, interpretation);
  data_out.build_patches(degree + 1);
  data_out.write_vtu(solution_stream);

  exact_times_and_names.emplace_back(time, filename.str());
  DataOutBase::write_pvd_record(pair_stream, exact_times_and_names);
}

void InterfaceSolutionPolynomial_Prototype::print_mesh_info(
  std::ostream &stream) const
{
  stream << "Mesh Info:\n"
         << std::string(20, '_') << "\n"
         << " minimal cell diameter: "
         << GridTools::minimal_cell_diameter(volume_tria) << "\n"
         << " maximal cell diameter: "
         << GridTools::maximal_cell_diameter(volume_tria) << "\n"
         << " volume active cells:   " << volume_tria.n_active_cells() << "\n"
         << " volume DoFs:           " << dof_handler.n_dofs() << "\n"
         << " interface cells:       " << interface_tria.n_active_cells()
         << "\n"
         << " interface DoFs:        " << dof_handler_interface.n_dofs() << "\n"
         << std::string(20, '_') << std::endl;
}

void InterfaceSolutionPolynomial_Prototype::run()
{
  time = start_time;
  int timestep_number = 0;

  // H is discontinuous across F, so interpolating one function would give the
  // Omega_- support points on the interface the wrong side's value.
  lfcn_poly::ExactSolution exact_minus(nu, start_time, lfcn_poly::Side::minus);
  lfcn_poly::ExactSolution exact_plus(nu, start_time, lfcn_poly::Side::plus);
  const std::map<types::material_id, const Function<2, double> *> exact_map{
    {0, &exact_minus}, {1, &exact_plus}};
  VectorTools::interpolate_based_on_material_id(mapping, dof_handler, exact_map,
                                                solution);
  set_ode_initial_value(ode_solution);

  double field_error = compute_field_error();
  double K_error, J_error;
  compute_interface_error(K_error, J_error);
  output_error(timestep_number, field_error, K_error, J_error,
               /*unstable=*/false);
  if (vtu_output)
    output_step(timestep_number);
  if (exact_output)
    output_exact_step(timestep_number);
  std::cout << "degree " << degree << ", steps " << total_timesteps
            << ", dt " << timestep_width
            << ": initial field error " << field_error
            << ", interface error (K, J): " << K_error << ", " << J_error
            << "\n";

  LFCN<SparseMatrix<double>, SparseMatrix<double>> integrator(
    inv_mass.block(0, 0), inv_mass.block(1, 1), curl.block(1, 0),
    curl.block(0, 1), ode_system, ode_mass_matrix, ode_inv_mass_matrix,
    maxwell_ode_coupling_matrix, ode_maxwell_coupling_matrix, timestep_width,
    *ode_operators);

  const int error_stride = std::max(1, total_timesteps / 200);

  // Past the CFL limit the error grows without bound; bail out and flag it
  // rather than churning through the remaining steps.
  const double blowup_threshold = 1.0e2;
  bool unstable = false;

  Timer eta_timer;
  eta_timer.start();

  for (timestep_number = 1; timestep_number <= total_timesteps;
       ++timestep_number)
  {
    const double t_old = start_time + (timestep_number - 1) * timestep_width;
    const double t_new = start_time + timestep_number * timestep_width;

    // LFCN treats the forcing trapezoidally over the step
    assemble_ode_forcing(t_old, g_0);
    assemble_ode_forcing(t_new, g_1);

    // The E equation has -J on the right while LFCN adds +j_current, hence
    // the sign flip.
    j_current_0 = j_current_space;
    j_current_0 *= -std::cos(nu * t_old);
    j_current_1 = j_current_space;
    j_current_1 *= -std::cos(nu * t_new);

    integrator.integrate_step(solution.block(0), solution.block(1),
                              ode_solution, j_current_0, j_current_1, g_0, g_1);

    time = t_new;

    if (timestep_number % error_stride == 0 ||
        timestep_number == total_timesteps)
    {
      field_error = compute_field_error();
      compute_interface_error(K_error, J_error);

      if (!std::isfinite(field_error) || field_error > blowup_threshold)
      {
        unstable = true;
        // finite sentinels keep the error file parseable
        if (!std::isfinite(field_error))
          field_error = 1.0e6;
        if (!std::isfinite(K_error))
          K_error = 1.0e6;
        if (!std::isfinite(J_error))
          J_error = 1.0e6;
        output_error(timestep_number, field_error, K_error, J_error,
                     /*unstable=*/true);
        std::cout << "step " << timestep_number << "/" << total_timesteps
                  << "\t UNSTABLE (CFL violated), error " << field_error
                  << " -- stopping run\n"
                  << std::flush;
        break;
      }

      output_error(timestep_number, field_error, K_error, J_error,
                   /*unstable=*/false);
      if (vtu_output)
        output_step(timestep_number);
      if (exact_output)
        output_exact_step(timestep_number);

      eta_timer.stop();
      const double eta_min =
        (eta_timer.wall_time() / error_stride) *
        (total_timesteps - timestep_number) / 60.0;
      std::cout << "step " << timestep_number << "/" << total_timesteps
                << "\t field error: " << field_error
                << "\t interface error (K, J): " << K_error << ", " << J_error
                << "\t eta: " << eta_min << " min\n"
                << std::flush;
      eta_timer.restart();
    }
  }

  if (unstable)
    std::cout << "degree " << degree << ", steps " << total_timesteps
              << ", dt " << timestep_width << ": run flagged UNSTABLE\n"
              << std::flush;
}

int main()
{
  try
  {
    const double start_time = 0.0;
    const double end_time = 1.0;

    const double nu = 24.0;     // temporal frequency
    const double gamma = 1.0;  // oscillator damping
    const double omega = 10;  // oscillator frequency

    // H and E_1 carry x1^2 / x2^2 terms, so degree 1 would still see a spatial
    // error and has no place in a pure time-error study.
    const std::vector<int> degrees{2};

    const int refinements = 0;

    // Finer meshes do not lower the time-error floor, but they do change its
    // constant and push the CFL limit to smaller dt.
    const double mesh_upper = 0.1;
    const double mesh_lower = 0.01;
    const int mesh_points = 10;
    auto mesh_range =
      MaxwellProblem::Tools::log_spaced(mesh_lower, mesh_upper, mesh_points);
    std::reverse(mesh_range.begin(), mesh_range.end()); // coarse -> fine

    // The coarse end is deliberately past the CFL limit so the instability is
    // visible; the fine end resolves the O(dt^2) order.
    const int steps_lower = 10;
    const int steps_upper = 10000;
    const int steps_points = 50;
    auto steps_real =
      MaxwellProblem::Tools::log_spaced((double)steps_lower,
                                        (double)steps_upper, steps_points);
    std::set<int> steps_set;
    for (const double s : steps_real)
      steps_set.insert(std::max(1, (int)std::lround(s)));
    const std::vector<int> step_counts(steps_set.begin(), steps_set.end());

    const auto output_path = std::filesystem::current_path();

    const int num_threads = 60;
    boost::asio::thread_pool pool(num_threads);

    for (const int degree : degrees)
      for (const double mesh_h : mesh_range)
        for (const int steps : step_counts)
        {
          boost::asio::post(pool, [=]() {
            InterfaceSolutionPolynomial_Prototype experiment(
              degree, mesh_h, refinements, steps, start_time, end_time, nu,
              gamma, omega, output_path);
            experiment.assemble();
            // enable these for a single config to inspect the fields
            // experiment.set_vtu_output(true);
            // experiment.set_exact_output(true);
            experiment.run();
          });
        }

    pool.join();
  }
  catch (std::exception &exc)
  {
    std::cerr << "\nException: " << exc.what() << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << "\nUnknown exception!" << std::endl;
    return 1;
  }

  return 0;
}
