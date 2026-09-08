// Mesh convergence study for the LFCN scheme against the trigonometric
// manufactured solution of Exact_Solution.tex (src/utils/LorentzSolution.h).
// The bulk is source free; everything is driven by the surface current on
// F = {0} x (0,1), advanced as the first-order system u = (K, J).

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

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
#include "LorentzSolution.h"
#include "ODESystem.h"
#include "ODESystemOperators.h"

using namespace dealii;

class InterfaceSolution_Prototype
{
public:
  InterfaceSolution_Prototype(int degree,
                              double mesh_size,
                              int refinements,
                              int total_timesteps,
                              double start_time,
                              double end_time,
                              unsigned int n,
                              double beta,
                              double gamma,
                              double omega,
                              const std::filesystem::path &output);

  void assemble();
  void run();
  void set_vtu_output(bool output) { vtu_output = output; }
  void set_exact_output(bool output) { exact_output = output; }
  void print_mesh_info(std::ostream &stream) const;

private:
  void make_volume_mesh();
  void build_interface_mesh();
  void setup_system();
  void assemble_system();

  void assemble_ode_forcing(double time, BlockVector<double> &g) const;
  void set_ode_initial_value(BlockVector<double> &u) const;

  double compute_field_error() const;

  void compute_interface_error(double &K_error, double &J_error) const;

  void output_error(int timestep_number, double field_error, double K_error,
                    double J_error) const;
  void output_step(int timestep_number);
  void output_exact_step(int timestep_number);

  // parameters
  const int degree;
  const double mesh_size;    // target cell size of the volume mesh
  const int refinements;     // extra interface (edge) refinements
  int total_timesteps;
  const double start_time;
  const double end_time;

  // manufactured solution / oscillator
  const unsigned int n;
  const double beta;
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
  BlockVector<double> solution;     // Maxwell: block 0 = H, block 1 = E
  BlockVector<double> ode_solution; // interface ODE: ell blocks (K, J)
  BlockVector<double> g_0;          // ODE forcing at the start of a step
  BlockVector<double> g_1;          // ODE forcing at the end of a step
  Vector<double> j_current_zero;    // bulk current (identically zero here)

  std::vector<std::pair<double, std::string>> times_and_names;
  std::vector<std::pair<double, std::string>> exact_times_and_names;
};

InterfaceSolution_Prototype::InterfaceSolution_Prototype(
  int degree,
  double mesh_size,
  int refinements,
  int total_timesteps,
  double start_time,
  double end_time,
  unsigned int n,
  double beta,
  double gamma,
  double omega,
  const std::filesystem::path &output)
  : degree(degree),
    mesh_size(mesh_size),
    refinements(refinements),
    total_timesteps(total_timesteps),
    start_time(start_time),
    end_time(end_time),
    n(n),
    beta(beta),
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

void InterfaceSolution_Prototype::assemble()
{
  make_volume_mesh();
  build_interface_mesh();
  setup_system();
  assemble_system();
}

void InterfaceSolution_Prototype::make_volume_mesh()
{
  // The generator keeps the interface on cell faces, which the codim-1
  // coupling depends on.
  const Point<2> p0(x_left, y_bottom);  // (-1, 0)
  const Point<2> p1(x_mid, y_top);      // ( 0, 1)  interface top
  const Point<2> p2(x_right, y_top);    // ( 1, 1)
  if (degree  == 3)
    assemble_grid_xy_distortion(volume_tria, p0, p1, p2, mesh_size*1.3, 0.00);
  else
    assemble_grid_xy_distortion(volume_tria, p0, p1, p2, mesh_size, 0.15);

  // Mark first: random_refiner needs the ids to recognise and skip interface
  // cells, otherwise the interface stops being conforming.
  mark_cells(volume_tria, 0, 1, x_mid);

  // Random bulk refinement breaks the central-flux parity of a Cartesian mesh.
  const double distortion_factor = (degree == 3) ? 0.0 : 0.15;
  //const double distortion_factor = 0.15;
  random_refiner(volume_tria, distortion_factor);

  refine_interface(volume_tria, refinements, 0, 1, x_mid);

  dof_handler.distribute_dofs(fe);
  const std::vector<unsigned int> block_components = {0, 1, 1};
  DoFRenumbering::component_wise(dof_handler, block_components);
}

void InterfaceSolution_Prototype::build_interface_mesh()
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

void InterfaceSolution_Prototype::setup_system()
{
  // Maxwell matrices
  assembler.generate_mass_pattern(mass, mass_pattern);
  assembler.generate_mass_pattern(inv_mass, inv_mass_pattern);
  assembler.generate_curl_pattern(curl, curl_pattern);

  // Maxwell vectors
  {
    const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, {0, 1});
    const auto n_H = dofs_per_block[0];
    const auto n_E = dofs_per_block[1];

    solution.reinit(2);
    solution.block(0).reinit(n_H);
    solution.block(1).reinit(n_E);
    solution.collect_sizes();

    j_current_zero.reinit(n_E);
    j_current_zero = 0;
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

void InterfaceSolution_Prototype::assemble_system()
{
  assembler.assemble_mass_matrix_parallel(mass, inv_mass);
  assembler.assemble_curl_matrix_parallel(curl);
  curl.block(0, 1).operator*=(-1.0);

  ode_operators->assemble_ode_mass_matrix(ode_mass_matrix, ode_inv_mass_matrix);
  ode_operators->assemble_maxwell_to_ode_coupling_matrix(
    maxwell_ode_coupling_matrix);
  ode_operators->assemble_ode_to_maxwell_coupling_matrix(
    ode_maxwell_coupling_matrix);
}

// LFCN adds 0.5*dt*(g_0 + g_1) straight onto the mass-weighted ODE right-hand
// side, so this has to be the integrated \int phi G, not nodal values. The
// forcing only enters the J row.
void InterfaceSolution_Prototype::assemble_ode_forcing(
  double time, BlockVector<double> &g) const
{
  g = 0;

  lfcn::IntegratedForcing interface_forcing(n, beta, gamma, omega, time);

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
          fe_interface.system_to_component_index(i).first; // ODE variable (K=0, J=1)

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
void InterfaceSolution_Prototype::set_ode_initial_value(
  BlockVector<double> &u) const
{
  u = 0;

  lfcn::SurfaceCurrent surface_current(n, beta, start_time);

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

double InterfaceSolution_Prototype::compute_field_error() const
{
  lfcn::ExactSolution exact(n, beta, time);

  Vector<double> local_errors(volume_tria.n_active_cells());
  const QGauss<2> quad(2 * degree + 4);

  VectorTools::integrate_difference(mapping, dof_handler, solution, exact,
                                    local_errors, quad,
                                    VectorTools::L2_norm);

  return VectorTools::compute_global_error(volume_tria, local_errors,
                                           VectorTools::L2_norm);
}

// K and J are reported separately, hence the component masks.
void InterfaceSolution_Prototype::compute_interface_error(
  double &K_error, double &J_error) const
{
  lfcn::AuxiliaryVariables exact(n, beta, omega, time);

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

void InterfaceSolution_Prototype::output_error(int timestep_number,
                                               double field_error,
                                               double K_error,
                                               double J_error) const
{
  std::stringstream dir_name;
  dir_name << "degree_" << degree << "/mesh_size_" << mesh_size
           << "/steps_" << total_timesteps;
  const auto dir_path = output_path / dir_name.str();
  std::filesystem::create_directories(dir_path);

  const auto error_file_path = dir_path / "error_steps.txt";
  const bool exists = std::filesystem::exists(error_file_path);
  std::ofstream error_file(error_file_path, std::ios_base::app);

  const double combined_error =
    std::sqrt(K_error * K_error + J_error * J_error);

  if (!exists)
    error_file << "time_step_number\ttime\tL2_field_error\tL2_K_error\t"
                  "L2_J_error\tL2_combined_error\tmin_h\tmax_h\n";

  error_file << timestep_number << "\t" << std::setw(12) << time << "\t"
             << std::setw(12) << field_error << "\t" << std::setw(12) << K_error
             << "\t" << std::setw(12) << J_error << "\t" << std::setw(12)
             << combined_error << "\t" << std::setw(12)
             << GridTools::minimal_cell_diameter(volume_tria) << "\t"
             << std::setw(12) << GridTools::maximal_cell_diameter(volume_tria)
             << std::endl;
}

void InterfaceSolution_Prototype::output_step(int timestep_number)
{
  std::stringstream dir_name;
  dir_name << "degree_" << degree << "/mesh_size_" << mesh_size
           << "/steps_" << total_timesteps << "/solution";
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

// Companion output to solution.pvd, for a frame-by-frame comparison.
void InterfaceSolution_Prototype::output_exact_step(int timestep_number)
{
  std::stringstream dir_name;
  dir_name << "degree_" << degree << "/mesh_size_" << mesh_size
           << "/steps_" << total_timesteps << "/solution";
  const auto dir_path = output_path / dir_name.str();
  std::filesystem::create_directories(dir_path);

  std::stringstream filename;
  filename << "exact_" << timestep_number << ".vtu";
  const auto solution_path = dir_path / filename.str();
  const auto pair_path = dir_path / "exact.pvd";

  std::ofstream solution_stream(solution_path);
  std::ofstream pair_stream(pair_path);

  lfcn::ExactSolution exact_minus(n, beta, time, lfcn::Side::minus);
  lfcn::ExactSolution exact_plus(n, beta, time, lfcn::Side::plus);
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

void InterfaceSolution_Prototype::print_mesh_info(std::ostream &stream) const
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

void InterfaceSolution_Prototype::run()
{
  time = start_time;
  int timestep_number = 0;

  // H is discontinuous across F, so interpolating one function would give the
  // Omega_- support points on the interface the wrong side's value -- an O(1)
  // error in the jump that then radiates outwards.
  lfcn::ExactSolution exact_minus(n, beta, start_time, lfcn::Side::minus);
  lfcn::ExactSolution exact_plus(n, beta, start_time, lfcn::Side::plus);
  const std::map<types::material_id, const Function<2, double> *> exact_map{
    {0, &exact_minus}, {1, &exact_plus}};
  VectorTools::interpolate_based_on_material_id(mapping, dof_handler, exact_map,
                                                solution);
  set_ode_initial_value(ode_solution);

  double field_error = compute_field_error();
  double K_error, J_error;
  compute_interface_error(K_error, J_error);
  output_error(timestep_number, field_error, K_error, J_error);
  if (vtu_output)
    output_step(timestep_number);
  if (exact_output)
    output_exact_step(timestep_number);
  std::cout << "Initial field error: " << field_error
            << ", interface error (K, J): " << K_error << ", " << J_error
            << "\n";

  LFCN<SparseMatrix<double>, SparseMatrix<double>> integrator(
    inv_mass.block(0, 0), inv_mass.block(1, 1), curl.block(1, 0),
    curl.block(0, 1), ode_system, ode_mass_matrix, ode_inv_mass_matrix,
    maxwell_ode_coupling_matrix, ode_maxwell_coupling_matrix, timestep_width,
    *ode_operators);

  const int error_stride = std::max(1, total_timesteps / 200);

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

    // no bulk current for this solution
    integrator.integrate_step(solution.block(0), solution.block(1),
                              ode_solution, j_current_zero, j_current_zero, g_0,
                              g_1);

    time = t_new;

    if (timestep_number % error_stride == 0 ||
        timestep_number == total_timesteps)
    {
      field_error = compute_field_error();
      compute_interface_error(K_error, J_error);
      output_error(timestep_number, field_error, K_error, J_error);
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
}

int main()
{
  try
  {
    const double start_time = 0.0;
    const double end_time = 1.0;

    const unsigned int n = 10;     // transverse mode, k = n*pi
    const double beta = 3.0;      // needs cos(beta) != 0
    const double gamma = 1.0;     // oscillator damping
    const double omega = 0.5;     // oscillator frequency

    const std::vector<int> degrees{3};

    // Resolution is driven by target mesh sizes, coarse to fine, rather than
    // by global refinements.
    const double mesh_upper = 0.05;
    const double mesh_lower = 0.005;
    const int mesh_points = 20;
    auto mesh_range =
      MaxwellProblem::Tools::log_spaced(mesh_lower, mesh_upper, mesh_points);
    std::reverse(mesh_range.begin(), mesh_range.end());

    const int refinements = 0;
    const int total_timesteps = 10000;

    const auto output_path = std::filesystem::current_path();

    const int num_threads = 80;
    boost::asio::thread_pool pool(num_threads);

    for (const int degree : degrees)
      for (const double mesh_h : mesh_range)
      {
        boost::asio::post(pool, [=]() {
          InterfaceSolution_Prototype experiment(degree, mesh_h, refinements,
                                                 total_timesteps, start_time,
                                                 end_time, n, beta, gamma, omega,
                                                 output_path);
          experiment.assemble();
          // enable these for a single config to inspect the fields
          // experiment.set_vtu_output(true);
          // experiment.set_exact_output(true);
          experiment.print_mesh_info(std::cout);
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
