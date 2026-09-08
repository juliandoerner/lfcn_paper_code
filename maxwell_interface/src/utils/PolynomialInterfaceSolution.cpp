#include "PolynomialInterfaceSolution.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/vector.h>

#include <cmath>

namespace lfcn_poly {

namespace {
// Transverse profile. Raise the FE degree accordingly if this is changed to a
// higher-degree polynomial, otherwise the spatial error stops being zero.
double Q(double /*x2*/) { return 1.0; }
double dQ(double /*x2*/) { return 0.0; }

bool use_minus(Side side, double x1) {
  return (side == Side::minus) || (side == Side::automatic && x1 < 0.0);
}
}  // namespace

ExactSolution::ExactSolution(double nu, double initial_time, Side side)
    : dealii::Function<2, double>(3, initial_time), nu(nu), side(side) {}

double ExactSolution::r(double x1) const {
  return use_minus(side, x1) ? (1.0 + x1) : (1.0 - x1);
}

double ExactSolution::value(const dealii::Point<2> &p,
                            const unsigned int component) const {
  const double t = this->get_time();
  const double x1 = p[0];
  const double x2 = p[1];
  const double ct = std::cos(nu * t);
  const double st = std::sin(nu * t);

  // +Q on Omega_+, -Q on Omega_-
  const double sgn = use_minus(side, x1) ? -1.0 : 1.0;

  switch (component) {
    case 0:  // H
      return ct * (sgn * Q(x2) + (1.0 - 2.0 * x2) * (1.0 - x1 * x1));
    case 1:  // E_1
      return -nu * st * x2 * (1.0 - x2) * (1.0 - x1 * x1);
    case 2:  // E_2
      return -nu * st * Q(x2) * r(x1);
    default:
      Assert(false, dealii::ExcMessage(
                        "ExactSolution has 3 components (H, E_1, E_2)."));
      return 0.0;
  }
}

void ExactSolution::vector_value(const dealii::Point<2> &p,
                                 dealii::Vector<double> &values) const {
  AssertDimension(values.size(), 3);
  for (unsigned int c = 0; c < 3; ++c) values[c] = value(p, c);
}

VolumeCurrent::VolumeCurrent(double nu, double initial_time, Side side)
    : dealii::Function<2, double>(3, initial_time), nu(nu), side(side) {}

double VolumeCurrent::r(double x1) const {
  return use_minus(side, x1) ? (1.0 + x1) : (1.0 - x1);
}

double VolumeCurrent::value(const dealii::Point<2> &p,
                            const unsigned int component) const {
  const double t = this->get_time();
  const double x1 = p[0];
  const double x2 = p[1];
  const double ct = std::cos(nu * t);
  const double nu2 = nu * nu;

  // +Q' on Omega_+, -Q' on Omega_-
  const double sgn = use_minus(side, x1) ? -1.0 : 1.0;

  switch (component) {
    case 0:  // no current in the H equation
      return 0.0;
    case 1:  // J_1
      return ct * (sgn * dQ(x2) - 2.0 * (1.0 - x1 * x1) +
                   nu2 * x2 * (1.0 - x2) * (1.0 - x1 * x1));
    case 2:  // J_2
      return ct * (2.0 * x1 * (1.0 - 2.0 * x2) + nu2 * Q(x2) * r(x1));
    default:
      Assert(false, dealii::ExcMessage(
                        "VolumeCurrent has 3 components (0, J_1, J_2)."));
      return 0.0;
  }
}

void VolumeCurrent::vector_value(const dealii::Point<2> &p,
                                 dealii::Vector<double> &values) const {
  AssertDimension(values.size(), 3);
  for (unsigned int c = 0; c < 3; ++c) values[c] = value(p, c);
}

SurfaceCurrent::SurfaceCurrent(double nu, double initial_time)
    : dealii::Function<2, double>(1, initial_time), nu(nu) {}

double SurfaceCurrent::value(const dealii::Point<2> &p,
                             const unsigned int /*component*/) const {
  const double t = this->get_time();
  return -2.0 * std::cos(nu * t) * Q(p[1]);
}

AuxiliaryVariables::AuxiliaryVariables(double nu, double omega,
                                       double initial_time)
    : dealii::Function<2, double>(2, initial_time), nu(nu), omega(omega) {}

double AuxiliaryVariables::value(const dealii::Point<2> &p,
                                 const unsigned int component) const {
  const double t = this->get_time();
  // K = omega * int_0^t J_F
  const double profile = -2.0 * Q(p[1]);
  switch (component) {
    case 0:  // K
      return (omega / nu) * std::sin(nu * t) * profile;
    case 1:  // J_F
      return std::cos(nu * t) * profile;
    default:
      Assert(false, dealii::ExcMessage(
                        "AuxiliaryVariables has 2 components (K, J_F)."));
      return 0.0;
  }
}

void AuxiliaryVariables::vector_value(const dealii::Point<2> &p,
                                      dealii::Vector<double> &values) const {
  AssertDimension(values.size(), 2);
  for (unsigned int c = 0; c < 2; ++c) values[c] = value(p, c);
}

InterfaceForcing::InterfaceForcing(double nu, double gamma, double omega,
                                   double initial_time)
    : dealii::Function<2, double>(1, initial_time),
      nu(nu),
      gamma(gamma),
      omega(omega) {}

double InterfaceForcing::value(const dealii::Point<2> &p,
                               const unsigned int /*component*/) const {
  const double t = this->get_time();
  const double A = 3.0 * nu * nu - 2.0 * omega * omega;
  const double B = 2.0 * gamma * nu;
  return Q(p[1]) * (A * std::cos(nu * t) + B * std::sin(nu * t));
}

IntegratedForcing::IntegratedForcing(double nu, double gamma, double omega,
                                     double initial_time)
    : dealii::Function<2, double>(1, initial_time),
      nu(nu),
      gamma(gamma),
      omega(omega) {}

double IntegratedForcing::value(const dealii::Point<2> &p,
                                const unsigned int /*component*/) const {
  const double t = this->get_time();
  const double A = 3.0 * nu * nu - 2.0 * omega * omega;
  return Q(p[1]) *
         ((A / nu) * std::sin(nu * t) - 2.0 * gamma * std::cos(nu * t));
}

}  // namespace lfcn_poly
