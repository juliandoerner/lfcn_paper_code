#include "LorentzSolution.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/lac/vector.h>

#include <cmath>

namespace lfcn {

namespace {
double make_k(unsigned int n) { return n * dealii::numbers::PI; }
double make_nu(double beta, double k) { return std::sqrt(beta * beta + k * k); }
}  // namespace

ExactSolution::ExactSolution(unsigned int n, double beta, double initial_time,
                             Side side)
    : dealii::Function<2, double>(3, initial_time),
      n(n),
      beta(beta),
      side(side),
      k(make_k(n)),
      nu(make_nu(beta, k)) {}

double ExactSolution::h(double x1) const {
  const bool use_minus =
      (side == Side::minus) || (side == Side::automatic && x1 < 0.0);
  return use_minus ? -std::cos(beta * (x1 + 1.0))
                   : std::cos(beta * (x1 - 1.0));
}

double ExactSolution::dh(double x1) const {
  const bool use_minus =
      (side == Side::minus) || (side == Side::automatic && x1 < 0.0);
  return use_minus ? beta * std::sin(beta * (x1 + 1.0))
                   : -beta * std::sin(beta * (x1 - 1.0));
}

double ExactSolution::value(const dealii::Point<2> &p,
                            const unsigned int component) const {
  const double t = this->get_time();
  const double x1 = p[0];
  const double x2 = p[1];
  const double ct = std::cos(nu * t);
  const double st = std::sin(nu * t);

  switch (component) {
    case 0:  // H
      return ct * std::cos(k * x2) * h(x1);
    case 1:  // E_1
      return -(k / nu) * st * std::sin(k * x2) * h(x1);
    case 2:  // E_2
      return -(1.0 / nu) * st * std::cos(k * x2) * dh(x1);
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

SurfaceCurrent::SurfaceCurrent(unsigned int n, double beta, double initial_time)
    : dealii::Function<2, double>(1, initial_time),
      n(n),
      beta(beta),
      k(make_k(n)),
      nu(make_nu(beta, k)) {}

double SurfaceCurrent::value(const dealii::Point<2> &p,
                             const unsigned int /*component*/) const {
  const double t = this->get_time();
  return -2.0 * std::cos(beta) * std::cos(nu * t) * std::cos(k * p[1]);
}

AuxiliaryVariables::AuxiliaryVariables(unsigned int n, double beta,
                                       double omega, double initial_time)
    : dealii::Function<2, double>(2, initial_time),
      n(n),
      beta(beta),
      omega(omega),
      k(make_k(n)),
      nu(make_nu(beta, k)) {}

double AuxiliaryVariables::value(const dealii::Point<2> &p,
                                 const unsigned int component) const {
  const double t = this->get_time();
  // K = omega * int_0^t J_F
  const double profile = -2.0 * std::cos(beta) * std::cos(k * p[1]);
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

InterfaceForcing::InterfaceForcing(unsigned int n, double beta, double gamma,
                                   double omega, double initial_time)
    : dealii::Function<2, double>(1, initial_time),
      n(n),
      beta(beta),
      gamma(gamma),
      omega(omega),
      k(make_k(n)),
      nu(make_nu(beta, k)) {}

double InterfaceForcing::value(const dealii::Point<2> &p,
                               const unsigned int /*component*/) const {
  const double t = this->get_time();
  const double cb = std::cos(beta);
  const double sb = std::sin(beta);
  const double A = beta * sb - 2.0 * cb * (omega * omega - nu * nu);
  const double B = 2.0 * gamma * nu * cb;
  return std::cos(k * p[1]) * (A * std::cos(nu * t) + B * std::sin(nu * t));
}

IntegratedForcing::IntegratedForcing(unsigned int n, double beta, double gamma,
                                     double omega, double initial_time)
    : dealii::Function<2, double>(1, initial_time),
      n(n),
      beta(beta),
      gamma(gamma),
      omega(omega),
      k(make_k(n)),
      nu(make_nu(beta, k)) {}

double IntegratedForcing::value(const dealii::Point<2> &p,
                                const unsigned int /*component*/) const {
  const double t = this->get_time();
  const double cb = std::cos(beta);
  const double sb = std::sin(beta);
  const double A = beta * sb - 2.0 * cb * (omega * omega - nu * nu);
  return std::cos(k * p[1]) *
         ((A / nu) * std::sin(nu * t) - 2.0 * gamma * cb * std::cos(nu * t));
}

}  // namespace lfcn
