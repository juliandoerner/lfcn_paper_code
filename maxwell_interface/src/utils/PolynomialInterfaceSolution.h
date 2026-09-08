// Polynomial-in-space exact solution of Exact_Solution_Polynomial.tex, on the
// same domain as LorentzSolution.h. Being piecewise polynomial, it is
// represented exactly by a DG space of high enough degree on an affine mesh
// (degree >= 2 for the constant profile Q below), which leaves only the time
// discretisation error. Unlike the trigonometric solution the bulk currents do
// not vanish: they are defined by the two electric equations.

#ifndef LFCN_PAPER_POLYNOMIAL_INTERFACE_SOLUTION_H
#define LFCN_PAPER_POLYNOMIAL_INTERFACE_SOLUTION_H

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

namespace lfcn_poly {

// H and E_2 are built from a side-dependent profile, so interpolating with
// `automatic` would average the two sides on the interface. The forced variants
// let the caller pick the branch per cell via the material id.
enum class Side { automatic, minus, plus };

// Fields (H, E_1, E_2). Evaluating at t = 0 gives the initial data.
class ExactSolution : public dealii::Function<2, double> {
 public:
  ExactSolution(double nu, double initial_time = 0.0,
                Side side = Side::automatic);

  double value(const dealii::Point<2> &p,
               const unsigned int component = 0) const override;

  void vector_value(const dealii::Point<2> &p,
                    dealii::Vector<double> &values) const override;

 private:
  double r(double x1) const;

  const double nu;
  const Side side;
};

// Bulk currents (0, J_1, J_2) driving the fields. The whole thing carries the
// common factor cos(nu t), so t = 0 returns the purely spatial part.
class VolumeCurrent : public dealii::Function<2, double> {
 public:
  VolumeCurrent(double nu, double initial_time = 0.0,
                Side side = Side::automatic);

  double value(const dealii::Point<2> &p,
               const unsigned int component = 0) const override;

  void vector_value(const dealii::Point<2> &p,
                    dealii::Vector<double> &values) const override;

 private:
  double r(double x1) const;

  const double nu;
  const Side side;
};

// Trace of the surface current J_F on F; x1 is ignored.
class SurfaceCurrent : public dealii::Function<2, double> {
 public:
  SurfaceCurrent(double nu, double initial_time = 0.0);

  double value(const dealii::Point<2> &p,
               const unsigned int component = 0) const override;

 private:
  const double nu;
};

// Interface ODE unknowns u = (K, J_F) with K = omega * int_0^t J_F, in the
// component order of the interface FESystem. Used to measure the ODE error.
class AuxiliaryVariables : public dealii::Function<2, double> {
 public:
  AuxiliaryVariables(double nu, double omega, double initial_time = 0.0);

  double value(const dealii::Point<2> &p,
               const unsigned int component = 0) const override;

  void vector_value(const dealii::Point<2> &p,
                    dealii::Vector<double> &values) const override;

 private:
  const double nu;
  const double omega;
};

// Forcing g of the second order oscillator equation. Not used by LFCN, which
// takes IntegratedForcing instead.
class InterfaceForcing : public dealii::Function<2, double> {
 public:
  InterfaceForcing(double nu, double gamma, double omega,
                   double initial_time = 0.0);

  double value(const dealii::Point<2> &p,
               const unsigned int component = 0) const override;

 private:
  const double nu;
  const double gamma;
  const double omega;
};

// Forcing G of the first-order (K, J) system actually solved by LFCN:
// d_t u + L u = c E_par + (0, G)^T. This is the time integral of g.
class IntegratedForcing : public dealii::Function<2, double> {
 public:
  IntegratedForcing(double nu, double gamma, double omega,
                    double initial_time = 0.0);

  double value(const dealii::Point<2> &p,
               const unsigned int component = 0) const override;

 private:
  const double nu;
  const double gamma;
  const double omega;
};

}  // namespace lfcn_poly

#endif  // LFCN_PAPER_POLYNOMIAL_INTERFACE_SOLUTION_H
