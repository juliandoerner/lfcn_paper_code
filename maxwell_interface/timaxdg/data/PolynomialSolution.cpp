#include "PolynomialSolution.h"

#include "deal.II/lac/vector.h"

#include "HornerScheme.h"

namespace MaxwellProblem::Data {

PolynomialSolution3D::PolynomialSolution3D(
	std::vector<double> time_coefficients,
	double current_time,
	std::vector<double> spatial_coefficients)
	: dealii::Function<3, double>(6, current_time),
	  time_coefficients{time_coefficients},
	  spatial_coefficients{spatial_coefficients} {}

double PolynomialSolution3D::value(
	const dealii::Point<3> &p,
	const unsigned int component) const {

  if (component == 0 || component == 1 || component == 2) return 0;

  const auto time = this->get_time();
  const auto time_polynomial = MaxwellProblem::Tools::horner_scheme(time_coefficients, time);

  switch (component) {
	case 3:
		{
	  const auto space_0_polynomial_derivative =
		  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[0]);
	  const auto space_1_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[1]);
	  const auto space_2_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[2]);

	  return time_polynomial * space_0_polynomial_derivative * space_1_polynomial * space_2_polynomial;
		}
	case 4:
		{
	  const auto space_0_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[0]);
	  const auto space_1_polynomial_derivative =
		  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[1]);
	  const auto space_2_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[2]);

	  return time_polynomial * space_0_polynomial * space_1_polynomial_derivative * space_2_polynomial;
		}
	case 5:
		{
	  const auto space_0_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[0]);
	  const auto space_1_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[1]);
	  const auto space_2_polynomial_derivative = MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[2]);

	  return time_polynomial * space_0_polynomial * space_1_polynomial * space_2_polynomial_derivative;
		}
	default:
	  throw dealii::ExcMessage("Polynomial Solution only has 6 components.");
  }
}

void PolynomialSolution3D::vector_value(
	const dealii::Point<3> &p,
	dealii::Vector<double> &values) const {

  const auto time = this->get_time();
  const auto time_polynomial = MaxwellProblem::Tools::horner_scheme(time_coefficients, time);
  const auto space_0_polynomial_derivative =
	  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[0]);
  const auto space_1_polynomial_derivative =
	  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[1]);
  const auto space_2_polynomial_derivative =
	  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[2]);
  const auto space_0_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[0]);
  const auto space_1_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[1]);
  const auto space_2_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[2]);

  values(0) = 0;
  values(1) = 0;
  values(2) = 0;
  values(3) = time_polynomial * space_0_polynomial_derivative * space_1_polynomial * space_2_polynomial;
  values(4) = time_polynomial * space_0_polynomial * space_1_polynomial_derivative * space_2_polynomial;
  values(5) = time_polynomial * space_0_polynomial * space_1_polynomial * space_2_polynomial_derivative;
}

PolynomialRhs3D::PolynomialRhs3D(
	std::vector<double> time_coefficients,
	double current_time,
	std::vector<double> spatial_coefficients)
	: dealii::Function<3, double>(3, current_time),
	  time_coefficients{time_coefficients},
	  spatial_coefficients{spatial_coefficients} {}

double PolynomialRhs3D::value(
	const dealii::Point<3> &p,
	const unsigned int component) const {

  if (component == 0 || component == 1 || component == 2) return 0;

  const auto time = this->get_time();
  const auto time_polynomial_derivative = MaxwellProblem::Tools::horner_scheme_derivative(time_coefficients, time);

  switch (component) {
	case 3:
		{
	  const auto space_0_polynomial_derivative =
		  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[0]);
	  const auto space_1_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[1]);
	  const auto space_2_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[2]);

	  return - time_polynomial_derivative * space_0_polynomial_derivative * space_1_polynomial * space_2_polynomial;
		}
	case 4:
		{
	  const auto space_0_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[0]);
	  const auto space_1_polynomial_derivative =
		  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[1]);
	  const auto space_2_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[2]);

	  return - time_polynomial_derivative * space_0_polynomial * space_1_polynomial_derivative * space_2_polynomial;
		}
	case 5:
		{
	  const auto space_0_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[0]);
	  const auto space_1_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[1]);
	  const auto space_2_polynomial_derivative = MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[2]);

	  return - time_polynomial_derivative * space_0_polynomial * space_1_polynomial * space_2_polynomial_derivative;
		}
	default:
	  throw dealii::ExcMessage("Polynomial Solution only has 6 components.");
  }
}

void PolynomialRhs3D::vector_value(
	const dealii::Point<3> &p,
	dealii::Vector<double> &values) const {

  const auto time = this->get_time();
  const auto time_polynomial_derivative = MaxwellProblem::Tools::horner_scheme_derivative(time_coefficients, time);
  const auto space_0_polynomial_derivative =
	  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[0]);
  const auto space_1_polynomial_derivative =
	  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[1]);
  const auto space_2_polynomial_derivative =
	  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[2]);
  const auto space_0_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[0]);
  const auto space_1_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[1]);
  const auto space_2_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[2]);

  values(0) = 0;
  values(1) = 0;
  values(2) = 0;
  values(3) = - time_polynomial_derivative * space_0_polynomial_derivative * space_1_polynomial * space_2_polynomial;
  values(4) = - time_polynomial_derivative * space_0_polynomial * space_1_polynomial_derivative * space_2_polynomial;
  values(5) = - time_polynomial_derivative * space_0_polynomial * space_1_polynomial * space_2_polynomial_derivative;
}

PolynomialSolutionTE::PolynomialSolutionTE(
	std::vector<double> time_coefficients,
	double current_time,
	std::vector<double> spatial_coefficients)
	: dealii::Function<2, double>(2, current_time),
	  time_coefficients{time_coefficients},
	  spatial_coefficients{spatial_coefficients} {}

double PolynomialSolutionTE::value(
	const dealii::Point<2> &p,
	const unsigned int component) const {

  if (component == 0) return 0;

  const auto time = this->get_time();
  const auto time_polynomial = MaxwellProblem::Tools::horner_scheme(time_coefficients, time);

  switch (component) {
	case 1:
		{
	  const auto space_0_polynomial_derivative =
		  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[0]);
	  const auto space_1_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[1]);

		return time_polynomial * space_0_polynomial_derivative * space_1_polynomial;
		}
	case 2:
		{
	  const auto space_0_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[0]);
	  const auto space_1_polynomial_derivative =
		  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[1]);

	  return time_polynomial * space_0_polynomial * space_1_polynomial_derivative;
		}
	default:
	  throw dealii::ExcMessage("Polynomial Solution only has 3 components.");
  }
}

void PolynomialSolutionTE::vector_value(
	const dealii::Point<2> &p,
	dealii::Vector<double> &values) const {

  const auto time = this->get_time();
  const auto time_polynomial = MaxwellProblem::Tools::horner_scheme(time_coefficients, time);
  const auto space_0_polynomial_derivative =
	  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[0]);
  const auto space_1_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[1]);
  const auto space_0_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[0]);
  const auto space_1_polynomial_derivative =
	  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[1]);

  values(0) = 0;
  values(1) = time_polynomial * space_0_polynomial_derivative * space_1_polynomial;
  values(2) = time_polynomial * space_0_polynomial * space_1_polynomial_derivative;
}

PolynomialRhsTE::PolynomialRhsTE(
	std::vector<double> time_coefficients,
	double current_time,
	std::vector<double> spatial_coefficients)
	: dealii::Function<2, double>(2, current_time),
	  time_coefficients{time_coefficients},
	  spatial_coefficients{spatial_coefficients} {}

double PolynomialRhsTE::value(
	const dealii::Point<2> &p,
	const unsigned int component) const {

  if (component == 0) return 0;

  const auto time = this->get_time();
  const auto time_polynomial_derivative = MaxwellProblem::Tools::horner_scheme_derivative(time_coefficients, time);

  switch (component) {
	case 1:
		{
	  const auto space_0_polynomial_derivative =
		  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[0]);
	  const auto space_1_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[1]);

	  return - time_polynomial_derivative * space_0_polynomial_derivative * space_1_polynomial;
		}
	case 2:
		{
	  const auto space_0_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[0]);
	  const auto space_1_polynomial_derivative =
		  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[1]);

	  return - time_polynomial_derivative * space_0_polynomial * space_1_polynomial_derivative;
		}
	default:
	  throw dealii::ExcMessage("Polynomial Solution only has 3 components.");
  }
}

void PolynomialRhsTE::vector_value(
	const dealii::Point<2> &p,
	dealii::Vector<double> &values) const {

  const auto time = this->get_time();
  const auto time_polynomial_derivative = MaxwellProblem::Tools::horner_scheme_derivative(time_coefficients, time);
  const auto space_0_polynomial_derivative =
	  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[0]);
  const auto space_1_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[1]);
  const auto space_0_polynomial = MaxwellProblem::Tools::horner_scheme(spatial_coefficients, p[0]);
  const auto space_1_polynomial_derivative =
	  MaxwellProblem::Tools::horner_scheme_derivative(spatial_coefficients, p[1]);

  values(0) = 0;
  values(1) = - time_polynomial_derivative * space_0_polynomial_derivative * space_1_polynomial;
  values(2) = - time_polynomial_derivative * space_0_polynomial * space_1_polynomial_derivative;
}

}// namespace MaxwellProblem::Data