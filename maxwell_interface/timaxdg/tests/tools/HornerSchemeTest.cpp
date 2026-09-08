#include "gtest/gtest.h"

#include "HornerScheme.h"

#define TOLERANCE (10e-6)

TEST(HornerSchemeTest, value) {
  std::vector<double> coeffs{14.1, 3.7, 12.11, 4.2, 1.41};
  
  EXPECT_NEAR(
    MaxwellProblem::Tools::horner_scheme(coeffs, 1.3), 
    52.6304010000000, 
    TOLERANCE);
  EXPECT_NEAR(
    MaxwellProblem::Tools::horner_scheme(coeffs, 0.0), 
    14.1000000000000, 
    TOLERANCE);
  EXPECT_NEAR(
    MaxwellProblem::Tools::horner_scheme(coeffs, -3.511), 
    182.873199749384, 
    TOLERANCE);
}

TEST(HornerSchemeTest, derivative) {
  std::vector<double> coeffs{14.1, 3.7, 12.11, 4.2, 1.41};
  
  EXPECT_NEAR(
    MaxwellProblem::Tools::horner_scheme_derivative(coeffs, 1.3), 
    68.8710800000000, 
    TOLERANCE);
  EXPECT_NEAR(
    MaxwellProblem::Tools::horner_scheme_derivative(coeffs, 0.0), 
    3.70000000000000, 
    TOLERANCE);
  EXPECT_NEAR(
    MaxwellProblem::Tools::horner_scheme_derivative(coeffs, -3.511), 
    -170.116838526840, 
    TOLERANCE);
}

TEST(HornerSchemeTest, seconde_derivative) {
  std::vector<double> coeffs{14.1, 3.7, 12.11, 4.2, 1.41};
  
  EXPECT_NEAR(
    MaxwellProblem::Tools::horner_scheme_second_derivative(coeffs, 1.3), 
    85.5748000000000, 
    TOLERANCE);
  EXPECT_NEAR(
    MaxwellProblem::Tools::horner_scheme_second_derivative(coeffs, 0.0), 
    24.2200000000000, 
    TOLERANCE);
  EXPECT_NEAR(
    MaxwellProblem::Tools::horner_scheme_second_derivative(coeffs, -3.511), 
    144.317687320000, 
    TOLERANCE);
}

