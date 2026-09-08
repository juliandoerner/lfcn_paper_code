"""Manufactured solutions: analytic (x1, x2, theta) with the f and g that make
them exact solutions of the continuous system."""

import ufl

import numpy as np

# Sin/Cos in time
# Polynomial in space
class OscSolution:

    """x2 = cos(omega t) x(1-x), theta = sin(eta t) x(1-x)."""

    def __init__(self, omega: float, eta: float) -> None:
        self.omega = omega
        self.eta = eta

        self.x1_expr = lambda x,t: -self.omega * np.sin(self.omega * t) * x[0] * (1 - x[0])
        self.x2_expr = lambda x,t: np.cos(self.omega * t) * x[0] * (1 - x[0])
        self.theta_expr = lambda x,t: np.sin(self.eta * t) * x[0] * (1 - x[0])

        self.f_expr = lambda x,t: (
            -self.omega**2 * ufl.cos(self.omega * t) * x[0] * (1 - x[0])
            + 2 * ufl.cos(self.omega * t)
            + ufl.sin(self.eta * t) * (1 - 2 * x[0]) 
        )

        self.g_expr = lambda x,t: (
            self.eta * ufl.cos(self.eta * t) * x[0] * (1 - x[0])
            + 2 * ufl.sin(self.eta * t)
            - self.omega * ufl.sin(self.omega * t ) * (1 - 2 * x[0])
        )

# Exponential in time
# Polynomial in space
class ExpSolution:

    """x2 = e^t cos(omega t) x(1-x), theta = e^t sin(eta t) x(1-x)."""

    def __init__(self, omega: float, eta: float) -> None:
        self.omega = omega
        self.eta = eta

        self.x1_expr = lambda x,t: (
            np.exp(t) * (
                np.cos(self.omega * t) 
                - self.omega * np.sin(self.omega * t)
                )
             * x[0] * (1 - x[0]))
        self.x2_expr = lambda x,t: np.exp(t) * np.cos(self.omega * t) * x[0] * (1 - x[0])
        self.theta_expr = lambda x,t: np.exp(t) * np.sin(self.eta * t) * x[0] * (1 - x[0])

        self.f_expr = lambda x,t: (
            ufl.exp(t) * (ufl.cos(self.omega * t) 
                          -  2 *self.omega*ufl.sin(self.omega * t) 
                          - self.omega*self.omega * ufl.cos(self.omega*t)
                          ) 
                        * x[0] * (1-x[0])
            + 2 * ufl.exp(t) * ufl.cos(self.omega * t)
            + ufl.exp(t) * ufl.sin(self.eta * t) * (1 - 2 * x[0])
        )

        self.g_expr = lambda x,t: (
            ufl.exp(t) * (
                    ufl.sin(self.eta * t) 
                    + self.eta * ufl.cos(self.eta * t)) 
                * x[0] * (1 - x[0])
            + 2 * ufl.exp(t) * ufl.sin(self.eta * t)
            + ufl.exp(t) * (
                ufl.cos(self.omega * t) 
                - self.omega * ufl.sin(self.omega * t)) 
                * (1 - 2 * x[0])
        )

# Exponential in time
# Exponential in space
class ExpExpSolution:

    """e^t times a Gaussian-modulated sine; coefficients from CalcDiffs.py."""

    def __init__(self, omega: float, k: int, eta: float, n: int) -> None:
        self.omega = omega
        self.eta = eta
        self.k = k
        self.n = n

        self.x1_expr = lambda x,t: \
            (
                0.00673794699908547
                *(
                    self.omega*np.cos(self.omega*t) 
                    + np.sin(self.omega*t)
                )
                *np.exp(t)
                *np.exp(20.0*x[0])
                *np.exp(-20.0*x[0]**2)
                *np.sin(self.k*x[0])
            )
        self.x2_expr = lambda x,t: \
            (
                0.00673794699908547
                *np.exp(t)
                *np.exp(20.0*x[0])
                *np.exp(-20.0*x[0]**2)
                *np.sin(self.k*x[0])
                *np.sin(self.omega*t)
            )
        self.theta_expr = lambda x,t: \
        (
            0.00673794699908547
            *np.exp(t)
            *np.exp(20.0*x[0])
            *np.exp(-20.0*x[0]**2)
            *np.sin(self.eta*t)
            *np.sin(self.n*x[0])
        )

        self.f_expr = lambda x,t: \
        (
            -10.7807151985367
            *(
                -0.000625*self.k**2
                *ufl.sin(self.k*x[0])
                *ufl.sin(self.omega*t) 
                - 0.05*self.k*x[0]
                *ufl.sin(self.omega*t)
                *ufl.cos(self.k*x[0]) 
                + 0.025*self.k
                *ufl.sin(self.omega*t)
                *ufl.cos(self.k*x[0]) 
                - 0.000625*self.n
                *ufl.sin(self.eta*t)
                *ufl.cos(self.n*x[0]) 
                + 0.000625*self.omega**2
                *ufl.sin(self.k*x[0])
                *ufl.sin(self.omega*t) 
                - 0.00125*self.omega
                *ufl.sin(self.k*x[0])
                *ufl.cos(self.omega*t) 
                + 1.0*x[0]**2
                *ufl.sin(self.k*x[0])
                *ufl.sin(self.omega*t) 
                + 0.025*x[0]
                *ufl.sin(self.eta*t)
                *ufl.sin(self.n*x[0]) 
                - 1.0*x[0]
                *ufl.sin(self.k*x[0])
                *ufl.sin(self.omega*t) 
                - 0.0125
                *ufl.sin(self.eta*t)
                *ufl.sin(self.n*x[0]) 
                + 0.224375
                *ufl.sin(self.k*x[0])
                *ufl.sin(self.omega*t)
            )
            *ufl.exp(t)
            *ufl.exp(20.0*x[0])
            *ufl.exp(-20.0*x[0]**2)
        )

        self.g_expr = lambda x,t: \
        (
            -10.7807151985367
            *(
                -0.000625*self.eta
                *ufl.sin(self.n*x[0])
                *ufl.cos(self.eta*t) 
                - 0.000625*self.k*self.omega
                *ufl.cos(self.k*x[0])
                *ufl.cos(self.omega*t) 
                - 0.000625*self.k
                *ufl.sin(self.omega*t)
                *ufl.cos(self.k*x[0]) 
                - 0.000625*self.n**2
                *ufl.sin(self.eta*t)
                *ufl.sin(self.n*x[0]) 
                - 0.05*self.n*x[0]
                *ufl.sin(self.eta*t)
                *ufl.cos(self.n*x[0]) 
                + 0.025*self.n
                *ufl.sin(self.eta*t)
                *ufl.cos(self.n*x[0])
                + 0.025*self.omega*x[0]
                *ufl.sin(self.k*x[0])
                *ufl.cos(self.omega*t) 
                - 0.0125*self.omega
                *ufl.sin(self.k*x[0])
                *ufl.cos(self.omega*t) 
                + 1.0*x[0]**2
                *ufl.sin(self.eta*t)
                *ufl.sin(self.n*x[0]) 
                - 1.0*x[0]
                *ufl.sin(self.eta*t)
                *ufl.sin(self.n*x[0]) 
                + 0.025*x[0]
                *ufl.sin(self.k*x[0])
                *ufl.sin(self.omega*t) 
                + 0.224375
                *ufl.sin(self.eta*t)
                *ufl.sin(self.n*x[0]) 
                - 0.0125
                *ufl.sin(self.k*x[0])
                *ufl.sin(self.omega*t)
            )
            *ufl.exp(t)
            *ufl.exp(20.0*x[0])
            *ufl.exp(-20.0*x**2)
        )

# Polynomial in time
# Exponential in space
class PolyExpSolution:

    """t^2 times a Gaussian-modulated sine; coefficients from CalcDiffs.py."""

    def __init__(self, k: int, n: int) -> None:
        self.k = k
        self.n = n

        self.x1_expr = lambda x,t: \
            (
                0.0134758939981709
                *t
                *np.exp(20.0*x[0])
                *np.exp(-20.0*x[0]**2)
                *np.sin(self.k*x[0])
            )
        self.x2_expr = lambda x,t: \
            (
                0.00673794699908547
                *t**2
                *np.exp(20.0*x[0])
                *np.exp(-20.0*x[0]**2)
                *np.sin(self.k*x[0])
            )
        self.theta_expr = lambda x,t: \
        (
            0.00673794699908547
            *t**2
            *np.exp(20.0*x[0])
            *np.exp(-20.0*x[0]**2)
            *np.sin(self.n*x[0])
        )

        self.f_expr = lambda x,t: \
        (
            -10.7807151985367
            *(
                -0.000625
                *self.k**2
                *t**2
                *ufl.sin(self.k*x[0]) 
                - 0.05
                *self.k
                *t**2
                *x[0]
                *ufl.cos(self.k*x[0]) 
                + 0.025
                *self.k
                *t**2
                *ufl.cos(self.k*x[0]) 
                - 0.000625
                *self.n
                *t**2
                *ufl.cos(self.n*x[0]) 
                + 1.0
                *t**2
                *x[0]**2
                *ufl.sin(self.k*x[0]) 
                - 1.0
                *t**2
                *x[0]
                *ufl.sin(self.k*x[0]) 
                + 0.025
                *t**2
                *x[0]
                *ufl.sin(self.n*x[0]) 
                + 0.225
                *t**2
                *ufl.sin(self.k*x[0]) 
                - 0.0125
                *t**2
                *ufl.sin(self.n*x[0]) 
                - 0.00125
                *ufl.sin(self.k*x[0])
                )
            *ufl.exp(20.0*x[0])
            *ufl.exp(-20.0*x[0]**2)
        )

        self.g_expr = lambda x,t: \
        (
            -10.7807151985367
            *t
            *(
                -0.00125
                *self.k
                *ufl.cos(self.k*x[0]) 
                - 0.000625
                *self.n**2
                *t
                *ufl.sin(self.n*x[0]) 
                - 0.05
                *self.n
                *t
                *x[0]
                *ufl.cos(self.n*x[0]) 
                + 0.025
                *self.n
                *t
                *ufl.cos(self.n*x[0]) 
                + 1.0
                *t
                *x[0]**2
                *ufl.sin(self.n*x[0]) 
                - 1.0
                *t
                *x[0]
                *ufl.sin(self.n*x[0]) 
                + 0.225
                *t
                *ufl.sin(self.n*x[0]) 
                + 0.05
                *x[0]
                *ufl.sin(self.k*x[0]) 
                - 0.025
                *ufl.sin(self.k*x[0]) 
                - 0.00125
                *ufl.sin(self.n*x[0])
                )
            *ufl.exp(20.0*x[0])
            *ufl.exp(-20.0*x[0]**2)
        )

# Monomial (linear) in time
# Exponential in space
class MonExpSolution:

    """t times a Gaussian-modulated sine, so x1 is stationary; used by HPlot.py."""

    def __init__(self, k: int, n: int) -> None:
        self.k = k
        self.n = n

        self.x1_expr = lambda x,t: \
            (
                0.00673794699908547
                *np.exp(20.0*x[0])
                *np.exp(-20.0*x[0]**2)
                *np.sin(self.k*x[0])
            )
        self.x2_expr = lambda x,t: \
            (
               0.00673794699908547
               *t
               *np.exp(20.0*x[0])
               *np.exp(-20.0*x[0]**2)
               *np.sin(self.k*x[0])
            )
        self.theta_expr = lambda x,t: \
        (
            0.00673794699908547
            *t
            *np.exp(20.0*x[0])
            *np.exp(-20.0*x[0]**2)
            *np.sin(self.n*x[0])
        )

        self.f_expr = lambda x,t: \
        (
            -10.7807151985367
            *t
            *(
                -0.000625
                *self.k**2
                *ufl.sin(self.k*x[0]) 
                - 0.05
                *self.k
                *x[0]
                *ufl.cos(self.k*x[0]) 
                + 0.025
                *self.k
                *ufl.cos(self.k*x[0]) 
                - 0.000625
                *self.n
                *ufl.cos(self.n*x[0]) 
                + 1.0
                *x[0]**2
                *ufl.sin(self.k*x[0]) 
                - 1.0
                *x[0]
                *ufl.sin(self.k*x[0]) 
                + 0.025
                *x[0]
                *ufl.sin(self.n*x[0]) 
                + 0.225
                *ufl.sin(self.k*x[0]) 
                - 0.0125
                *ufl.sin(self.n*x[0])
                )
            *ufl.exp(20.0*x[0])
            *ufl.exp(-20.0*x[0]**2)
        )

        self.g_expr = lambda x,t: \
        (
            -10.7807151985367
            *(
                -0.000625
                *self.k
                *ufl.cos(self.k*x[0]) 
                - 0.000625
                *self.n**2
                *t
                *ufl.sin(self.n*x[0]) 
                - 0.05
                *self.n
                *t
                *x[0]
                *ufl.cos(self.n*x[0]) 
                + 0.025
                *self.n
                *t
                *ufl.cos(self.n*x[0]) 
                + 1.0
                *t
                *x[0]**2
                *ufl.sin(self.n*x[0]) 
                - 1.0
                *t
                *x[0]
                *ufl.sin(self.n*x[0]) 
                + 0.225
                *t
                *ufl.sin(self.n*x[0]) 
                + 0.025
                *x[0]
                *ufl.sin(self.k*x[0]) 
                - 0.0125
                *ufl.sin(self.k*x[0]) 
                - 0.000625
                *ufl.sin(self.n*x[0])
            )
            *ufl.exp(20.0*x[0])
            *ufl.exp(-20.0*x[0]**2)
        )