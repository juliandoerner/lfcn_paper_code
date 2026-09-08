"""Sympy derivation of the manufactured right-hand sides in Solutions.py.

Edit the x2_concrete / theta_concrete block, run it, paste the printed source::

    python3 Data/CalcDiffs.py
"""

import sympy as sp
from sympy.printing.pycode import pycode

# -----------------------------
# 1. Symbols and function
# -----------------------------
x, t = sp.symbols('x t')
x2 = sp.Function('x2')(x, t)
theta = sp.Function('theta')(x, t)
x1 = sp.diff(x2, t)

# Enable pretty printing
sp.init_printing(use_unicode=True)


# -----------------------------
# 2. Derivatives
# -----------------------------

x2_tt = sp.diff(x2, t, 2)
x2_xx = sp.diff(x2, x, 2)
theta_x = sp.diff(theta, x, 1)
theta_t = sp.diff(theta, t, 1)
theta_xx = sp.diff(theta, x, 2)
x1_x = sp.diff(x1, x, 1)

# -----------------------------
# 3. Expressions
# -----------------------------

f_rhs = x2_tt - x2_xx + theta_x
g_rhs = theta_t - theta_xx + x1_x


# -----------------------------
# 4. Simplification
# -----------------------------
f_rhs_s      = sp.simplify(f_rhs)
g_rhs_s      = sp.simplify(g_rhs)



# -----------------------------
# 5. Optional: substitute a concrete function
# -----------------------------
omega = sp.Symbol("omega", real=True, positive=True)
k = sp.Symbol("k", integer=True, positive=True)
x2_concrete = t \
    * sp.sin(k * x) * sp.exp(-(x - 0.5)**2 / 0.05)

eta = sp.Symbol("eta", real=True, positive=True)
n = sp.Symbol("n", integer=True, positive=True)
theta_concrete = t \
    * sp.sin(n * x) * sp.exp(-(x - 0.5)**2 / 0.05)

f_rhs_s_subs = sp.simplify(
        f_rhs_s.subs(x2, x2_concrete))
f_rhs_s_subs = sp.simplify(
        f_rhs_s_subs.subs(theta, theta_concrete))
g_rhs_s_subs = sp.simplify(
        g_rhs_s.subs(x2, x2_concrete))
g_rhs_s_subs = sp.simplify(
        g_rhs_s_subs.subs(theta, theta_concrete))


print("\nx_1\n")
expr = sp.factor(sp.simplify(x1.subs(x2, x2_concrete)))
sp.pprint(expr)
code = pycode(expr)
print(code)

print("\nx_2\n")
expr = sp.factor(sp.simplify(x2_concrete))
sp.pprint(expr)
code = pycode(expr)
print(code)

print("\ntheta\n")
expr = sp.factor(sp.simplify(theta_concrete))
sp.pprint(expr)
code = pycode(expr)
print(code)

print("\nf_rhs\n")
expr = sp.factor(sp.together(f_rhs_s_subs))
sp.pprint(expr)
code = pycode(expr)
print(code)

print("\ng_rhs\n")
expr = sp.factor(sp.together(g_rhs_s_subs))
sp.pprint(expr)
code = pycode(expr)
print(code)

