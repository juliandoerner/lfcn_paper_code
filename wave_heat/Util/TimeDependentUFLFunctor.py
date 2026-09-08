"""UFL expressions with a time argument that can be moved without rebuilding."""

import dolfinx as dfx

class TimeDependentUFLFunctor:
    """A Functor to represent time dependent functions.

    Time enters as a `dfx.fem.Constant`, so advancing it is a write to
    `self.t.value` and the compiled form is never regenerated.  Set
    `reevaluation` only if the expression structure itself depends on t.
    """

    def __init__(func, expression_lambda, domain, reevaluation=False):
        func.domain = domain
        func._expression_lambda = expression_lambda
        func.t = dfx.fem.Constant(domain, 0.0)
        func.expression = expression_lambda(func.t)
        func.reevaluation = reevaluation

    def copy(func) -> "TimeDependentUFLFunctor":
        return TimeDependentUFLFunctor(
            func._expression_lambda, func.domain, reevaluation=func.reevaluation
        )

    def locked_evaluation(func, t_eval=0.0):
        return func._expression_lambda(t_eval)

    def __str__(func):
        expression_str = str(func.expression).replace(str(func.t), "t")
        return f"{expression_str} evaluated at t={func.t.value:04f}"

    def __call__(func, t_eval):
        func.t.value = t_eval
        if func.reevaluation:
            func.expression = func._expression_lambda(func.t)
        return func.expression
