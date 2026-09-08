import dolfinx as dfx

class TimeDependentUFLFunctor:
    """Wraps a ``t -> ufl`` callable so the expression can be advanced in time by
    setting ``.t.value``."""

    def __init__(func, expression_lambda, domain):
        func.domain = domain
        func._expression_lambda = expression_lambda
        func.t = dfx.fem.Constant(domain, 0.0)
        func.expression = expression_lambda(func.t)

    def __call__(func, t_eval):
        func.t.value = t_eval
        return func.expression
