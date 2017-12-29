import torch
from torch.nn.functional import sigmoid, softmax


class Constraint(object):
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a continuous variable is
    valid, e.g. within which a variable can optimized.

    Each constraint class registers a pseudoinverse pair of transforms
    `to_unconstrained` and `from_unconstrained`. These allow standard paramters
    to be transformed to an unconstrained space for optimization and
    transformed back after optimization. Note that these are not necessarily
    inverse pairs since the unconstrained space may have extra dimensions that
    are projected out; only the one-sided inverse equation is guaranteed::

        x == c.to_constrained(c.to_unconstrained(x))
    """
    def to_unconstrained(self, x):
        """
        Transform from constrained coordinates to unconstrained coordinates.
        """
        raise NotImplementedError

    def to_constrained(self, x):
        """
        Transform from unconstrained coordinates to constrained coordinates.
        """
        raise NotImplementedError


class Unconstrained(Constraint):
    """
    Trivial constraint to the real line `(-inf, inf)`.
    """
    def to_unconstrained(self, x):
        return x

    def to_constrained(self, u):
        return u


class Dependent(Constraint):
    """
    Placeholder for variables whose support depends on other variables.
    """
    def to_unconstrained(self, x):
        raise ValueError('Parameter cannot be transformed from an unconstraned space; '
                         'Try another parameterization to make parameters independent.')

    def to_constrained(self, u):
        raise ValueError('Parameter cannot be transformed to an unconstraned space; '
                         'Try another parameterization to make parameters independent.')


class Positive(Constraint):
    """
    Constraint to the positive half line `(0, inf)`.
    """
    def to_unconstrained(self, x):
        return torch.log(x)

    def to_constrained(self, u):
        return torch.exp(u)


class GreaterThan(Constraint):
    """
    Constraint to the positive half line `(0, inf)`.
    """
    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def to_unconstrained(self, x):
        return torch.log(x - self.lower_bound)

    def to_constrained(self, u):
        return torch.exp(u) + self.lower_bound


class Interval(Constraint):
    """
    Constraint to an interval `(lower_bound, upper_bound)`.
    """
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def to_unconstrained(self, x):
        unit = (x - self.lower_bound) / (self.upper_bound - self.lower_bound)
        return torch.log(unit / (1 - unit))

    def to_constrained(self, u):
        unit = torch.sigmoid(u)
        return self.lower_bound + unit * (self.upper_bound - self.lower_bound)


class Simplex(Constraint):
    """
    Constraint to the unit simplex in the innermost (rightmost) dimension.
    Specifically: `x > 0` and `x.sum(-1) == 1`.
    """
    def to_unconstrained(self, x):
        return torch.log(x)

    def to_constrained(self, u):
        return softmax(u, dim=-1)


# Functions and constants.
unconstrained = Unconstrained()
dependent = Dependent()
positive = Positive()
greater_than = GreaterThan
unit_interval = Interval(0, 1)
interval = Interval
simplex = Simplex()
