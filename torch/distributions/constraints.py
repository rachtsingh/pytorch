import torch
from torch.nn.functional import sigmoid, softmax


class Constraint(object):
    """
    Abstract base class for constraints.

    A constraint object represents a region over which a continuous variable is
    valid, e.g. within which a variable can optimized.
    """
    def __call__(self, value):
        """
        Returns a byte tensor of sample_shape + batch_shape indicating whether
        each value satisfies this constraint.
        """
        raise NotImplementedError


class Unconstrained(Constraint):
    """
    Trivial constraint to the extended real line `[-inf, inf]`.
    """
    def __call__(self, value):
        return value == value  # False for NANs.


class Dependent(Constraint):
    """
    Placeholder for variables whose support depends on other variables.
    """
    def __call__(self, x):
        raise ValueError('Cannot determine validity of dependent constraint')


class Positive(Constraint):
    """
    Constraint to the positive half line `[0, inf]`.
    """
    def __call__(self, value):
        return value >= 0


class Interval(Constraint):
    """
    Constraint to an interval `[lower_bound, upper_bound]`.
    """
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, value):
        return (self.lower_bound <= value) & (value <= self.upper_bound)


class Simplex(Constraint):
    """
    Constraint to the unit simplex in the innermost (rightmost) dimension.
    Specifically: `x >= 0` and `x.sum(-1) == 1`.
    """
    def __call__(self, value):
        return (value >= 0) & ((value.sum(-1, True) - 1).abs() < 1e-6)


# Functions and constants are the recommended interface.
unconstrained = Unconstrained()
dependent = Dependent()
positive = Positive()
unit_interval = Interval(0, 1)
interval = Interval
simplex = Simplex()
