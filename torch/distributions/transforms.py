from __future__ import division

import torch
from torch.distributions import constraints
from torch.distributions.utils import tril_mask
from torch.nn.functional import sigmoid, softmax

__all__ = ['transform', 'register_transform', 'Transform']

_TRANSFORMS = {}


def transform(constraint):
    """
    Looks up a pair of transforms to- and from- unconstrained space, given
    a constraint. Usage::

        constraint = Normal.params['scale']
        scale = transform(constraint).to_constrained(torch.zeros(1))
        u = transform(constraint).to_unconstrained(scale)
    """
    try:
        Trans = _TRANSFORMS[type(constraint)]
    except KeyError:
        raise NotImplementedError(
            'Cannot transform {} constraints'.format(type(constraint).__name__))
    return Trans(constraint)


def register_transform(constraint_class):
    """
    Decorator to register a constraint class with the
    `torch.distributions.transforms.transform()` function. Usage::

        @register_transform(MyConstraintClass)
        class MyTransform(Transform):
            def to_unconstrained(self, x):
                ...
            def to_constrained(self, u):
                ...
    """

    def decorator(transform_class):
        _TRANSFORMS[constraint_class] = transform_class
        return transform_class

    return decorator


class Transform(object):
    """
    Each constraint class registers a pseudoinverse pair of transforms
    `to_unconstrained` and `from_unconstrained`. These allow standard
    parameters to be transformed to an unconstrained space for optimization and
    transformed back after optimization. Note that these are not necessarily
    inverse pairs since the unconstrained space may have extra dimensions that
    are projected out; only the one-sided inverse equation is guaranteed::

        x == c.to_constrained(c.to_unconstrained(x))

    """
    def __init__(self, constraint):
        self.constraint = constraint

    def to_unconstrained(self, x):
        """
        Transform from constrained coordinates to unconstrained coordinates.
        """
        raise NotImplementedError

    def to_constrained(self, u):
        """
        Transform from unconstrained coordinates to constrained coordinates.
        """
        raise NotImplementedError


@register_transform(constraints.Unconstrained)
class IdentityTransform(Transform):
    """
    Identity transform for arbitrary real-valued data.
    """
    def to_unconstrained(self, x):
        return x

    def to_constrained(self, u):
        return u


@register_transform(constraints.Positive)
class LogExpTransform(Transform):
    """
    Transform from the positive reals and back via `log()` and `exp()`.
    """
    def to_unconstrained(self, x):
        return torch.log(x)

    def to_constrained(self, u):
        return torch.exp(u)


@register_transform(constraints.Interval)
class SigmoidLogitTransform(Transform):
    """
    Transform from an arbitrary interval and back via an affine transform and
    the `logit()` and `sigmoid()` functions.
    """
    def to_unconstrained(self, x):
        c = self.constraint
        unit = (x - c.lower_bound) / (c.upper_bound - c.lower_bound)
        return torch.log(unit / (1 - unit))

    def to_constrained(self, u):
        c = self.constraint
        unit = torch.sigmoid(u)
        return c.lower_bound + unit * (c.upper_bound - c.lower_bound)


@register_transform(constraints.Simplex)
class LogSoftmaxTransform(Transform):
    """
    Transform from the unit simplex and back via `log()` and `softmax()`.
    """
    def to_unconstrained(self, x):
        return torch.log(x)

    def to_constrained(self, u):
        return softmax(u, dim=-1)


@register_transform(constraints.LowerTriangular)
class LowerTriangularTransform(Transform):
    """
    Transform from lower-triangular square matrices of size `(n,n)` to
    contiguous vectors of size `m = n*(n+1)/2`. Dimensions left of the
    rightmost shape `(n,n)` or `(m,)` are preserved.
    """
    def to_unconstrained(self, x):
        mask = tril_mask(x)
        n = x.size(-1)
        m = n * (n + 1) // 2
        return x[mask].view(x.shape[-2:] + (m,))

    def to_constrained(self, u):
        n = int(round(((8 * u.size(-1) + 1)**0.5 + 1) / 2))
        x = u.new(n).zero_()
        x[tril_mask(x)] = u
        return x
