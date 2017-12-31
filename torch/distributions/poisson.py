from numbers import Number

import torch
from torch.autograd import Variable
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


class Poisson(Distribution):
    r"""
    Creates a Poisson distribution parameterized by `_lambda`, the rate parameter.
    Currently, we use _lambda instead of lambda because the latter is a keyword.

    Samples are nonnegative integers, with a pmf given by 
    $\lambda^k e^{-\lambda}/k!$

    Example::

        >>> m = Poisson(torch.Tensor([4]))
        >>> m.sample()
         3
        [torch.LongTensor of size 1]

    Args:
        lambda (Number, Tensor or Variable): the rate parameter
    """

    def __init__(self, _lambda):
        self._lambda, = broadcast_all(_lambda)
        if isinstance(_lambda, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self._lambda.size()
        super(Poisson, self).__init__(batch_shape)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return torch.poisson(self._lambda.expand(shape))

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        param_shape = value.size()
        _lambda = self._lambda.expand(param_shape)
        return (_lambda.log() * value) - _lambda - (value + 1).lgamma()