r"""
The ``distributions`` package contains parameterizable probability distributions
and sampling functions.

Policy gradient methods can be implemented using the
:meth:`~torch.distributions.Distribution.log_prob` method, when the probability
density function is differentiable with respect to its parameters. A basic
method is the REINFORCE rule:

.. math::

    \Delta\theta  = \alpha r \frac{\partial\log p(a|\pi^\theta(s))}{\partial\theta}

where :math:`\theta` are the parameters, :math:`\alpha` is the learning rate,
:math:`r` is the reward and :math:`p(a|\pi^\theta(s))` is the probability of
taking action :math:`a` in state :math:`s` given policy :math:`\pi^\theta`.

In practice we would sample an action from the output of a network, apply this
action in an environment, and then use ``log_prob`` to construct an equivalent
loss function. Note that we use a negative because optimisers use gradient
descent, whilst the rule above assumes gradient ascent. With a multinomial
policy, the code for implementing REINFORCE would be as follows::

    probs = policy_network(state)
    m = Multinomial(probs)
    action = m.sample()
    next_state, reward = env.step(action)
    loss = -m.log_prob(action) * reward
    loss.backward()
"""
import math
from numbers import Number
import torch
from torch.autograd import Variable

__all__ = ['Distribution', 'Bernoulli', 'Multinomial', 'Normal']


def expanded_size(expand_size, orig_size):
    """Returns the expanded size given two sizes"""
    # strip leading 1s from original size
    if not expand_size:
        return orig_size
    # favor Normal(mean_1d, std_1d).sample(k).size()= (k, 1) instead of (k,) 
    #if orig_size == (1,):
    #    return expand_size
    else:
        return expand_size + orig_size

class Distribution(object):
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    def log_prob(self, value):
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor or Variable):
        """
        raise NotImplementedError

    def __init__(self, event_size, data_type, reparametrized=False):
        self._size = event_size
        self._type = data_type
        self._reparametrized = reparametrized

    @property
    def reparametrized(self):
        return self._reparametrized
		
    @property
    def type(self):
        return self._type

    @property
    def event_size(self):
        return self._size

    def sample(self, *sizes):
        """
        Generates a single sample or single batch of samples if the distribution
        parameters are batched.
        """
        raise NotImplementedError

    def prob(self, value):
        return torch.exp(self.log_prob(value))

    def log_cdf(self, value):
        return torch.log(self.cdf(value))

    def cdf(self, value):
        raise NotImplementedError

    def inv_cdf(self, value):
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def variance(self):
        raise NotImplementedError

    def covariance(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError
        
class Bernoulli(Distribution):
    r"""
    Creates a Bernoulli distribution parameterized by `probs`.

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Example::

        >>> m = Bernoulli(torch.Tensor([0.3]))
        >>> m.sample()  # 30% chance 1; 70% chance 0
         0.0
        [torch.FloatTensor of size 1]

    Args:
        probs (Tensor or Variable): the probabilty of sampling `1`
    """

    def __init__(self, probs):
        self.probs = probs
        if isinstance(probs, Number):
            super(Bernoulli, self).__init__((1,),
                                          'torch.FloatTensor')
        else:
            super(Bernoulli, self).__init__(probs.size(),
                                          probs.data.type())

    def sample(self, *sizes):
        return torch.bernoulli(self.probs.expand(*sizes, *self.probs.size()))

    def log_prob(self, value):
        # compute the log probabilities for 0 and 1
        log_pmf = (torch.stack([1 - self.probs, self.probs])).log()

        # evaluate using the values
        return log_pmf.gather(0, value.unsqueeze(0).long()).squeeze(0)


class Multinomial(Distribution):
    r"""
    Creates a multinomial distribution parameterized by `probs`.

    Samples are integers from `0 ... K-1` where `K` is probs.size(-1).

    If `probs` is 1D with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is 2D, it is treated as a batch of probability vectors.

    See also: :func:`torch.multinomial`

    Example::

        >>> m = Multinomial(torch.Tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
         3
        [torch.LongTensor of size 1]

    Args:
        probs (Tensor or Variable): event probabilities
    """

    def __init__(self, probs):
        self.probs = probs
        if isinstance(probs, Number):
            super(Multinomial, self).__init__((1,),
                                          'torch.FloatTensor')
        else:
            super(Multinomial, self).__init__(probs.size(),
                                          probs.data.type())

    def sample(self):
        return torch.multinomial(self.probs, 1, True).squeeze(-1)
    
    def sample_n(self, n):
        if n == 1:
            return self.sample().expand(1, 1)
        else:
            return torch.multinomial(self.probs, n, True).t()
        
    def log_prob(self, value):
        """Returns the probability mass, which is the probability of the argmax
        of the value under the corresponding Discrete distribution."""
        if value.data.type() != 'torch.LongTensor':
            _, value = value.max(-1)
        if value.dim() < len(self._size[:-1]):
            value = value.expand(*self._size[:-1])
        log_probs = self.probs.log()
        if value.dim() > len(self._size[:-1]):
            size = value.size() + (self._size[-1],)
            log_probs = self._log_probs.expand(*size)
        return log_probs.gather(-1, value.unsqueeze(-1)).squeeze(-1)

class Normal(Distribution):
    r"""The univariate normal distribution.

    .. math::
       f(x \mid \mu, \sigma) =
           \sqrt{\frac{1}{2\pi \sigma^2}}
           \exp \left[ -\frac{1}{2} \frac{(x-\mu)^2}{\sigma^2} \right]

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`\sigma^2` or :math:`\frac{1}{\tau}`
    ========  ==========================================

    Normal distribution can be parameterized either in terms of standard
    deviation or precision. The link between the two parametrizations is
    given by :math:`\tau = 1 / \sigma^2`

    Parameters:
        mu(:obj:`Variable`): Mean.
        sigma(:obj:`Variable`, optional): Standard deviation (sigma > 0).
        tau(:obj:`Variable`, optional): Precision (tau > 0).
        size(tuple, optional): Sample size.

    Attributes:
        mean(:obj:`Variable`): Mean (mu).
        mode(:obj:`Variable`): Mode (mu).
        variance(:obj:`Variable`): Variance (equal to sigma**2 or 1/tau)

    Note:
        Only one of sigma or tau can be specified. When neither is specified,
        the default is sigma = tau = 1.0
    """

    def __init__(self, mu, sigma=None, tau=None):
        if sigma is not None and tau is not None:
            raise ValueError("Either sigma or tau may be specified, not both.")
        if sigma is None and tau is None:
            sigma = 1.0
            tau = 1.0
        if tau is None:
            tau = sigma**-2
        if sigma is None:
            sigma = tau**-0.5
        self._mu = mu
        self._sigma = sigma
        self._tau = tau
        _mu0 = mu / sigma
        if isinstance(_mu0, Number):
            super(Normal, self).__init__((1,),
                                         'torch.FloatTensor', True)
        else:
            super(Normal, self).__init__(_mu0.size(),
                                         _mu0.data.type())

    def mean(self):
        return self._mu

    def mode(self):
        return self._mu

    def variance(self):
        return self._sigma**2

    def sample(self, *sizes):
        size = expanded_size(sizes, self._size)
        eps = Variable(torch.randn(*size).type(self._type))
        return self._mu + self._sigma * eps

    def log_prob(self, value):
        # TODO: hopefully this goes away soon
        log = math.log if isinstance(self._sigma, Number) else torch.log
        return -0.5 * (log(2 * math.pi * self._sigma**2) +
                       ((value - self._mu) / self._sigma)**2)
        #var = (self._sigma ** 2)
        #log_std = math.log(self._sigma) if isinstance(self._sigma, Number) else self._sigma.log()
        #return -((value - self._mu) ** 2) / (2 * var) - log_std - math.log(math.sqrt(2 * math.pi))