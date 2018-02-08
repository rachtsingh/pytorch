r"""
The ``torch.distributions`` package contains parameterizable probability distributions
and sampling functions, as well as other useful statistical methods.

Basics
------

A distribution instance is constructed using parameters of shape ``batch_shape + param_shape``,
and can be sampled from using the ``sample`` method, which returns
a ``torch.Tensor`` or ``Variable`` of shape ``sample_shape + batch_shape + event_shape``.

 - Batches over ``sample_shape`` are *independently, identically distributed*
 - Batches over ``batch_shape`` are *independent*, but differently distributed, depending on params
 - Batches over ``event_shape`` are generally dependent and correlated

Distributions parameterized by ``torch.Tensor`` give ``torch.Tensor`` samples, and those by ``Variable``
give ``Variable`` samples. *However*, backpropagation through sampling is not immediate. Consider a
parameterized distribution :math:`q(x; \lambda)` - sampling and forward propagating through some loss function
:math:`f(x)` is equivalent to estimating the expectation :math:`\mathbb{E}_{x \sim q(x; \lambda)}[f(x)]`. In
general, there are two ways to compute the gradient of this quantity with respect to :math:`\lambda`:

1. A basic technique is REINFORCE or the score function method which can be implemented using the
:meth:`~torch.distributions.Distribution.log_prob` method, when the probability density function is
differentiable with respect to its parameters:

.. math::
    \nabla_\lambda \mathbb{E}_{x \sim q(x; \lambda)}[f(x)] =
        \mathbb{E}_{x \sim q}[f(x) \cdot \nabla_\lambda \log q(x; \lambda)]

Letting :math:`x` be the action taken under a policy and :math:`f` be the corresponding reward gives a way
to implement policy gradient methods.

.. math::
    \Delta\theta  = \alpha r \frac{\partial\log p(a|\pi^\theta(s))}{\partial\theta}

where :math:`\theta` are the parameters, :math:`\alpha` is the learning rate, :math:`r` is the reward
and :math:`p(a|\pi^\theta(s))` is the probability of taking action :math:`a` in state :math:`s`
given policy :math:`\pi^\theta`.

In practice we would sample an action from the output of a network, apply this
action in an environment, and then use ``log_prob`` to construct an equivalent
loss function. Note that we use a negative because optimisers use gradient
descent, whilst the rule above assumes gradient ascent. With a categorical
policy, the code for implementing REINFORCE would be as follows::

    probs = policy_network(state)
    # NOTE: this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = m.sample()
    next_state, reward = env.step(action)
    loss = -m.log_prob(action) * reward
    loss.backward()

2. Another way to implement these stochastic/policy gradients would be to use the reparameterization
trick via the :meth:`~torch.distributions.Distribution.rsample` method, where the random variable
can be defined as a parameterized deterministic function of a parameter-free random variable.
Reparameterized samples are differentiable with respect to the parameters of their distribution, and
are limited to distributions which have the ``rsample`` method. Code for implementing the pathwise
gradient estimate would be as follows::

    params = policy_network(state)
    m = Normal(*params)
    # any distribution with .has_rsample == True can work
    action = m.rsample()
    next_state, reward = env.step(action)  # Assume that reward is differentiable
    loss = -reward
    loss.backward()

Advanced Features
-----------------

The ``torch.distributions`` package also implements KL divergences between many common distributions, their
entropies, and methods for transforming distributions. For example::

    a = Normal(loc=0, scale=1)
    b = Normal(loc=2, scale=1)
    KL = kl_divergence(a, b)

"""

from .bernoulli import Bernoulli
from .beta import Beta
from .transforms import *
from .binomial import Binomial
from .categorical import Categorical
from .cauchy import Cauchy
from .chi2 import Chi2
from .constraint_registry import biject_to, transform_to
from .dirichlet import Dirichlet
from .distribution import Distribution
from .exponential import Exponential
from .exp_family import ExponentialFamily
from .fishersnedecor import FisherSnedecor
from .gamma import Gamma
from .geometric import Geometric
from .gumbel import Gumbel
from .kl import kl_divergence, register_kl
from .laplace import Laplace
from .log_normal import LogNormal
from .multinomial import Multinomial
from .normal import Normal
from .one_hot_categorical import OneHotCategorical
from .pareto import Pareto
from .poisson import Poisson
from .studentT import StudentT
from .transformed_distribution import TransformedDistribution
from .uniform import Uniform

__all__ = [
    'Bernoulli',
    'Beta',
    'Binomial',
    'Categorical',
    'Cauchy',
    'Chi2',
    'Dirichlet',
    'Distribution',
    'Exponential',
    'ExponentialFamily',
    'FisherSnedecor',
    'Gamma',
    'Geometric',
    'Gumbel',
    'Laplace',
    'LogNormal',
    'Multinomial',
    'Normal',
    'OneHotCategorical',
    'Pareto',
    'StudentT',
    'Poisson',
    'Uniform',
    'TransformedDistribution',
    'biject_to',
    'kl_divergence',
    'register_kl',
    'transform_to',
]
__all__.extend(transforms.__all__)
