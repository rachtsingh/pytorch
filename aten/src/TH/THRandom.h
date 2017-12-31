#ifndef TH_RANDOM_INC
#define TH_RANDOM_INC

#include "THGeneral.h"

#define _MERSENNE_STATE_N 624
#define _MERSENNE_STATE_M 397
/* A THGenerator contains all the state required for a single random number stream */
typedef struct THGenerator {
  /* The initial seed. */
  uint64_t the_initial_seed;
  int left;  /* = 1; */
  int seeded; /* = 0; */
  uint64_t next;
  uint64_t state[_MERSENNE_STATE_N]; /* the array for the state vector  */
  /********************************/

  /* For normal distribution */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid; /* = 0; */
} THGenerator;

#define torch_Generator "torch.Generator"

/* Manipulate THGenerator objects */
TH_API THGenerator * THGenerator_new(void);
TH_API THGenerator * THGenerator_copy(THGenerator *self, THGenerator *from);
TH_API void THGenerator_free(THGenerator *gen);

/* Checks if given generator is valid */
TH_API int THGenerator_isValid(THGenerator *_generator);

/* Initializes the random number generator from /dev/urandom (or on Windows
platforms with the current time (granularity: seconds)) and returns the seed. */
TH_API uint64_t THRandom_seed(THGenerator *_generator);

/* Initializes the random number generator with the given int64_t "the_seed_". */
TH_API void THRandom_manualSeed(THGenerator *_generator, uint64_t the_seed_);

/* Returns the starting seed used. */
TH_API uint64_t THRandom_initialSeed(THGenerator *_generator);

/* Generates a uniform 32 bits integer. */
TH_API uint64_t THRandom_random(THGenerator *_generator);

/* Generates a uniform 64 bits integer. */
TH_API uint64_t THRandom_random64(THGenerator *_generator);

/* Generates a uniform random double on [0,1). */
TH_API double THRandom_uniform(THGenerator *_generator, double a, double b);

/* Generates a uniform random float on [0,1). */
TH_API float THRandom_uniformFloat(THGenerator *_generator, float a, float b);

/** Generates a random number from a normal distribution.
    (With mean #mean# and standard deviation #stdv >= 0#).
*/
TH_API double THRandom_normal(THGenerator *_generator, double mean, double stdv);

/** Generates a random number from an exponential distribution.
    The density is $p(x) = lambda * exp(-lambda * x)$, where
    lambda is a positive number.
*/
TH_API double THRandom_exponential(THGenerator *_generator, double lambda);

/** Generates a random number from a standard Gamma distribution.
    The Gamma density is proportional to $x^{alpha-1} exp(-x)$
    The shape parameter alpha (a.k.a. k) is a positive real number.
*/
TH_API double THRandom_standard_gamma(THGenerator *_generator, double alpha);

/** Generates a random number from a Poisson distribution.
    The Poisson pmf is proportional to $lambda^k e^{-lambda}/k!$
    The rate parameter lambda is a positive real number.
*/
TH_API int THRandom_poisson(THGenerator *_generator, double lambda);

/** Returns a random number from a Cauchy distribution.
    The Cauchy density is $p(x) = sigma/(pi*(sigma^2 + (x-median)^2))$
*/
TH_API double THRandom_cauchy(THGenerator *_generator, double median, double sigma);

/** Generates a random number from a log-normal distribution.
    (#mean > 0# is the mean of the log-normal distribution
    and #stdv# is its standard deviation).
*/
TH_API double THRandom_logNormal(THGenerator *_generator, double mean, double stdv);

/** Generates a random number from a geometric distribution.
    It returns an integer #i#, where $p(i) = (1-p) * p^(i-1)$.
    p must satisfy $0 < p < 1$.
*/
TH_API int THRandom_geometric(THGenerator *_generator, double p);

/* Returns true with probability $p$ and false with probability $1-p$ (p > 0). */
TH_API int THRandom_bernoulli(THGenerator *_generator, double p);
#endif
