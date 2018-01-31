#include "ATen/ATen.h"
#include "ATen/CPUApplyUtils.h"
#include "ATen/Dispatch.h"
#include "ATen/Config.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"

#include "ATen/CPUGenerator.h"
#include "ATen/CheckGenerator.h"
#include "ATen/Generator.h"

#include <functional>

#include "TH/THRandom.h"

namespace at {
namespace native {

Tensor& bernoulli_(Tensor& self, const Tensor& p, Generator* generator) {
  self.copy_(at::bernoulli(std::get<0>(expand_inplace(self, p)), generator));
  return self;
}

Tensor& bernoulli_(Tensor& self, double p, Generator* generator) {
  Tensor probs = self.type().toScalarType(kDouble).tensor({}).fill_(p);
  return native::bernoulli_(self, probs, generator);
}


// TODO Replace this with more accurate digamma().
template <typename scalar>
static inline scalar digamma_one(scalar x) {
  const double eps = x * 1e-3;
  return (std::lgamma(x + eps) - std::lgamma(x - eps)) / (eps + eps);
}

// Computes the reparameterized gradient -(d/dalpha cdf(x;alpha)) / pdf(x;alpha)
// for random number x drawn from a standard Gamma distribution Gamma(alpha).
template <typename scalar>
static inline scalar standard_gamma_grad_one(scalar alpha, scalar x) {
  // Use a Taylor series expansion for small x.
  if (x < 0.8f) {
    scalar numer = 1;
    scalar denom = alpha;
    auto series1 = numer / denom;
    auto series2 = numer / (denom * denom);
    for (int i = 1; i <= 5; ++i) {
      numer *= -x / i;
      denom += 1;
      series1 += numer / denom;
      series2 += numer / (denom * denom);
    }
    const auto pow_x_alpha = std::pow(x, alpha);
    const auto gamma_pdf = std::pow(x, alpha - 1) * std::exp(-x);
    const auto gamma_cdf = pow_x_alpha * series1;
    const auto gamma_cdf_alpha = (std::log(x) - digamma_one(alpha)) * gamma_cdf
        - pow_x_alpha * series2;
    const auto result = -gamma_cdf_alpha / gamma_pdf;
    return std::isnan(result) ? 0 : result;
  }

  // Use a Rice saddle point expansion for large alpha.
  if (alpha > 8.0f) {
    if (0.9f * alpha <= x && x <= 1.1f * alpha) {
      const auto numer_1 = 1 + 24 * alpha * (1 + 12 * alpha);
      const auto numer_2 = 1440 * (alpha * alpha) + 6 * x * (53 - 120 * x)
          - 65 * x * x / alpha + alpha * (107 + 3600 * x);
      const auto denom = 1244160 * (alpha * alpha) * (alpha * alpha);
      return numer_1 * numer_2 / denom;
    }
    const auto denom = std::sqrt(8 * alpha);
    const auto term2 = denom / (alpha - x);
    const auto term3 = std::pow(x - alpha - alpha * std::log(x / alpha), -1.5f);
    const auto term23 = (x < alpha) ? term2 - term3 : term2 + term3;
    const auto term1 = std::log(x / alpha) * term23
                     - std::sqrt(2 / alpha) * (alpha + x) / ((alpha - x) * (alpha - x));
    const auto stirling = 1 + 1 / (12 * alpha) * (1 + 1 / (24 * alpha));
    const auto numer = x * term1;
    return -stirling * numer / denom;
  }

  // Use a bivariate rational approximation to the reparameterized gradient.
  const auto u = std::log(x / alpha);
  const auto v = std::log(alpha);
  static const scalar coef_uv[3][8] = {
    {0.16009398, -0.094634809, 0.025146376, -0.0030648343,
     1, 0.32668115, 0.10406089, 0.0014179084},
    {0.53487893, 0.1298071, 0.065735949, -0.0015649758,
     0.16639465, 0.020070113, -0.0035938915, -0.00058392623},
    {0.040121004, -0.0065914022, -0.0026286047, -0.0013441777,
     0.017050642, -0.0021309326, 0.00085092367, -1.5247877e-07},
  };
  scalar coef_v[8];
  for (int i = 0; i < 8; ++ i) {
    coef_v[i] = coef_uv[0][i] + u * (coef_uv[1][i] + u * coef_uv[2][i]);
  }
  const auto p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
  const auto q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
  return std::exp(p / q);
}

template <typename scalar>
struct StandardGammaGradOp {
  static void apply(Tensor& ret, const Tensor& self, const Tensor& output) {
    CPU_tensor_apply3<scalar, scalar, scalar>(ret, self, output,
      [](scalar& ret_val, const scalar& self_val, const scalar &output_val) {
         ret_val = standard_gamma_grad_one(self_val, output_val);
      }
    );
  }
};

Tensor _standard_gamma_grad_cpu(const Tensor& self, const Tensor& output) {
  Tensor ret = self.type().tensor(self.sizes());
  dispatch_floating_types<void, StandardGammaGradOp>(self.type(), "_standard_gamma_grad", ret, self, output);
  return ret;
}

Tensor _standard_gamma_grad_cuda(const Tensor& self, const Tensor& output) {
  runtime_error("_standard_gamma_grad is not implemented for CUDA types");
}

/*
 * This section is a counterpart to Distributions.cu
 */

namespace dist {

#if !AT_CUDA_ENABLED()
  template<typename precision_t>
  struct baseSampler {
    std::function<precision_t(void)> sampler;
    baseSampler(std::function<precision_t(void)> sampler): sampler(sampler) {}
    precision_t sample() {
      return sampler();
    }
  };
#endif
  
  // The functions `sample_poisson`, `sample_gamma`
  // are adapted from Numpy's distributions.c implementation.
  // It is MIT licensed, so here is the copyright:

  /* Copyright 2005 Robert Kern (robert.kern@gmail.com)
   *
   * Permission is hereby granted, free of charge, to any person obtaining a
   * copy of this software and associated documentation files (the
   * "Software"), to deal in the Software without restriction, including
   * without limitation the rights to use, copy, modify, merge, publish,
   * distribute, sublicense, and/or sell copies of the Software, and to
   * permit persons to whom the Software is furnished to do so, subject to
   * the following conditions:
   *
   * The above copyright notice and this permission notice shall be included
   * in all copies or substantial portions of the Software.
   *
   * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
   * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
   */

  
  template<typename precision_t>
#if AT_CUDA_ENABLED()
  __host__ __device__
#endif
  precision_t sample_gamma(precision_t alpha, baseSampler<precision_t>& standard_uniform, baseSampler<precision_t>& standard_normal) {

    precision_t scale = 1.0;

		// Boost alpha for higher acceptance probability.
		if (alpha < 1.0) {
			scale *= ::pow(1 - standard_uniform.sample(), 1.0 / alpha);
			alpha += 1.0;
		}

		// This implements the acceptance-rejection method of Marsaglia and Tsang (2000)
		// doi:10.1145/358407.358414
		const precision_t d = alpha - 1.0 / 3.0;
		const precision_t c = 1.0 / ::sqrt(9.0 * d);
		for (;;) {
			precision_t x, y;
			do {
				x = standard_normal.sample();
				y = 1.0 + c * x;
			} while (y <= 0);
			const precision_t v = y * y * y;
			const precision_t u = 1 - standard_uniform.sample();
			const precision_t xx = x * x;
			if (u < 1.0 - 0.0331 * xx * xx)
				return scale * d * v;
			if (::log(u) < 0.5 * xx + d * (1.0 - v + ::log(v)))
				return scale * d * v;
		}
	}

  THGenerator * get_generator(Generator *gen) {
    auto default_gen = &at::globalContext().defaultGenerator(Backend::CPU);
    auto gen_ = check_generator<CPUGenerator>(gen, default_gen);
    return gen_->generator;
  }

  template <typename scalar>
  struct GammaOp {
    static void apply(Tensor& ret, const Tensor& alpha, THGenerator *generator) {
      CPU_tensor_apply2<scalar, double>(ret, alpha,
        [generator](scalar& ret_val, const double& alpha){
          dist::baseSampler<float> standard_uniform([generator] () {
            return THRandom_standard_uniform(generator);
          });
          dist::baseSampler<float> standard_normal([generator] () {
            return THRandom_normal(generator, 0.0, 1.0);
          });
          auto sample = dist::sample_gamma<float>(alpha, standard_uniform, standard_normal);
          ret_val = std::max(std::numeric_limits<scalar>::min(), (scalar) sample);
        }
      );
    }
  };

  template <typename scalar>
  struct PoissonOp {
    static int64_t sample_poisson(double lambda, THGenerator *generator) {
      if (lambda >= 10) {
        // transformed rejection method, (Hoermann, 1993)
        int64_t k;
        double U, V, a, b, invalpha, vr, us;

        double slam = std::sqrt(lambda);
        double loglam = std::log(lambda);
        b = 0.931 + 2.53 * slam;
        a = -0.059 + 0.02483 * b;
        invalpha = 1.1239 + 1.1328/(b-3.4);
        vr = 0.9277 - 3.6224/(b-2);

        while (1) {
          U = THRandom_standard_uniform(generator) - 0.5;
          V = THRandom_standard_uniform(generator);
          us = 0.5 - std::fabs(U);
          k = (int64_t) std::floor((2*a/us + b)*U + lambda + 0.43);
          if ((us >= 0.07) && (V <= vr)) {
            return k;
          }
          if ((k < 0) || ((us < 0.013) && (V > us))) {
            continue;
          }
          if ((std::log(V) + std::log(invalpha) - std::log(a/(us*us)+b)) <= (-lambda + k*loglam - std::lgamma((double) k+1)))
          {
            return k;
          }
        }
      }
      else if (lambda == 0) {
        return 0;
      }
      else {
        int64_t X;
        double prod, U, enlam;

        enlam = std::exp(-lambda);
        X = 0;
        prod = 1.0;
        while (1) {
          U = THRandom_standard_uniform(generator);
          prod *= U;
          if (prod > enlam) {
            X += 1;
          }
          else {
            return X;
          }
        }
      }
    }

    static void apply(Tensor& ret, const Tensor& lambda, THGenerator *generator) {
      CPU_tensor_apply2<scalar, double>(ret, lambda,
        [generator](scalar& ret_val, const double& lambda){
          ret_val = sample_poisson(lambda, generator);
        }
      );
    }
  };
} // at::native::dist

Tensor _s_poisson_cpu(const Tensor& lambda, Generator *gen) {
  Tensor ret = lambda.type().zeros(lambda.sizes());
  auto lambda_ = lambda.toType(ScalarType::Double);
  dispatch_floating_types<void, dist::PoissonOp>(ret.type(), "poisson", ret, lambda_, dist::get_generator(gen));
  return ret;
}

Tensor _s_gamma_cpu(const Tensor& alpha, Generator *gen) {
  Tensor ret = alpha.type().zeros(alpha.sizes());
  auto alpha_ = alpha.toType(ScalarType::Double);
  dispatch_floating_types<void, dist::GammaOp>(ret.type(), "gamma", ret, alpha_, dist::get_generator(gen));
  return ret;
}

} // at::native
} // at
