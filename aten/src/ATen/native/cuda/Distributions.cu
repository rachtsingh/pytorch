#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#include <utility>

#include <THC/THCGeneral.h>
#include <THC/THCHalf.h>
#include <THC/THCApply.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCTensorRandom.h>

THCGenerator* THCRandom_getGenerator(THCState* state);

namespace at {
namespace native {

namespace dist {
  std::pair<uint64_t, uint64_t> get_philox_seed(Generator *gen) {
    auto gen_ = THCRandom_getGenerator(at::globalContext().thc_state);
    return std::make_pair(gen_->initial_seed, gen_->philox_seed_offset++);
  }

  // note that sample_poisson is adapted from Numpy's distributions.c
  // see Distributions.cpp for the license
  __device__ int64_t sample_poisson(float lambda, curandStatePhilox4_32_10_t *state) {
    if (lambda >= 10) {
      // transformed rejection method, (Hoermann, 1993)
      int64_t k;
      float U, V, a, b, invalpha, vr, us;

      float slam = ::sqrt(lambda);
      float loglam = ::log(lambda);
      b = 0.931 + 2.53 * slam;
      a = -0.059 + 0.02483 * b;
      invalpha = 1.1239 + 1.1328/(b-3.4);
      vr = 0.9277 - 3.6224/(b-2);

      while (1) {
        U = curand_uniform(state) - 0.5;
        V = curand_uniform(state);
        us = 0.5 - ::fabs(U);
        k = (int64_t) ::floor((2*a/us + b)*U + lambda + 0.43);
        if ((us >= 0.07) && (V <= vr)) {
          return k;
        }
        if ((k < 0) || ((us < 0.013) && (V > us))) {
          continue;
        }
        if ((::log(V) + ::log(invalpha) - ::log(a/(us*us)+b)) <= (-lambda + k*loglam - ::lgamma((float) k+1)))
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
      float prod, U, enlam;

      enlam = ::exp(-lambda);
      X = 0;
      prod = 1.0;
      while (1) {
        U = curand_uniform(state);
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
  
  template <typename scalar>
  struct PoissonOpCUDA {
    static void apply(Tensor& ret, const Tensor& lambda, std::pair<uint64_t, uint64_t> seeds) {
      at::cuda::CUDA_tensor_apply2<scalar, float>(ret, lambda,
        [seeds] __device__ (scalar& ret_val, const float& lambda, bool early_exit) {
          curandStatePhilox4_32_10_t state;
          curand_init(seeds.first, blockIdx.x * blockDim.x + threadIdx.x, seeds.second, &state);
          ret_val = scalar_cast<scalar>(sample_poisson(lambda, &state));
        }
      );
    }
  };

} // at::native::dist

Tensor _s_poisson_cuda(const Tensor& lambda, Generator* gen) {
  Tensor ret = lambda.type().tensor(lambda.sizes());
  auto lambda_ = lambda.toType(ScalarType::Float);
  dispatch_floating_types<void, dist::PoissonOpCUDA>(ret.type(), "poisson", ret, lambda_, dist::get_philox_seed(gen));
  return ret;
}

} // at::native
} // at
