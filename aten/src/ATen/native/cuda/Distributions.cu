#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include "ATen/Context.h"
#include "ATen/CUDAGenerator.h"
#include "ATen/CheckGenerator.h"

#include <curand.h>
#include <curand_kernel.h>

#include "THC/THCNumerics.cuh"
#include "THC/THCTensorRandom.h"
#include "THCTensor.h"

// The functions `sample_poisson`, `sample_gamma`, `sample_dirichlet`
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

THCGenerator* THCRandom_getGenerator(THCState* state);

namespace at {
namespace native {

namespace dist {
  curandStateMtgp32* get_states(Generator *gen) {
    auto gen_ = THCRandom_getGenerator(at::globalContext().thc_state);
    return gen_->gen_states;
  }

  // ignored
  template <typename scalar>
  struct PoissonOpCUDA {
    static void apply(Tensor& ret, const Tensor& lambda, curandStateMtgp32 *states) {
      cuda::CUDA_tensor_apply2<int64_t, double>(ret, lambda,
        [states] __device__ (int64_t& ret_val, const double& lambda, bool early_exit) {
          ret_val = curand_poisson(&states[blockIdx.x], lambda);
        }
      );
    }
  };

} // at::native::dist

Tensor _s_poisson_cuda(const Tensor& lambda, Generator* gen) {
  Tensor ret = lambda.type().toScalarType(kDouble).zeros(lambda.sizes());
  auto lambda_ = lambda.toType(ScalarType::Double);
  dispatch_all<void, dist::PoissonOpCUDA>(lambda_.type(), "poisson", ret, lambda_, dist::get_states(gen));
  return ret;
}

} // at::native
} // at
