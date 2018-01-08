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
      cuda::CUDA_tensor_apply2<int64_t, float>(ret, lambda,
        [states] __device__ (int64_t& ret_val, const float& lambda, bool early_exit) {
          ret_val = curand_poisson(&states[blockIdx.x], lambda);
        }
      );
    }
  };

} // at::native::dist

Tensor _s_poisson_cuda(const Tensor& lambda, Generator* gen) {
  Tensor ret = lambda.type().toScalarType(kFloat).tensor(lambda.sizes());
  auto lambda_ = lambda.toType(ScalarType::Float);
  dispatch_all<void, dist::PoissonOpCUDA>(lambda_.type(), "poisson", ret, lambda_, dist::get_states(gen));
  return ret;
}

} // at::native
} // at
