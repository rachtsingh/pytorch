#include "ATen/NativeFunctions.h"
#include "ATen/Dispatch.h"
#include "ATen/cuda/CUDAApplyUtils.cuh"
#include <curand.h>
#include <curand_kernel.h>

#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <THC/THCHalf.h>
#include <THC/THCTensorCopy.h>
#include <THC/THCApply.cuh>
#include <THC/THCNumerics.cuh>
#include <THC/THCReduce.cuh>
#include <THC/THCTensorRandom.h>

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>


THCGenerator* THCRandom_getGenerator(THCState* state);

namespace at {
namespace native {

namespace dist {
  curandStateMtgp32* get_states(Generator *gen) {
    auto gen_ = THCRandom_getGenerator(at::globalContext().thc_state);
    return gen_->gen_states;
  }
  
  template <typename scalar>
  struct PoissonOpCUDA {
    static void apply(Tensor& ret, const Tensor& lambda, curandStateMtgp32 *states) {
      at::cuda::CUDA_tensor_apply2<scalar, float>(ret, lambda,
        [states] __device__ (scalar& ret_val, const float& lambda, bool early_exit) {
          ret_val = scalar_cast<scalar>(curand_poisson(&states[blockIdx.x], lambda));
        }
      );
    }
  };
} // at::native::dist

Tensor _s_poisson_cuda(const Tensor& lambda, Generator* gen) {
  Tensor ret = lambda.type().tensor(lambda.sizes());
  auto lambda_ = lambda.toType(ScalarType::Float);
  dispatch_all<void, dist::PoissonOpCUDA>(ret.type(), "poisson", ret, lambda_, dist::get_states(gen));
  return ret;
}

} // at::native
} // at
