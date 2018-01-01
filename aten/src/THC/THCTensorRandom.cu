#include "THCTensorRandom.h"
#include "THCDeviceUtils.cuh"
#include "THCGeneral.h"
#include "THCTensorCopy.h"
#include "THCTensorMath.h"
#include "THCReduceApplyUtils.cuh"
#include "THCTensorRandom.cuh"

#include <thrust/functional.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256


Generator* THCRandom_getGenerator(THCState* state);

/* Sets up generator. Allocates but does not create the generator states. */
__host__ void initializeGenerator(THCState *state, Generator* gen)
{
  THCudaCheck(THCudaMalloc(state, (void**)&gen->gen_states, MAX_NUM_BLOCKS * sizeof(curandStateMtgp32)));
  THCudaCheck(THCudaMalloc(state, (void**)&gen->kernel_params, sizeof(mtgp32_kernel_params)));
}

/* Creates a new generator state given the seed. */
__host__ void createGeneratorState(Generator* gen, uint64_t seed)
{
  if (curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, gen->kernel_params) != CURAND_STATUS_SUCCESS)
  {
    THError("Creating MTGP constants failed.");
  }
  if (curandMakeMTGP32KernelState(gen->gen_states, mtgp32dc_params_fast_11213,
                                  gen->kernel_params, MAX_NUM_BLOCKS, seed) != CURAND_STATUS_SUCCESS)
  {
    THError("Creating MTGP kernel state failed.");
  }
}

__host__ void THCRandom_getRNGState(THCState* state, THByteTensor *rng_state)
{
  Generator* gen = THCRandom_getGenerator(state);

  // The RNG state comprises the MTPG32 states and the seed.
  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(gen->initial_seed);
  static const size_t total_size = states_size + seed_size;
  THByteTensor_resize1d(rng_state, total_size);
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");
  THCudaCheck(cudaMemcpy(THByteTensor_data(rng_state), gen->gen_states,
                         states_size, cudaMemcpyDeviceToHost));
  memcpy(THByteTensor_data(rng_state) + states_size, &gen->initial_seed, seed_size);
}

__global__ void set_rngstate_kernel(curandStateMtgp32 *state, mtgp32_kernel_params *kernel)
{
  state[threadIdx.x].k = kernel;
}

__host__ void THCRandom_setRNGState(THCState* state, THByteTensor *rng_state)
{
  Generator* gen = THCRandom_getGenerator(state);

  static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(gen->initial_seed);
  static const size_t total_size = states_size + seed_size;
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");

  THCudaCheck(cudaMemcpy(gen->gen_states, THByteTensor_data(rng_state),
                         states_size, cudaMemcpyHostToDevice));
  set_rngstate_kernel<<<1, MAX_NUM_BLOCKS, 0, THCState_getCurrentStream(state)>>>(
      gen->gen_states, gen->kernel_params);
  memcpy(&gen->initial_seed, THByteTensor_data(rng_state) + states_size, seed_size);
}

// Goes from (0, 1] to [0, 1). Note 1-x is not sufficient since for some floats
// eps near 0, 1-eps will round to 1.
template <typename T>
__device__ inline T reverse_bounds(T value) {
  if (THCNumerics<T>::eq(value, ScalarConvert<int, T>::to(1))) {
    return ScalarConvert<int, T>::to(0);
  }
  return value;
}

// This function is adapted from Numpy's distributions.c file.
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

__device__ inline int64_t sample_poisson(curandStateMtgp32 *state, double lambda) {
	if (lambda >= 10) {
    // transformed rejection method, (Hoermann, 1993)
    int64_t k;
    double U, V, slam, loglam, a, b, invalpha, vr, us;

    slam = sqrt(lambda);
    loglam = log(lambda);
    b = 0.931 + 2.53 * slam;
    a = -0.059 + 0.02483 * b;
    invalpha = 1.1239 + 1.1328/(b-3.4);
    vr = 0.9277 - 3.6224/(b-2);

    while (1) {
      U = curand_uniform_double(state) - 0.5;
      V = curand_uniform_double(state);
      us = 0.5 - fabs(U);
      k = (int64_t) floor((2*a/us + b)*U + lambda + 0.43);
      if ((us >= 0.07) && (V <= vr)) {
        return k;
      }
      if ((k < 0) || ((us < 0.013) && (V > us))) {
        continue;
      }
      if ((log(V) + log(invalpha) - log(a/(us*us)+b)) <= (-lambda + k*loglam - lgamma((double) k+1)))
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

    enlam = exp(-lambda);
    X = 0;
    prod = 1.0;
    while (1) {
      U = curand_uniform_double(state);
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


#ifdef CUDA_HALF_TENSOR
__device__ inline half half_uniform_scale_and_shift(float x, double a, double b) {
  half width = ScalarConvert<double, half>::to(b - a);
  half start = ScalarConvert<double, half>::to(a);
  half scaled = THCNumerics<half>::mul(reverse_bounds(ScalarConvert<float, half>::to(x)), width);
  return THCNumerics<half>::add(scaled, start);
}
#endif

#define GENERATE_KERNEL1(NAME, T, ARG1, CURAND_T, CURAND_FUNC, TRANSFORM)           \
__global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1)           \
{                                                                                   \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                  \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                     \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {           \
    CURAND_T x = CURAND_FUNC(&state[blockIdx.x]);                                   \
    if (i < size) {                                                                 \
      T y = TRANSFORM;                                                              \
      result[i] = y;                                                                \
    }                                                                               \
  }                                                                                 \
}

#define GENERATE_KERNEL2(NAME, T, ARG1, ARG2, CURAND_T, CURAND_FUNC, TRANSFORM)     \
__global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1, ARG2)     \
{                                                                                   \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                  \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                     \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {           \
    CURAND_T x = CURAND_FUNC(&state[blockIdx.x]);                                   \
    if (i < size) {                                                                 \
      T y = TRANSFORM;                                                              \
      result[i] = y;                                                                \
    }                                                                               \
  }                                                                                 \
}

#define GENERATE_KERNEL3(NAME, T, ARG1_T, ARG1, SAMPLE_FUNC, TRANSFORM)             \
__global__ void NAME(curandStateMtgp32 *state, int size, T *result, ARG1_T *ARG1)   \
{                                                                                   \
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;                                  \
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;                     \
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {           \
    double ARG1##_double = ScalarConvert<ARG1_T, double>::to(ARG1[i]);              \
    T x = SAMPLE_FUNC(&state[blockIdx.x], ARG1##_double);                           \
    if (i < size) {                                                                 \
      result[i] = x;                                                                \
    }                                                                               \
  }                                                                                 \
}                                                                                   

template<typename T, typename U>
struct is_same { static const bool value = false; };

template<typename T>
struct is_same<T, T> { static const bool value = true; };

template<typename real, typename prob_type>
__global__ void generate_bernoulli_tensor(curandStateMtgp32 *state, int size,
        real *result, prob_type *probs)
{
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int rounded_size = THCCeilDiv(size, BLOCK_SIZE) * BLOCK_SIZE;
  for (int i = idx; i < rounded_size; i += BLOCK_SIZE * MAX_NUM_BLOCKS) {
    if (is_same<prob_type, double>::value) {
      double x = curand_uniform_double(&state[blockIdx.x]);
      if (i < size)
        result[i] = ScalarConvert<bool, real>::to(x <= probs[i]);
    } else {
      float x = curand_uniform(&state[blockIdx.x]);
      if (i < size)
        result[i] = ScalarConvert<bool, real>::to(x <= probs[i]);
    }
  }
}

// NOTE: curand_uniform is (0, 1] and we want [a, b)
GENERATE_KERNEL2(generate_uniform, float, float a, float b, float, curand_uniform, reverse_bounds(x) * (b-a) + a)
GENERATE_KERNEL2(generate_uniform, double, double a, double b, double, curand_uniform_double, reverse_bounds(x) * (b-a) + a)

GENERATE_KERNEL2(generate_normal, float, double mean, double stdv, float, curand_normal, (x * stdv) + mean)
GENERATE_KERNEL2(generate_normal, double, double mean, double stdv, double, curand_normal_double, (x * stdv) + mean)

GENERATE_KERNEL1(generate_exponential, float, double lambda, float, curand_uniform, (float)(-1. / lambda * log(x)))
GENERATE_KERNEL1(generate_exponential, double, double lambda, double, curand_uniform_double, (double)(-1. / lambda * log(x)))

GENERATE_KERNEL2(generate_cauchy, float, double median, double sigma, float, curand_uniform, (float)(median + sigma * tan(M_PI*(x-0.5))))
GENERATE_KERNEL2(generate_cauchy, double, double median, double sigma, double, curand_uniform_double, (double)(median + sigma * tan(M_PI*(x-0.5))))

GENERATE_KERNEL3(generate_poisson, int64_t, float, lambda, sample_poisson, y)
GENERATE_KERNEL3(generate_poisson, int64_t, double, lambda, sample_poisson, y)

#ifdef CUDA_HALF_TENSOR
GENERATE_KERNEL2(generate_uniform, half, double a, double b, float, curand_uniform, (half_uniform_scale_and_shift(x, a, b)))
GENERATE_KERNEL2(generate_normal, half, double mean, double stdv, float, curand_normal, (ScalarConvert<float, half>::to((x * stdv) + mean)))
GENERATE_KERNEL1(generate_exponential, half, double lambda, float, curand_uniform, (ScalarConvert<float, half>::to((float)(-1. / lambda * log(x)))))
GENERATE_KERNEL2(generate_cauchy, half, double median, double sigma, float, curand_uniform, (ScalarConvert<float, half>::to((float)(median + sigma * tan(M_PI*(x-0.5))))))
GENERATE_KERNEL3(generate_poisson, int64_t, half, lambda, sample_poisson, y)
#endif // CUDA_HALF_TENSOR

#include "generic/THCTensorRandom.cu"
#include "THCGenerateAllTypes.h"

#undef GENERATE_KERNEL1
#undef GENERATE_KERNEL2
#undef GENERATE_KERNEL3
