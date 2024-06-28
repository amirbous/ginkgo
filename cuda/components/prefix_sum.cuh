// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_COMPONENTS_PREFIX_SUM_CUH_
#define GKO_CUDA_COMPONENTS_PREFIX_SUM_CUH_


#include <type_traits>

#include "common/cuda_hip/base/blas_bindings.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {


#include "common/cuda_hip/components/prefix_sum.hpp.inc"


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_PREFIX_SUM_CUH_
