// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/exception.hpp"

#include <string>


#if HIP_VERSION >= 50200000
#include <hipblas/hipblas.h>
#include <hiprand/hiprand.h>
#include <hipsparse/hipsparse.h>
#else
#include <hipblas.h>
#include <hiprand.h>
#include <hipsparse.h>
#endif


#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/runtime.hpp"


namespace gko {


std::string HipError::get_error(int64 error_code)
{
    std::string name = hipGetErrorName(static_cast<hipError_t>(error_code));
    std::string message =
        hipGetErrorString(static_cast<hipError_t>(error_code));
    return name + ": " + message;
}


std::string HipblasError::get_error(int64 error_code)
{
#define GKO_REGISTER_HIPBLAS_ERROR(error_name)          \
    if (error_code == static_cast<int64>(error_name)) { \
        return #error_name;                             \
    }
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_SUCCESS);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_NOT_INITIALIZED);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_ALLOC_FAILED);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_INVALID_VALUE);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_ARCH_MISMATCH);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_MAPPING_ERROR);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_EXECUTION_FAILED);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_INTERNAL_ERROR);
    GKO_REGISTER_HIPBLAS_ERROR(HIPBLAS_STATUS_NOT_SUPPORTED);
    return "Unknown error";

#undef GKO_REGISTER_HIPBLAS_ERROR
}


std::string HiprandError::get_error(int64 error_code)
{
#define GKO_REGISTER_HIPRAND_ERROR(error_name)          \
    if (error_code == static_cast<int64>(error_name)) { \
        return #error_name;                             \
    }
    GKO_REGISTER_HIPRAND_ERROR(HIPRAND_STATUS_SUCCESS);
    GKO_REGISTER_HIPRAND_ERROR(HIPRAND_STATUS_VERSION_MISMATCH);
    GKO_REGISTER_HIPRAND_ERROR(HIPRAND_STATUS_NOT_INITIALIZED);
    GKO_REGISTER_HIPRAND_ERROR(HIPRAND_STATUS_ALLOCATION_FAILED);
    GKO_REGISTER_HIPRAND_ERROR(HIPRAND_STATUS_TYPE_ERROR);
    GKO_REGISTER_HIPRAND_ERROR(HIPRAND_STATUS_OUT_OF_RANGE);
    GKO_REGISTER_HIPRAND_ERROR(HIPRAND_STATUS_LENGTH_NOT_MULTIPLE);
    GKO_REGISTER_HIPRAND_ERROR(HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED);
    GKO_REGISTER_HIPRAND_ERROR(HIPRAND_STATUS_LAUNCH_FAILURE);
    GKO_REGISTER_HIPRAND_ERROR(HIPRAND_STATUS_PREEXISTING_FAILURE);
    GKO_REGISTER_HIPRAND_ERROR(HIPRAND_STATUS_INITIALIZATION_FAILED);
    GKO_REGISTER_HIPRAND_ERROR(HIPRAND_STATUS_ARCH_MISMATCH);
    GKO_REGISTER_HIPRAND_ERROR(HIPRAND_STATUS_INTERNAL_ERROR);
    return "Unknown error";

#undef GKO_REGISTER_HIPRAND_ERROR
}


std::string HipsparseError::get_error(int64 error_code)
{
#define GKO_REGISTER_HIPSPARSE_ERROR(error_name) \
    if (error_code == int64(error_name)) {       \
        return #error_name;                      \
    }
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_SUCCESS);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_NOT_INITIALIZED);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_ALLOC_FAILED);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_INVALID_VALUE);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_ARCH_MISMATCH);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_MAPPING_ERROR);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_EXECUTION_FAILED);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_INTERNAL_ERROR);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_ZERO_PIVOT);
#if HIP_VERSION >= 50200000
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_NOT_SUPPORTED);
    GKO_REGISTER_HIPSPARSE_ERROR(HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES);
#endif
    return "Unknown error";

#undef GKO_REGISTER_HIPSPARSE_ERROR
}


}  // namespace gko
