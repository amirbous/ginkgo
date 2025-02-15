// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_COMPONENTS_REDUCTION_HPP_
#define GKO_COMMON_CUDA_HIP_COMPONENTS_REDUCTION_HPP_


#include <type_traits>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/uninitialized_array.hpp"
#include "core/base/array_access.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


constexpr int default_reduce_block_size = 512;


/**
 * @internal
 *
 * Computes a reduction using the binary operation `reduce_op` on a group
 * `group`. Each thread contributes with one element `local_data`. The local
 * thread element is always passed as the first parameter to the `reduce_op`.
 * The function returns the result of the reduction on all threads.
 *
 * @note The function is guaranteed to return the correct value on all threads
 *       only if `reduce_op` is commutative (in addition to being associative).
 *       Otherwise, the correct value is returned only to the thread with
 *       subwarp index 0.
 */
template <
    typename Group, typename ValueType, typename Operator,
    typename = std::enable_if_t<group::is_communicator_group<Group>::value>>
__device__ __forceinline__ ValueType reduce(const Group& group,
                                            ValueType local_data,
                                            Operator reduce_op = Operator{})
{
#pragma unroll
    for (int32 bitmask = 1; bitmask < group.size(); bitmask <<= 1) {
        const auto remote_data = group.shfl_xor(local_data, bitmask);
        local_data = reduce_op(local_data, remote_data);
    }
    return local_data;
}


/**
 * @internal
 *
 * Returns the index of the thread that has the element with the largest
 * magnitude among all the threads in the group.
 * Only the values from threads which set `is_pivoted` to `false` will be
 * considered.
 */
template <
    typename Group, typename ValueType,
    typename = std::enable_if_t<group::is_communicator_group<Group>::value>>
__device__ __forceinline__ int choose_pivot(const Group& group,
                                            ValueType local_data,
                                            bool is_pivoted)
{
    using real = remove_complex<ValueType>;
    real lmag = is_pivoted ? -one<real>() : abs(local_data);
    const auto pivot =
        reduce(group, group.thread_rank(), [&](int lidx, int ridx) {
            const auto rmag = group.shfl(lmag, ridx);
            if (rmag > lmag) {
                lmag = rmag;
                lidx = ridx;
            }
            return lidx;
        });
    // pivot operator not commutative, make sure everyone has the same pivot
    return group.shfl(pivot, 0);
}


/**
 * @internal
 *
 * Computes a reduction using the binary operation `reduce_op` on entire block.
 * The data for the reduction is taken from the `data` array which has to be of
 * size `block_size` and accessible from all threads. The `data` array is also
 * used as work space (so its content will be destroyed in the process), as well
 * as to store the return value - which is stored in the 0-th position of the
 * array.
 */
template <
    typename Group, typename ValueType, typename Operator,
    typename = std::enable_if_t<group::is_synchronizable_group<Group>::value>>
__device__ void reduce(const Group& __restrict__ group,
                       ValueType* __restrict__ data,
                       Operator reduce_op = Operator{})
{
    const auto local_id = group.thread_rank();

    for (int k = group.size() / 2; k >= config::warp_size; k /= 2) {
        group.sync();
        if (local_id < k) {
            data[local_id] = reduce_op(data[local_id], data[local_id + k]);
        }
    }

    const auto warp = group::tiled_partition<config::warp_size>(group);
    const auto warp_id = group.thread_rank() / warp.size();
    if (warp_id > 0) {
        return;
    }
    auto result = reduce(warp, data[warp.thread_rank()], reduce_op);
    if (warp.thread_rank() == 0) {
        data[0] = result;
    }
}


/**
 * @internal
 *
 * Computes `num` reductions using the binary operation `reduce_op` on an
 * entire block.
 * The data range for the ith (i < num) reduction is:
 * [data + i * stride, data + block_size) (block_size == group.size())
 * The `data` array for each reduction must be of size `block_size` and
 * accessible from all threads. The `data` array is also
 * used as work space (so its content will be destroyed in the process), as well
 * as to store the return value - which is stored in the (i * stride)-th
 * position of the array.
 */
template <
    typename Group, typename ValueType, typename Operator,
    typename = std::enable_if_t<group::is_synchronizable_group<Group>::value>>
__device__ void multireduce(const Group& __restrict__ group,
                            ValueType* __restrict__ data, size_type stride,
                            size_type num, Operator reduce_op = Operator{})
{
    const auto local_id = group.thread_rank();

    for (int k = group.size() / 2; k >= config::warp_size; k /= 2) {
        group.sync();
        if (local_id < k) {
            for (int j = 0; j < num; j++) {
                data[j * stride + local_id] =
                    reduce_op(data[j * stride + local_id],
                              data[j * stride + local_id + k]);
            }
        }
    }

    const auto warp = group::tiled_partition<config::warp_size>(group);
    const auto warp_id = group.thread_rank() / warp.size();
    if (warp_id > 0) {
        return;
    }
    for (int j = 0; j < num; j++) {
        auto result =
            reduce(warp, data[j * stride + warp.thread_rank()], reduce_op);
        if (warp.thread_rank() == 0) {
            data[j * stride] = result;
        }
    }
}


/**
 * @internal
 *
 * Computes a reduction using the binary operation `reduce_op` on an array
 * `source` of any size. Has to be called a second time on `result` to reduce
 * an array larger than `block_size`.
 */
template <typename Operator, typename ValueType>
__device__ void reduce_array(size_type size,
                             const ValueType* __restrict__ source,
                             ValueType* __restrict__ result,
                             Operator reduce_op = Operator{})
{
    const auto tidx = thread::get_thread_id_flat();
    auto thread_result = zero<ValueType>();
    for (auto i = tidx; i < size; i += blockDim.x * gridDim.x) {
        thread_result = reduce_op(thread_result, source[i]);
    }
    result[threadIdx.x] = thread_result;

    group::this_thread_block().sync();

    // Stores the result of the reduction inside `result[0]`
    reduce(group::this_thread_block(), result, reduce_op);
}


/**
 * @internal
 *
 * Computes a reduction using the add operation (+) on an array
 * `source` of any size. Has to be called a second time on `result` to reduce
 * an array larger than `default_reduce_block_size`.
 */
template <typename ValueType>
__global__ __launch_bounds__(default_reduce_block_size) void reduce_add_array(
    size_type size, const ValueType* __restrict__ source,
    ValueType* __restrict__ result)
{
    __shared__ uninitialized_array<ValueType, default_reduce_block_size>
        block_sum;
    reduce_array(size, source, static_cast<ValueType*>(block_sum),
                 [](const ValueType& x, const ValueType& y) { return x + y; });

    if (threadIdx.x == 0) {
        result[blockIdx.x] = block_sum[0];
    }
}


/**
 * @internal
 *
 * Computes a reduction using the add operation (+) on an array
 * `source` of any size. Has to be called a second time on `result` to reduce
 * an array larger than `default_block_size`.
 *
 * @note uses existing value in result
 */
template <typename ValueType>
__global__
__launch_bounds__(default_reduce_block_size) void reduce_add_array_with_initial_value(
    size_type size, const ValueType* __restrict__ source,
    ValueType* __restrict__ result)
{
    __shared__ uninitialized_array<ValueType, default_reduce_block_size>
        block_sum;
    reduce_array(size, source, static_cast<ValueType*>(block_sum),
                 [](const ValueType& x, const ValueType& y) { return x + y; });

    if (threadIdx.x == 0) {
        result[blockIdx.x] += block_sum[0];
    }
}


/**
 * Compute a reduction using add operation (+).
 *
 * @param exec  Executor associated to the array
 * @param size  size of the array
 * @param source  the pointer of the array
 *
 * @return the reduction result
 */
template <typename ValueType>
ValueType reduce_add_array(std::shared_ptr<const DefaultExecutor> exec,
                           size_type size, const ValueType* source)
{
    auto block_results_val = source;
    size_type grid_dim = size;
    auto block_results = array<ValueType>(exec);
    if (size > default_reduce_block_size) {
        const auto n = ceildiv(size, default_reduce_block_size);
        grid_dim =
            (n <= default_reduce_block_size) ? n : default_reduce_block_size;

        block_results.resize_and_reset(grid_dim);

        reduce_add_array<<<grid_dim, default_reduce_block_size, 0,
                           exec->get_stream()>>>(
            size, as_device_type(source),
            as_device_type(block_results.get_data()));

        block_results_val = block_results.get_const_data();
    }

    auto d_result = array<ValueType>(exec, 1);

    reduce_add_array<<<1, default_reduce_block_size, 0, exec->get_stream()>>>(
        grid_dim, as_device_type(block_results_val),
        as_device_type(d_result.get_data()));
    auto answer = get_element(d_result, 0);
    return answer;
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_CUDA_HIP_COMPONENTS_REDUCTION_HPP_
