/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/matrix/hybrid_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>


#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Hybrid matrix format namespace.
 * @ref Hybrid
 * @ingroup hybrid
 */
namespace hybrid {


void compute_coo_row_ptrs(std::shared_ptr<const DefaultExecutor> exec,
                          const Array<size_type>& row_nnz, size_type ell_lim,
                          int64* coo_row_ptrs)
{
    for (size_type row = 0; row < row_nnz.get_num_elems(); row++) {
        if (row_nnz.get_const_data()[row] <= ell_lim) {
            coo_row_ptrs[row] = 0;
        } else {
            coo_row_ptrs[row] = row_nnz.get_const_data()[row] - ell_lim;
        }
    }
    components::prefix_sum(exec, coo_row_ptrs, row_nnz.get_num_elems() + 1);
}


void compute_row_nnz(std::shared_ptr<const DefaultExecutor> exec,
                     const Array<int64>& row_ptrs, size_type* row_nnzs)
{
    for (size_type i = 0; i < row_ptrs.get_num_elems() - 1; i++) {
        row_nnzs[i] =
            row_ptrs.get_const_data()[i + 1] - row_ptrs.get_const_data()[i];
    }
}


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,
                         const device_matrix_data<ValueType, IndexType>& data,
                         const int64* row_ptrs, const int64*,
                         matrix::Hybrid<ValueType, IndexType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto ell_max_nnz = result->get_ell_num_stored_elements_per_row();
    const auto nonzeros = data.nonzeros.get_const_data();
    size_type coo_nz{};
    for (size_type row = 0; row < num_rows; row++) {
        size_type ell_nz{};
        for (auto nz = row_ptrs[row]; nz < row_ptrs[row + 1]; nz++) {
            if (ell_nz < ell_max_nnz) {
                result->ell_col_at(row, ell_nz) = nonzeros[nz].column;
                result->ell_val_at(row, ell_nz) = nonzeros[nz].value;
                ell_nz++;
            } else {
                result->get_coo_row_idxs()[coo_nz] = nonzeros[nz].row;
                result->get_coo_col_idxs()[coo_nz] = nonzeros[nz].column;
                result->get_coo_values()[coo_nz] = nonzeros[nz].value;
                coo_nz++;
            }
        }
        for (; ell_nz < ell_max_nnz; ell_nz++) {
            result->ell_col_at(row, ell_nz) = 0;
            result->ell_val_at(row, ell_nz) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_FILL_IN_MATRIX_DATA_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Hybrid<ValueType, IndexType>* source,
                    const IndexType*, const IndexType*,
                    matrix::Csr<ValueType, IndexType>* result)
{
    auto csr_val = result->get_values();
    auto csr_col_idxs = result->get_col_idxs();
    auto csr_row_ptrs = result->get_row_ptrs();
    const auto ell = source->get_ell();
    const auto max_nnz_per_row = ell->get_num_stored_elements_per_row();
    const auto coo_val = source->get_const_coo_values();
    const auto coo_col = source->get_const_coo_col_idxs();
    const auto coo_row = source->get_const_coo_row_idxs();
    const auto coo_nnz = source->get_coo_num_stored_elements();
    csr_row_ptrs[0] = 0;
    size_type csr_idx = 0;
    size_type coo_idx = 0;
    for (IndexType row = 0; row < source->get_size()[0]; row++) {
        // Ell part
        for (IndexType col = 0; col < max_nnz_per_row; col++) {
            const auto val = ell->val_at(row, col);
            if (is_nonzero(val)) {
                csr_val[csr_idx] = val;
                csr_col_idxs[csr_idx] = ell->col_at(row, col);
                csr_idx++;
            }
        }
        // Coo part (row should be ascending)
        while (coo_idx < coo_nnz && coo_row[coo_idx] == row) {
            csr_val[csr_idx] = coo_val[coo_idx];
            csr_col_idxs[csr_idx] = coo_col[coo_idx];
            csr_idx++;
            coo_idx++;
        }
        csr_row_ptrs[row + 1] = csr_idx;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_CONVERT_TO_CSR_KERNEL);


}  // namespace hybrid
}  // namespace reference
}  // namespace kernels
}  // namespace gko
