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

#include <ginkgo/core/preconditioner/isai.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/preconditioner/isai_kernels.hpp"
#include "hip/test/utils.hip.hpp"


namespace {


enum struct matrix_type { lower, upper, general, spd };


class Isai : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    Isai() : rand_engine(42) {}

    void SetUp()
    {
        ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        hip = gko::HipExecutor::create(0, ref);
    }

    std::unique_ptr<Csr> clone_allocations(const Csr* csr_mtx)
    {
        if (csr_mtx->get_executor() != ref) {
            return {nullptr};
        }
        const auto num_elems = csr_mtx->get_num_stored_elements();
        auto sparsity = csr_mtx->clone();

        // values are now filled with invalid data to catch potential errors
        auto begin_values = sparsity->get_values();
        auto end_values = begin_values + num_elems;
        std::fill(begin_values, end_values, -gko::one<value_type>());
        return sparsity;
    }

    void initialize_data(matrix_type type, gko::size_type n,
                         gko::size_type row_limit)
    {
        const bool for_lower_tm = type == matrix_type::lower;
        auto nz_dist = std::uniform_int_distribution<index_type>(1, row_limit);
        auto val_dist = std::uniform_real_distribution<value_type>(-1., 1.);
        mtx = Csr::create(ref);
        if (type == matrix_type::general) {
            auto dense_mtx = gko::test::generate_random_matrix<Dense>(
                n, n, nz_dist, val_dist, rand_engine, ref, gko::dim<2>{n, n});
            ensure_diagonal(dense_mtx.get());
            mtx->copy_from(dense_mtx.get());
        } else if (type == matrix_type::spd) {
            auto dense_mtx = gko::test::generate_random_band_matrix<Dense>(
                n, row_limit / 4, row_limit / 4, val_dist, rand_engine, ref,
                gko::dim<2>{n, n});
            auto transp = gko::as<Dense>(dense_mtx->transpose());
            auto spd_mtx = Dense::create(ref, gko::dim<2>{n, n});
            dense_mtx->apply(transp.get(), spd_mtx.get());
            mtx->copy_from(spd_mtx.get());
        } else {
            mtx = gko::test::generate_random_triangular_matrix<Csr>(
                n, true, for_lower_tm, nz_dist, val_dist, rand_engine, ref,
                gko::dim<2>{n, n});
        }
        inverse = clone_allocations(mtx.get());

        d_mtx = gko::clone(hip, mtx);
        d_inverse = gko::clone(hip, inverse);
    }

    void ensure_diagonal(Dense* mtx)
    {
        for (int i = 0; i < mtx->get_size()[0]; ++i) {
            mtx->at(i, i) = gko::one<Dense::value_type>();
        }
    }


    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::HipExecutor> hip;

    std::default_random_engine rand_engine;

    std::unique_ptr<Csr> mtx;
    std::unique_ptr<Csr> inverse;

    std::unique_ptr<Csr> d_mtx;
    std::unique_ptr<Csr> d_inverse;
};


TEST_F(Isai, HipIsaiGenerateLinverseShortIsEquivalentToRef)
{
    initialize_data(matrix_type::lower, 536, 31);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(hip, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::kernels::hip::isai::generate_tri_inverse(
        hip, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        true);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_EQ(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, HipIsaiGenerateUinverseShortIsEquivalentToRef)
{
    initialize_data(matrix_type::upper, 615, 31);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(hip, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::kernels::hip::isai::generate_tri_inverse(
        hip, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        false);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_EQ(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, HipIsaiGenerateAinverseShortIsEquivalentToRef)
{
    initialize_data(matrix_type::general, 615, 15);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(hip, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::kernels::hip::isai::generate_general_inverse(
        hip, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        false);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_EQ(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, HipIsaiGenerateSpdinverseShortIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 15);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(hip, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::kernels::hip::isai::generate_general_inverse(
        hip, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        true);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 15 * r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_EQ(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, HipIsaiGenerateLinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::lower, 554, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(hip, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::kernels::hip::isai::generate_tri_inverse(
        hip, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        true);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_GT(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, HipIsaiGenerateUinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::upper, 695, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(hip, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::kernels::hip::isai::generate_tri_inverse(
        hip, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        false);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_GT(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, HipIsaiGenerateAinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::general, 695, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(hip, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::kernels::hip::isai::generate_general_inverse(
        hip, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        false);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 10 * r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_GT(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, HipIsaiGenerateSpdinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::array<index_type> da1(hip, num_rows + 1);
    auto da2 = da1;

    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::kernels::hip::isai::generate_general_inverse(
        hip, d_mtx.get(), d_inverse.get(), da1.get_data(), da2.get_data(),
        true);

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse, d_inverse);
    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 10 * r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(a1, da1);
    GKO_ASSERT_ARRAY_EQ(a2, da2);
    ASSERT_GT(a1.get_const_data()[num_rows], 0);
}


TEST_F(Isai, HipIsaiGenerateExcessLinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::lower, 518, 40);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(hip, a1);
    gko::array<index_type> da2(hip, a2);
    auto e_dim = a1.get_data()[num_rows];
    auto e_nnz = a2.get_data()[num_rows];
    auto excess = Csr::create(ref, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    auto dexcess = Csr::create(hip, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto de_rhs = Dense::create(hip, gko::dim<2>(e_dim, 1));

    gko::kernels::reference::isai::generate_excess_system(
        ref, mtx.get(), inverse.get(), a1.get_const_data(), a2.get_const_data(),
        excess.get(), e_rhs.get(), 0, num_rows);
    gko::kernels::hip::isai::generate_excess_system(
        hip, d_mtx.get(), d_inverse.get(), da1.get_const_data(),
        da2.get_const_data(), dexcess.get(), de_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_EQ_SPARSITY(excess, dexcess);
    GKO_ASSERT_MTX_NEAR(excess, dexcess, 0);
    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, HipIsaiGenerateExcessUinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::upper, 673, 51);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::array<index_type> da1(hip, a1);
    gko::array<index_type> da2(hip, a2);
    auto e_dim = a1.get_data()[num_rows];
    auto e_nnz = a2.get_data()[num_rows];
    auto excess = Csr::create(ref, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    auto dexcess = Csr::create(hip, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto de_rhs = Dense::create(hip, gko::dim<2>(e_dim, 1));

    gko::kernels::reference::isai::generate_excess_system(
        ref, mtx.get(), inverse.get(), a1.get_const_data(), a2.get_const_data(),
        excess.get(), e_rhs.get(), 0, num_rows);
    gko::kernels::hip::isai::generate_excess_system(
        hip, d_mtx.get(), d_inverse.get(), da1.get_const_data(),
        da2.get_const_data(), dexcess.get(), de_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_EQ_SPARSITY(excess, dexcess);
    GKO_ASSERT_MTX_NEAR(excess, dexcess, 0);
    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, HipIsaiGenerateExcessAinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::general, 100, 51);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::array<index_type> da1(hip, a1);
    gko::array<index_type> da2(hip, a2);
    auto e_dim = a1.get_data()[num_rows];
    auto e_nnz = a2.get_data()[num_rows];
    auto excess = Csr::create(ref, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    auto dexcess = Csr::create(hip, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto de_rhs = Dense::create(hip, gko::dim<2>(e_dim, 1));

    gko::kernels::reference::isai::generate_excess_system(
        ref, mtx.get(), inverse.get(), a1.get_const_data(), a2.get_const_data(),
        excess.get(), e_rhs.get(), 0, num_rows);
    gko::kernels::hip::isai::generate_excess_system(
        hip, d_mtx.get(), d_inverse.get(), da1.get_const_data(),
        da2.get_const_data(), dexcess.get(), de_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_EQ_SPARSITY(excess, dexcess);
    GKO_ASSERT_MTX_NEAR(excess, dexcess, 0);
    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, HipIsaiGenerateExcessSpdinverseLongIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(hip, a1);
    gko::array<index_type> da2(hip, a2);
    auto e_dim = a1.get_data()[num_rows];
    auto e_nnz = a2.get_data()[num_rows];
    auto excess = Csr::create(ref, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    auto dexcess = Csr::create(hip, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto de_rhs = Dense::create(hip, gko::dim<2>(e_dim, 1));

    gko::kernels::reference::isai::generate_excess_system(
        ref, mtx.get(), inverse.get(), a1.get_const_data(), a2.get_const_data(),
        excess.get(), e_rhs.get(), 0, num_rows);
    gko::kernels::hip::isai::generate_excess_system(
        hip, d_mtx.get(), d_inverse.get(), da1.get_const_data(),
        da2.get_const_data(), dexcess.get(), de_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_EQ_SPARSITY(excess, dexcess);
    GKO_ASSERT_MTX_NEAR(excess, dexcess, 0);
    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, HipIsaiGeneratePartialExcessIsEquivalentToRef)
{
    initialize_data(matrix_type::general, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::array<index_type> da1(hip, a1);
    gko::array<index_type> da2(hip, a2);
    auto e_dim = a1.get_data()[10] - a1.get_data()[5];
    auto e_nnz = a2.get_data()[10] - a2.get_data()[5];
    auto excess = Csr::create(ref, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    auto dexcess = Csr::create(hip, gko::dim<2>(e_dim, e_dim), e_nnz);
    auto de_rhs = Dense::create(hip, gko::dim<2>(e_dim, 1));

    gko::kernels::reference::isai::generate_excess_system(
        ref, mtx.get(), inverse.get(), a1.get_const_data(), a2.get_const_data(),
        excess.get(), e_rhs.get(), 5u, 10u);
    gko::kernels::hip::isai::generate_excess_system(
        hip, d_mtx.get(), d_inverse.get(), da1.get_const_data(),
        da2.get_const_data(), dexcess.get(), de_rhs.get(), 5u, 10u);

    GKO_ASSERT_MTX_EQ_SPARSITY(excess, dexcess);
    GKO_ASSERT_MTX_NEAR(excess, dexcess, 0);
    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, HipIsaiScaleExcessSolutionIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(hip, a1);
    auto e_dim = a1.get_data()[num_rows];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(hip, e_rhs);
    d_inverse->copy_from(lend(inverse));

    gko::kernels::reference::isai::scale_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), 0, num_rows);
    gko::kernels::hip::isai::scale_excess_solution(hip, da1.get_const_data(),
                                                   de_rhs.get(), 0, num_rows);

    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
}


TEST_F(Isai, HipIsaiScalePartialExcessSolutionIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(hip, a1);
    auto e_dim = a1.get_data()[10] - a1.get_data()[5];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(hip, e_rhs);

    gko::kernels::reference::isai::scale_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), 5u, 10u);
    gko::kernels::hip::isai::scale_excess_solution(hip, da1.get_const_data(),
                                                   de_rhs.get(), 5u, 10u);

    GKO_ASSERT_MTX_NEAR(e_rhs, de_rhs, 0);
}


TEST_F(Isai, HipIsaiScatterExcessSolutionLIsEquivalentToRef)
{
    initialize_data(matrix_type::lower, 572, 52);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(hip, a1);
    auto e_dim = a1.get_data()[num_rows];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(hip, e_rhs);
    d_inverse->copy_from(lend(inverse));

    gko::kernels::reference::isai::scatter_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), inverse.get(), 0, num_rows);
    gko::kernels::hip::isai::scatter_excess_solution(
        hip, da1.get_const_data(), de_rhs.get(), d_inverse.get(), 0, num_rows);

    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, HipIsaiScatterExcessSolutionUIsEquivalentToRef)
{
    initialize_data(matrix_type::upper, 702, 45);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_tri_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::array<index_type> da1(hip, a1);
    auto e_dim = a1.get_data()[num_rows];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(hip, e_rhs);
    // overwrite -1 values with inverse
    d_inverse->copy_from(lend(inverse));

    gko::kernels::reference::isai::scatter_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), inverse.get(), 0, num_rows);
    gko::kernels::hip::isai::scatter_excess_solution(
        hip, da1.get_const_data(), de_rhs.get(), d_inverse.get(), 0, num_rows);

    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, HipIsaiScatterExcessSolutionAIsEquivalentToRef)
{
    initialize_data(matrix_type::general, 702, 45);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), false);
    gko::array<index_type> da1(hip, a1);
    auto e_dim = a1.get_data()[num_rows];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(hip, e_rhs);
    // overwrite -1 values with inverse
    d_inverse->copy_from(lend(inverse));

    gko::kernels::reference::isai::scatter_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), inverse.get(), 0, num_rows);
    gko::kernels::hip::isai::scatter_excess_solution(
        hip, da1.get_const_data(), de_rhs.get(), d_inverse.get(), 0, num_rows);

    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, HipIsaiScatterExcessSolutionSpdIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(hip, a1);
    auto e_dim = a1.get_data()[num_rows];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(hip, e_rhs);
    // overwrite -1 values with inverse
    d_inverse->copy_from(lend(inverse));

    gko::kernels::reference::isai::scatter_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), inverse.get(), 0, num_rows);
    gko::kernels::hip::isai::scatter_excess_solution(
        hip, da1.get_const_data(), de_rhs.get(), d_inverse.get(), 0, num_rows);

    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 0);
    ASSERT_GT(e_dim, 0);
}


TEST_F(Isai, HipIsaiScatterPartialExcessSolutionIsEquivalentToRef)
{
    initialize_data(matrix_type::spd, 100, 64);
    const auto num_rows = mtx->get_size()[0];
    gko::array<index_type> a1(ref, num_rows + 1);
    auto a2 = a1;
    gko::kernels::reference::isai::generate_general_inverse(
        ref, mtx.get(), inverse.get(), a1.get_data(), a2.get_data(), true);
    gko::array<index_type> da1(hip, a1);
    auto e_dim = a1.get_data()[10] - a1.get_data()[5];
    auto e_rhs = Dense::create(ref, gko::dim<2>(e_dim, 1));
    std::fill_n(e_rhs->get_values(), e_dim, 123456);
    auto de_rhs = gko::clone(hip, e_rhs);
    // overwrite -1 values with inverse
    d_inverse->copy_from(lend(inverse));

    gko::kernels::reference::isai::scatter_excess_solution(
        ref, a1.get_const_data(), e_rhs.get(), inverse.get(), 5u, 10u);
    gko::kernels::hip::isai::scatter_excess_solution(
        hip, da1.get_const_data(), de_rhs.get(), d_inverse.get(), 5u, 10u);

    GKO_ASSERT_MTX_NEAR(inverse, d_inverse, 0);
    ASSERT_GT(e_dim, 0);
}


}  // namespace
