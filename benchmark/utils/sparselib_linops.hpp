/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_BENCHMARK_UTILS_SPARSELIB_LINOPS_HPP_
#define GKO_BENCHMARK_UTILS_SPARSELIB_LINOPS_HPP_

#include <memory>


#include <ginkgo/core/base/lin_op.hpp>
#include "ginkgo/core/base/exception_helpers.hpp"


class cusp_csr;
class cusp_csrmp;
class cusp_csrmm;
class cusp_hybrid;
class cusp_coo;
class cusp_ell;
class cusp_gcsr;
class cusp_gcoo;
class cusp_csrex;
class cusp_gcsr;
class cusp_gcsr2;
class cusp_gcoo;


class hipsp_csr;
class hipsp_csrmm;
class hipsp_hybrid;
class hipsp_coo;
class hipsp_ell;


template <typename OpTagType>
std::unique_ptr<gko::LinOp> create_sparselib_linop(
    std::shared_ptr<const gko::Executor> exec);


#endif  // GKO_BENCHMARK_UTILS_SPARSELIB_LINOPS_HPP_