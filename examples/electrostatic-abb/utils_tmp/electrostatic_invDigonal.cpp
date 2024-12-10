// SPDX-FileCopyrightText: 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

auto diagVec = gko::matrix::Diagonal<ValueType>::create(exec, dim);

A->extract_diagonal(diagVec);

auto dense_diag =
    gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{dim, dim});
dense_diag->fill(0);

auto dense_diagVec =
    gko::matrix::Dense<ValueType>::create(exec, gko::dim<2>{1, dim});


// convert Diag to dense, and setup an identity matrix
for (IndexType i = 0; i < dim; ++i) {
    dense_diagVec->get_values()[i] = diagVec->get_const_values()[i];

    dense_diag->get_values()[i * dim + i] = 1;
}

dense_diag->inv_scale(dense_diagVec);
A->inv_scale(dense_diagVec);


dense_diag->apply(b, b_diag_scale);

//...
//... Solve system
//...
// scaling back the ID matrix with the diagonals
dense_diag->scale(dense_diagVec);
dense_diag->scale(dense_diagVec);

// retrieving back true x
dense_diag->apply(x, x_final);
