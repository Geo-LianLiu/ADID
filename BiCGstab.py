# ======================================================================== #
# Copyright 2024 Lian Liu. All Rights Reserved.                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #
#                       @Author  :   Lian Liu                              #
#                       @e-mail  :   lianliu1017@126.com                   #
# ======================================================================== #

# Import modules
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from scipy.sparse.linalg import bicgstab, spilu, LinearOperator
from scipy.sparse import csr_matrix


@tf.custom_gradient
def BiCGstab(A, b):
    a_coo = tf.raw_ops.CSRSparseMatrixToSparseTensor(sparse_matrix=A, type=b.dtype)
    indices, values, dense_shape = a_coo.indices, a_coo.values, a_coo.dense_shape

    row = tf.cast(indices[:, 0], dtype=tf.int32)
    col = tf.cast(indices[:, 1], dtype=tf.int32)
    csr_A = csr_matrix((values, (row, col)), shape=dense_shape)
    csc_A = csr_A.tocsc()
    
    b = b.numpy()

    ilu = spilu(csc_A, drop_tol= 1e-7)
    M = LinearOperator(ilu.shape, ilu.solve)
    
    x_best, info = bicgstab(csc_A, b, tol=1e-06, maxiter=b.shape[0], M=M)
    
    # print(np.allclose(csc_A.dot(x_best), b))

    x_best = tf.reshape(x_best, [-1, 1])
    
    def Bicggrad(grad):
        def _pruned_dense_matrix_multiplication(a, b, indices):
            rows = indices[:, 0]
            cols = indices[:, 1]
            a_rows = tf.gather(a, indices=rows)
            b_cols = tf.gather(b, indices=cols)

            return tf.reduce_sum(a_rows * b_cols, axis=-1)
        
        grad = grad.numpy()

        csc_A = csr_A.tocsc()
        csc_A = csc_A.getH()

        grad_b, _ = bicgstab(csc_A, grad, tol=1e-06, maxiter=grad.shape[0], M=M)
        # print(np.allclose(csc_A.dot(grad_b), grad))

        grad_b = tf.reshape(grad_b, [-1, 1])

        # compute {\partial f} / {\partial A}
        grad_a_values = - _pruned_dense_matrix_multiplication(
            grad_b, tf.math.conj(x_best), indices
        )
        grad_a = tf.raw_ops.SparseTensorToCSRSparseMatrix(
            indices=indices, values=grad_a_values, dense_shape=dense_shape
        )

        return (grad_a, grad_b)

    return x_best, Bicggrad