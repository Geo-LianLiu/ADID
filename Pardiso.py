# ======================================================================== #
# Copyright 2023 Lian Liu. All Rights Reserved.                            #
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

import tensorflow as tf
from scipy.sparse import csr_matrix, triu
from PyPardiso import PyPardiso
from time import time

# @tf.function
@tf.custom_gradient
def Pardiso(A, b, matrix_type=11, release=2):
    # tensorflow data type to scipy / numpy
    a_coo = tf.raw_ops.CSRSparseMatrixToSparseTensor(sparse_matrix=A, type=b.dtype)
    indices, values, dense_shape = a_coo.indices, a_coo.values, a_coo.dense_shape
    
    row = tf.cast(indices[:, 0], dtype=tf.int32)
    col = tf.cast(indices[:, 1], dtype=tf.int32)
    csr_A = csr_matrix((values, (row, col)), shape=dense_shape)

    # b = b.numpy()

    if matrix_type == 6:
        csr_A = triu(csr_A, format='csr')
    
    # start = time()
    # Analysis, numerical factorization
    InvA = PyPardiso(csr_A, matrix_type=matrix_type)
    x_best = InvA.solve(b.numpy())
    # print("Solving the linear equation costs ", time() - start, " seconds!")

    if release == 1:
        InvA.release()
    
    x_best = tf.constant(x_best, dtype=b.dtype)

    gc = [0]
    def PardisoGrad(grad):
        def _pruned_dense_matrix_multiplication(a, b, indices):
            rows = indices[:, 0]
            cols = indices[:, 1]
            a_rows = tf.gather(a, indices=rows)
            b_cols = tf.gather(b, indices=cols)

            return tf.reduce_sum(a_rows * b_cols, axis=-1)
        
        # grad = grad.eval(session=tf.compat.v1.Session()) # experimental_use_pfor = True
        grad = grad.numpy() # experimental_use_pfor = False
        
        grad_b = InvA.solve(grad, trans=1)
        
        # csr_H = csr_A.getH()
        # residual = np.linalg.norm(grad - csr_H.dot(grad_b), axis=0) / np.linalg.norm(grad)
        # print("Gradient: residual = ", residual)
        
        grad_b = tf.constant(grad_b, dtype=grad.dtype)

        # compute {\partial f} / {\partial A}
        grad_a_values = - _pruned_dense_matrix_multiplication(
            grad_b, tf.math.conj(x_best), indices
        )
        grad_a = tf.raw_ops.SparseTensorToCSRSparseMatrix(
            indices=indices, values=grad_a_values, dense_shape=dense_shape
        )

        gc[0] += 1
        if release == 2:
            InvA.release()
        elif  release == gc[0]:
            InvA.release()
        
        return (grad_a, grad_b)

    return x_best, PardisoGrad