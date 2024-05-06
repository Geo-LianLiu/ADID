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

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix


def _SQMR(A, b, M, iters=10000, tol=1e-6, x0=None):
    # The Symmetric Quasi Minimal Residual Method 
    # with Jacobian precondition but without Look-ahead.
    # Freund 和 Nachtigal - 1994 - An Implementation of 
    # the QMR Method Based on Coupled Two-Term Recurrences.
    # Freund and Nachtigal - 1994 - A new Krylov-subspace 
    # method for symmetric indefinite linear systems.
    b     = np.reshape(b, [A.shape[0], -1])
    bnorm = np.linalg.norm(b, axis=0)
    
    # initialization
    M     = np.reshape(M, [-1, 1])
    if x0 == None:
        x = b / M
    else:
        x = x0
    
    # r = b - Ax
    Ax    = A.dot(x)
    r     = b - Ax
    
    tau   = np.linalg.norm(r, axis=0)
    if (tau==0.).any():
        return x
    
    q     = r / M
    theta = np.zeros(b.shape[1])
    rho   = np.sum(r*q, axis=0)
    
    stop  = False
    k     = 0
    while stop is not True:
        k     = k + 1
        t     = A.dot(q)
        sigma = np.sum(q*t, axis=0)
        if (sigma==0.).any():
            print("SQMR failed to converge: sigma is zero")
            break
        
        alpha = rho / sigma
        r     = r - alpha*t

        theta_= theta
        rnorm = np.linalg.norm(r, axis=0)
        theta = rnorm / tau
        c     = 1. / np.sqrt(1.+theta**2.)
        tau   = tau * theta * c

        c2    = c**2.
        if k==1:
            temp = np.zeros(b.shape[1])
        else:
            temp = c2 * theta_ * theta_ * d
        d = temp + c2 * alpha * q

        # upadte
        x = x + d
        
        if (rho==0).any():
            print("SQMR failed to converge: rho is zero")
            break
        
        rho_ = rho
        u    = r / M
        rho  = np.sum(r*u, axis=0)
        beta = rho / rho_
        q    = u + beta*q

        if (rnorm/bnorm<=tol).all():
            residual = np.linalg.norm(b-A.dot(x), axis=0) / bnorm
            if (residual<=tol).all():
                print("iterations = ", k)
                print("residual   = ", residual)
                stop = True
        
        if k>=iters:
            residual = np.linalg.norm(b - A.dot(x), axis=0) / bnorm
            print("iterations = ", k)
            print("residual   = ", residual)
            stop = True
    return x


def jac(indices, values):
    diagonal = tf.boolean_mask(values, indices[:,0]==indices[:,1])

    return diagonal


@tf.custom_gradient
def SQMR(A, b, x0=None, iters=15000, tol=1e-6):
    dtype= b.dtype
    a_coo = tf.raw_ops.CSRSparseMatrixToSparseTensor(sparse_matrix=A, type=dtype)
    indices, values, dense_shape = a_coo.indices, a_coo.values, a_coo.dense_shape
    
    row   = tf.cast(indices[:, 0], dtype=tf.int32)
    col   = tf.cast(indices[:, 1], dtype=tf.int32)
    A     = csr_matrix((values, (row, col)), shape=dense_shape)

    M = jac(indices, values)
    
    x = _SQMR(A, b, M, iters=iters, tol=tol, x0=x0)
    x = tf.constant(x, dtype=dtype)

    def SQMRGrad(grad):
        def _pruned_dense_matrix_multiplication(a, b, indices):
            rows = indices[:, 0]
            cols = indices[:, 1]
            a_rows = tf.gather(a, indices=rows)
            b_cols = tf.gather(b, indices=cols)

            return tf.reduce_sum(a_rows * b_cols, axis=-1)
        
        A_H = A.getH()
        M_H = tf.math.conj(M)

        grad_b = _SQMR(A_H, grad, M_H, iters=iters, tol=tol)
        grad_b = tf.constant(grad_b, dtype=dtype)
        
        # compute {\partial f} / {\partial A}
        grad_a_values = - _pruned_dense_matrix_multiplication(
            grad_b, tf.math.conj(x), indices
        )
        grad_a = tf.raw_ops.SparseTensorToCSRSparseMatrix(
            indices=indices, values=grad_a_values, dense_shape=dense_shape
        )

        return (grad_a, grad_b)
    
    return x, SQMRGrad