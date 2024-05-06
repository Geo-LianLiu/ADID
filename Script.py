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

import tensorflow as tf
# from Pardiso import Pardiso as solver
# from SQMR import SQMR as solver
# from SuperLU import SuperLU as solver
from BiCGstab import BiCGstab as solver


def forward(m):
    a1 = tf.concat([m[:1], m[1:]], axis=0)
    a2 = tf.concat([m[1:], m[:1]], axis=0)
    cosm = tf.cos(a1) + a2
    K = tf.eye(2, dtype=tf.float64) * cosm
    print("K:", K.numpy())

    CSR_K = tf.raw_ops.DenseToCSRSparseMatrix(dense_input=K, indices=tf.where(K))

    s = tf.sin(a1) + a2
    print("s", s.numpy())

    u = solver(CSR_K, s)
    print("u", u.numpy())
    
    y = tf.reduce_sum(u + m)
    print("y", y.numpy())

    return y


if __name__=="__main__":
    m = tf.zeros([2, 1], dtype=tf.float64)
    with tf.GradientTape() as g:
        g.watch(m)
        y = forward(m)
        
    grad = g.gradient(y, m)

    print("grad", grad)