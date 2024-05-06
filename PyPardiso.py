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


"""
    Python interface to PARDISO

    Supported data formats: CSR matrix (float (real) and complex)

    Author: Ke Wang
    E-mail: windypasss@gmail.com
    Date  : 2022.5.9

    Update: modified for nrhs and matrix transpose (T/ H) by Liu Lian (2023.04.06).
"""

# Import modules
import numpy as np
import ctypes
from ctypes.util import find_library

class PyPardiso:
    def __init__(self, A=None, b=None, matrix_type=6, phase=12):
        self.A = A
        self.mkl_dll = None
        mkl_path = find_library('mkl_rt.1')
        if mkl_path is None:
            mkl_path = find_library('mkl_rt')
        if mkl_path is None:
            raise ImportError('Mkl DLL not found')
        else:
            self.mkl_dll = ctypes.cdll.LoadLibrary(mkl_path)
        
        self.mkl_dll.pardisoinit.restype = None
        self.mkl_dll.pardiso.restype = None
        self.mkl_dll.pardisoinit.argtypes = [ctypes.POINTER(ctypes.c_int64),        # pt
                                             ctypes.POINTER(ctypes.c_int32),        # mtype
                                             ctypes.POINTER(ctypes.c_int32)]        # iparm
        self.mkl_dll.pardiso.argtypes     = [ctypes.POINTER(ctypes.c_int64),        # pt
                                             ctypes.POINTER(ctypes.c_int32),        # maxfct
                                             ctypes.POINTER(ctypes.c_int32),        # mnum
                                             ctypes.POINTER(ctypes.c_int32),        # mtype
                                             ctypes.POINTER(ctypes.c_int32),        # phase
                                             ctypes.POINTER(ctypes.c_int32),        # n
                                             ctypes.POINTER(None),                  # a
                                             ctypes.POINTER(ctypes.c_int32),        # ia
                                             ctypes.POINTER(ctypes.c_int32),        # ja
                                             ctypes.POINTER(ctypes.c_int32),        # perm
                                             ctypes.POINTER(ctypes.c_int32),        # nrhs
                                             ctypes.POINTER(ctypes.c_int32),        # iparm
                                             ctypes.POINTER(ctypes.c_int32),        # msglvl
                                             ctypes.POINTER(None),                  # b
                                             ctypes.POINTER(None),                  # x
                                             ctypes.POINTER(ctypes.c_int32)]        # error

        # PARDISO Parameters
        # see details at https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-fortran/top.html
        self.pt = np.zeros(64, dtype=np.int64) # Solver internal data address pointer

        self.maxfct = 1                              # Maximal number of factors in memory
        self.mnum   = 1                              # The number of matrix (from 1 to maxfct) to solve
        self.mtype  = matrix_type                    # Matrix type: 1  = Real and structurally symmetric
                                                     #              2  = Real and symmetric positive definite
                                                     #             -2  = Real and symmetric indefinite
                                                     #              3  = Complex and structurally symmetric
                                                     #              4  = Complex and Hermitian positive definite
                                                     #             -4  = Complex and Hermitian indefinite
                                                     #              6  = Complex and symmetric matrix
                                                     #              11 = Real and nonsymmetric matrix
                                                     #              13 = Complex and nonsymmetric matrix
        self.phase  = phase                          # Controls the execution of the solver: 11 = Analysis
                                                     #                                       12 = Analysis, numerical factorization
                                                     #                                       13 = Analysis, numerical factorization, solve
                                                     #                                       33 = Solve, iterative refinement
        self.n      = self.A.shape[0]                # Number of equations in the sparse linear system Ax = b
        self.a      = A.data                         # Contains the non-zero elements of the coefficient matrix A
        self.ia     = A.indptr + 1                   # rowIndex array in CSR3 format
        self.ja     = A.indices + 1                  # columns array in CSR3 format
        self.perm   = np.zeros(0, dtype=np.int32)    # Holds the permutation vector of size n,
                                                     # specifies elements used for computing a partial solution,
                                                     # or specifies differing values of the input matrices for low rank update

        # iparm
        # see https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-fortran/top/sparse-solver-routines/onemkl-pardiso-parallel-direct-sparse-solver-iface/pardiso-iparm-parameter.html
        self.iparm    = np.zeros(64, dtype=np.int32) # iparm[0]    = 1   # No solver default
                                                     # iparm[1]    = 2   # Fill-in reducing ordering from Metis
                                                     # iparm[3]    = 0   # No iterative-direct algorithm
                                                     # iparm[4]    = 0   # No user fill-in reducing permutation
                                                     # iparm[5]    = 0   # Write solution into x
                                                     # iparm[6]    = 0   # Not in use
                                                     # iparm[7]    = 2   # Max numbers of iterative refinement steps
                                                     # iparm[9-1]  = 13  # Perturb the pivot elements with 1E-13
                                                     # iparm[10-1] = 1   # Use nonsymmetric permutation and scaling MPS
                                                     # iparm[11]   = 0   # Conjugate transposed/transpose solve
                                                     # iparm[12]   = 0   # Maximum weighted matching algorithm is switched-off
                                                     # iparm[13]   = 0   # Output: Number of perturbed pivots
                                                     # iparm[17]   =-1   # Output: Number of nonzeros in the factor LU
                                                     # iparm[18]   =-1   # Output: Mflops for LU factorization
                                                     # iparm[19]   = 0   # Output: Number of CG Iterations
                                                     # iparm[34]   = 1   # Zero-based indexing
                                                     
        self.msglvl   = 0                            # Message level information
        self.error    = ctypes.c_int32(0)            # Error indicator

        nullptr       = ctypes.c_void_p()
        pardiso_error = ctypes.c_int32(0)
        c_int32_p     = ctypes.POINTER(ctypes.c_int32)
        c_int64_p     = ctypes.POINTER(ctypes.c_int64)
        c_float64_p   = ctypes.POINTER(ctypes.c_double)
        self.mkl_dll.pardisoinit(self.pt.ctypes.data_as(c_int64_p),
                                 ctypes.byref(ctypes.c_int32(self.mtype)),
                                 self.iparm.ctypes.data_as(c_int32_p))
        
        b = np.zeros((A.shape[0], 1))

        self.mkl_dll.pardiso(self.pt.ctypes.data_as(c_int64_p),
                             ctypes.byref(ctypes.c_int32(self.maxfct)),
                             ctypes.byref(ctypes.c_int32(self.mnum)),
                             ctypes.byref(ctypes.c_int32(self.mtype)),
                             ctypes.byref(ctypes.c_int32(self.phase)),
                             ctypes.byref(ctypes.c_int32(self.n)),
                             self.A.data.ctypes.data_as(c_float64_p),
                             self.ia.ctypes.data_as(c_int32_p),
                             self.ja.ctypes.data_as(c_int32_p),
                             self.perm.ctypes.data_as(c_int32_p),
                             ctypes.byref(ctypes.c_int32(1)),
                             self.iparm.ctypes.data_as(c_int32_p),
                             ctypes.byref(ctypes.c_int32(self.msglvl)),
                             b.ctypes.data_as(c_float64_p),
                             nullptr,
                             ctypes.byref(self.error))


    def solve(self, b, trans=0):
        # pardiso expects fortran (column-major) order for b
        if not b.flags.f_contiguous:
            b = np.asfortranarray(b)
        
        x = np.zeros_like(b)
        
        phase          = 33
        c_int32_p      = ctypes.POINTER(ctypes.c_int32)
        c_int64_p      = ctypes.POINTER(ctypes.c_int64)
        c_float64_p    = ctypes.POINTER(ctypes.c_double)
        self.iparm[11] = trans # 0-regular; 1-conjugate and transpose; 2-transpose
        
        self.mkl_dll.pardiso(self.pt.ctypes.data_as(c_int64_p),
                             ctypes.byref(ctypes.c_int32(self.maxfct)),
                             ctypes.byref(ctypes.c_int32(self.mnum)),
                             ctypes.byref(ctypes.c_int32(self.mtype)),
                             ctypes.byref(ctypes.c_int32(phase)),
                             ctypes.byref(ctypes.c_int32(self.n)),
                             self.A.data.ctypes.data_as(c_float64_p),
                             self.ia.ctypes.data_as(c_int32_p),
                             self.ja.ctypes.data_as(c_int32_p),
                             self.perm.ctypes.data_as(c_int32_p),
                             ctypes.byref(ctypes.c_int32(1 if b.ndim == 1 else b.shape[1])),
                             self.iparm.ctypes.data_as(c_int32_p),
                             ctypes.byref(ctypes.c_int32(self.msglvl)),
                             b.ctypes.data_as(c_float64_p),
                             x.ctypes.data_as(c_float64_p),
                             ctypes.byref((self.error)))
        return x
    
    def set_phase(self, phase):
        self.phase = phase

    def release(self):
        phase         = -1
        nullptr       = ctypes.c_void_p()
        pardiso_error = ctypes.c_int32(0)
        c_int32_p     = ctypes.POINTER(ctypes.c_int32)
        c_int64_p     = ctypes.POINTER(ctypes.c_int64)
        c_float64_p   = ctypes.POINTER(ctypes.c_double)
        
        self.mkl_dll.pardiso(self.pt.ctypes.data_as(c_int64_p),
                             ctypes.byref(ctypes.c_int32(self.maxfct)),
                             ctypes.byref(ctypes.c_int32(self.mnum)),
                             ctypes.byref(ctypes.c_int32(self.mtype)),
                             ctypes.byref(ctypes.c_int32(phase)),
                             ctypes.byref(ctypes.c_int32(self.n)),
                             self.A.data.ctypes.data_as(c_float64_p),
                             self.ia.ctypes.data_as(c_int32_p),
                             self.ja.ctypes.data_as(c_int32_p),
                             self.perm.ctypes.data_as(c_int32_p),
                             ctypes.byref(ctypes.c_int32(1)),
                             self.iparm.ctypes.data_as(c_int32_p),
                             ctypes.byref(ctypes.c_int32(self.msglvl)),
                             nullptr,
                             nullptr,
                             ctypes.byref((self.error)))