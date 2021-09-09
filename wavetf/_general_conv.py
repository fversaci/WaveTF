# Copyright 2020 CRS4 (http://www.crs4.it/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import tensorflow as tf
from tensorflow import keras
from wavetf._base_wavelets import DirWaveLayer1D, InvWaveLayer1D, DirWaveLayer2D, InvWaveLayer2D

########################################################################
# 1D wavelet
########################################################################

########################################################################
# Direct wavelet transform
## input: (b, x, c) --> output: (b, nx, 2*c)

class KerWaveLayer1D(DirWaveLayer1D):
    """1D general (kernel based) direct trasform"""
    ########################################################################
    ## Init (with provided wavelet kernel)
    ########################################################################
    def __init__(self, ker, **kwargs):
        ## Wavelet kernel
        self.wavelet_ker = ker
        self.n = ker.shape[0] # e.g., DB2 ker.shape = [4,2]
        ## call constructor
        super(DirWaveLayer1D, self).__init__(**kwargs)
    def pad_tensor(self, t1) :
        # anti-symmetric-reflect padding, a.k.a. asym in matlab
        pad_l = self.n // 2
        if (t1.shape[-1]<pad_l):
            raise ValueError(f'Length of arrays should be at least {pad_l}')
        lb = t1.shape[-1] % 2 # if odd, add padding at the end
        first = 2.0 * t1[..., 0:1]
        last = 2.0 * t1[..., -1:]
        y = [first - t1[..., i+1:i+2] for i in range(pad_l-1)]
        y.reverse()
        z = [last - t1[..., -2-i:-1-i] for i in range(pad_l-1+lb)]
        s1 = tf.concat(y + [t1] + z, axis=-1)
        return (s1)
    def kernel_function(self, input):
        # choose float precision accordingly to input
        wavelet_ker = tf.cast(self.wavelet_ker, tf.float32) if (
            input.dtype==tf.float32) else self.wavelet_ker
        # input: (b, x, c)
        t1 = tf.transpose(input, perm=[0, 2, 1]) # out: (b, c, x)
        ## add padding
        s1 = self.pad_tensor(t1)
        ## s1: (b, c, 2*nx)
        nx_dim = s1.shape[2]
        s1 = tf.reshape(s1, [self.bs, self.cn, nx_dim, 1]) # out: (b, c, 2*nx', 1)
        # build kernels and apply to rows
        k1l = tf.reshape(wavelet_ker[:,0], (self.n, 1, 1))
        k1h = tf.reshape(wavelet_ker[:,1], (self.n, 1, 1))
        rl = tf.nn.conv1d(s1, k1l, stride=2, padding='VALID')
        rh = tf.nn.conv1d(s1, k1h, stride=2, padding='VALID')
        r = tf.concat((rl, rh), axis=-1) # out: (b, c, nx, 2)
        # r = tf.reshape(r, [self.bs, self.cn, self.nx, 2]) # out: (b, c, nx, 2)
        r = tf.transpose(r, [0, 2, 3, 1]) # out: (b, nx, 2, c)
        r = tf.reshape(r, [self.bs, self.nx, 2*self.cn]) # out: (b, nx, 2*c)
        return r

class InvKerWaveLayer1D(InvWaveLayer1D):

    """1D general (kernel based) inverse trasform"""
    ########################################################################
    ## Init (with provided wavelet kernel)
    ########################################################################
    def __init__(self, ker, **kwargs):
        ## Direct kernel
        n = ker.shape[0] # e.g., DB2 ker.shape = [4,2]
        self.n = n
        self.dir_ker = ker
        ## inverse kernel
        ker = tf.reshape(ker, [n // 2, 2, 2])
        ker = ker[::-1, ...]
        ker = tf.transpose(ker, perm=[0,2,1])
        ker = tf.reshape(ker, [n, 2])
        self.inv_ker = ker
        ## compute and border transformations
        self.comp_inv_bor_0()
        self.comp_inv_bor_1()
        ## call constructor
        super(InvWaveLayer1D, self).__init__(**kwargs)
    def comp_inv_bor_0(self):
        ## matrix for border effect 0 (begin)
        # ker_0.shape = [n, n-1]
        tr_ker = tf.transpose(self.dir_ker)
        n = self.n
        pker = tf.pad(tr_ker, [[0,0],[0,n]])
        UL = tf.concat([tf.roll(pker, 2*i, axis=1) for i in
                        range(n//2)], axis=0)
        ker_0 = UL[:n,:n-1]
        # pad_0.shape = [n-1, n//2]
        low = tf.eye(n//2, dtype=tf.float64)
        up = - tf.eye(n//2-1, dtype=tf.float64)[::-1]
        up = tf.pad(up, [[0,0],[1,0]], constant_values=2)
        pad_0 = tf.concat([up, low], axis=0)
        # initial border effect (shape = [n, n//2])
        self.inv_bor_0 = tf.transpose(tf.linalg.pinv(ker_0 @ pad_0), [1, 0])
    def comp_inv_bor_1(self):
        ## matrix for border effect 1 (end)
        # ker_1.shape = [n, n-1]
        tr_ker = tf.transpose(self.dir_ker)
        n = self.n
        pker = tf.pad(tr_ker, [[0,0],[n,0]])
        BR = tf.concat([tf.roll(pker, -2*(n//2-1-i), axis=1) for i in
                        range(n//2)], axis=0)
        ker_1 = BR[-n:,-n+1:]
        # pad_1.shape = [n-1, n//2]
        low = - tf.eye(n//2-1, dtype=tf.float64)[::-1]
        low = tf.pad(low, [[0,0],[0,1]], constant_values=2)
        up = tf.eye(n//2, dtype=tf.float64)
        pad_1 = tf.concat([up, low], axis=0)
        # final border effect (shape = [n, n//2])
        self.inv_bor_1 = tf.transpose(tf.linalg.pinv(ker_1 @ pad_1), [1, 0])
    def kernel_function(self, input):
        # choose float precision accordingly to input
        wavelet_ker = tf.cast(self.inv_ker, tf.float32) if (
            input.dtype==tf.float32) else self.inv_ker
        inv_bor_0 = tf.cast(self.inv_bor_0, tf.float32) if (
            input.dtype==tf.float32) else self.inv_bor_0
        inv_bor_1 = tf.cast(self.inv_bor_1, tf.float32) if (
            input.dtype==tf.float32) else self.inv_bor_1
        #######################################
        ## reshape
        #######################################
        t1 = tf.reshape(input, [self.bs, self.ox, self.cn])
        # out: (b, ox, c)
        t1 = tf.transpose(t1, perm=[0, 2, 1]) # out: (b, c, ox)
        #######################################
        ## compute borders
        #######################################
        # border 0
        b_0 = t1[..., :self.n]
        r2_0 = b_0 @ inv_bor_0
        # border 1
        b_1 = t1[..., -self.n:]
        r2_1 = b_1 @ inv_bor_1
        #######################################
        ## transform core
        #######################################
        t1 = tf.reshape(t1, [self.bs, self.cn, self.ox, 1]) # out: (b, c, ox, 1)
        # apply kernel to rows
        k1l = tf.reshape(wavelet_ker[:,0], (self.n, 1, 1))
        k1h = tf.reshape(wavelet_ker[:,1], (self.n, 1, 1))
        rl = tf.nn.conv1d(t1, k1l, stride=2, padding='VALID')
        rh = tf.nn.conv1d(t1, k1h, stride=2, padding='VALID')
        r1 = tf.concat((rl, rh), axis=-1) # out: (b, c, qx, 4)
        r1 = tf.reshape(r1, [self.bs, self.cn, self.ox-self.n+2])
        #######################################
        ## merge core and borders
        #######################################
        r = tf.concat((r2_0, r1[..., 1:-1], r2_1), axis=-1)
        # out: (b, c, nx)
        r = tf.transpose(r, perm=[0, 2, 1])
        # out: (b, nx, c)
        return r

    
