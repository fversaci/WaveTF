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
        lb = 1 - (t1.shape[-1] % 2)
        first = 2.0 * t1[... , 0:1]
        last = 2.0 * t1[... , -1:]
        y = [first - t1[... , i+1:i+2] for i in range(pad_l-1)]
        y.reverse()
        z = [last - t1[... , -2-i:-1-i] for i in range(pad_l-lb)]
        s1 = tf.concat(y + [t1] + z, axis=-1)
        return (s1)
    def kernel_function(self, input):
        # choose float precision accordingly to input
        wavelet_ker = tf.cast(self.wavelet_ker, tf.float32) if (input.dtype==tf.float32) else self.wavelet_ker
        # input: (b, x, c)
        t1 = tf.transpose(input, perm=[0, 2, 1]) # out: (b, c, x)
        ## add padding
        s1 = self.pad_tensor(t1)
        ## s1: (b, c, 2*nx)
        s1 = tf.reshape(s1, [self.bs, self.cn, -1, 1]) # out: (b, c, 2*nx', 1)
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

