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

import pytest
import tensorflow as tf
import pywt
import math
import numpy as np
from wavetf._wavetf_mm import WaveTFFactory

# numpy line width when printing
np.core.arrayprint._format_options["linewidth"] = 160
np.set_printoptions(suppress=True, precision=4)

#tf.compat.v1.enable_eager_execution()

@pytest.fixture
def gen_sx(szmax = 200):
    return(5 + np.random.randint(szmax))

@pytest.fixture
def gen_sy(szmax = 200):
    return(5 + np.random.randint(szmax))

@pytest.fixture
def even_sx(szmax = 200):
    return(4 + 2*np.random.randint(szmax))

@pytest.fixture
def even_sy(szmax = 200):
    return(4 + 2*np.random.randint(szmax))
       
class Test_Py_VS_MyWavelet():
    ## compare 1d kernel vs pywavelets
    def kernel_1d(self, kernel, sx, float_type=np.float32) :
        ran = np.random.rand(1, sx).astype(float_type)
        # choose appropriate padding for Pywavelets to have same size as wavetf
        if (kernel == 'haar') :
            padmode = 'antireflect'
        else :
            padmode = 'periodization'            
        # pywavelets
        pyw = pywt.dwt(ran, kernel, mode = padmode)
        pyw = np.stack((pyw[0], pyw[1]), axis=-1)
        # wavetf
        ten = tf.convert_to_tensor(ran)
        ten = tf.reshape(ten, ten.shape + (1,))
        w = WaveTFFactory().build(kernel, dim = 1)
        tfw = w.call(ten)
        # comparison
        diff = (tfw - pyw).numpy()
        if (kernel == 'haar') :
            maxd = abs(diff).max()
        else :
            maxd = abs(diff[:,1:-1]).max() # ignore different border handling
        return maxd
    ## float 32
    # test 1d Haar wavelet vs pywalets output
    def test_haar_1d_32(self, gen_sx) :
        kernel = 'haar'; ft=np.float32
        maxd = self.kernel_1d(kernel, gen_sx, float_type=ft)
        assert(maxd < 1e-6)
    # test 1d db2 wavelet vs pywalets output
    def test_db2_1d_32(self, gen_sx) :
        kernel = 'db2'; ft=np.float32
        maxd = self.kernel_1d(kernel, gen_sx, float_type=ft)
        assert(maxd < 1e-6)
    ## float 64
    # test 1d Haar wavelet vs pywalets output
    def test_haar_1d_64(self, gen_sx) :
        kernel = 'haar'; ft=np.float64
        maxd = self.kernel_1d(kernel, gen_sx, float_type=ft)
        assert(maxd < 1e-14)
    # test 1d db2 wavelet vs pywalets output
    def test_db2_1d_64(self, gen_sx) :
        kernel = 'db2'; ft=np.float64
        maxd = self.kernel_1d(kernel, gen_sx, float_type=ft)
        assert(maxd < 1e-14)
    ## compare 2d kernel vs pywavelets
    def kernel_2d(self, kernel, sx, sy, float_type=np.float32) :
        ran = np.random.rand(5, sx, sy).astype(float_type)
        # choose appropriate padding for Pywavelets to have same size as wavetf
        if (kernel == 'haar') :
            padmode = 'antireflect'
        else :
            padmode = 'periodization'            
        # mywavelet
        ten = tf.convert_to_tensor(ran)
        ten = tf.transpose(ten, perm=[1, 2, 0])
        ten = tf.reshape(ten, (1,) + tuple(ten.shape))
        w = WaveTFFactory.build(kernel)
        tfw = w.call(ten)[0]
        tfw = tf.transpose(tfw, perm = [2, 0, 1])
        # pywavelets
        pyw = pywt.dwt2(ran, kernel, mode = padmode)
        pyw = np.concatenate((pyw[0],) + pyw[1])
        # comparison
        diff = (tfw - pyw).numpy()
        if (kernel == 'haar') :
            maxd = abs(diff).max()
        else :
            maxd = abs(diff[:,1:-1,1:-1]).max() # ignore different border handling
        return maxd
    ## float 32
    # test 2d Haar wavelet vs pywalets output
    def test_haar_2d_32(self, gen_sx, gen_sy) :
        kernel = 'haar'; ft=np.float32
        maxd = self.kernel_2d(kernel, gen_sx, gen_sy, float_type=ft)
        assert(maxd < 1e-6)
    # test 2d db2 wavelet vs pywalets output    
    def test_db2_2d_32(self, gen_sx, gen_sy) :
        kernel = 'db2'; ft=np.float32
        maxd = self.kernel_2d(kernel, gen_sx, gen_sy, float_type=ft)
        assert(maxd < 1e-6)
    ## float 64
    # test 2d Haar wavelet vs pywalets output
    def test_haar_2d_64(self, gen_sx, gen_sy) :
        kernel = 'haar'; ft=np.float64
        maxd = self.kernel_2d(kernel, gen_sx, gen_sy, float_type=ft)
        assert(maxd < 1e-14)
    # test 2d db2 wavelet vs pywalets output    
    def test_db2_2d_64(self, gen_sx, gen_sy) :
        kernel = 'db2'; ft=np.float64
        maxd = self.kernel_2d(kernel, gen_sx, gen_sy, float_type=ft)
        assert(maxd < 1e-14)

        
class Test_InverseWavelet: #(unittest.TestCase):
    def setup_class(self, szmax = 200) :
        self.sx = 4 + 2*np.random.randint(szmax)
        self.sy = 4 + 2*np.random.randint(szmax)
    # invert my wavelet
    def my_inv_kernel(self, kernel, dim, sx, sy = 0, float_type=tf.float32) :
        if (dim == 1):
            ten = tf.random.uniform([5, sx, 3], dtype=float_type)
        else:
            ten = tf.random.uniform([5, sx, sy, 3], dtype=float_type)
        # mywavelet
        w = WaveTFFactory().build(kernel, dim = dim)
        tfw = w.call(ten)
        # shuffle + wavelet + shuffle
        w2 = WaveTFFactory().build(kernel, dim = dim, inverse = True)
        tfw = w2.call(tfw)
        # comparison
        diff = abs(ten - tfw)
        maxd = tf.math.reduce_max(diff)
        return maxd
    ## float32 tests
    # verify that my wavelet inverse works for 1d-haar
    def test_my_haar_1d_32(self, even_sx) :
        kernel = 'haar'; dim = 1; ft=tf.float32
        maxd = self.my_inv_kernel(kernel, dim, even_sx, float_type=ft)
        assert(maxd<1e-6) 
    # verify that my wavelet inverse works for 1d-db2
    def test_my_db2_1d_32(self, even_sx) :
        kernel = 'db2'; dim = 1; ft=tf.float32
        maxd = self.my_inv_kernel(kernel, dim, even_sx, float_type=ft)
        assert(maxd<1e-6) 
    # verify that my wavelet inverse works for 2d-haar
    def test_my_haar_2d_32(self, even_sx, even_sy) :
        kernel = 'haar'; dim = 2; ft=tf.float32
        maxd = self.my_inv_kernel(kernel, dim, even_sx, even_sy, float_type=ft)
        assert(maxd<1e-6) 
    # verify that my wavelet inverse works for 2d-db2
    def test_my_db2_2d_32(self, even_sx, even_sy) :
        kernel = 'db2'; dim = 2; ft=tf.float32
        maxd = self.my_inv_kernel(kernel, dim, even_sx, even_sy, float_type=ft)
        assert(maxd<1e-5) 
    ## float64 tests
    # verify that my wavelet inverse works for 1d-haar
    def test_my_haar_1d_64(self, even_sx) :
        kernel = 'haar'; dim = 1; ft=tf.float64
        maxd = self.my_inv_kernel(kernel, dim, even_sx, float_type=ft)
        assert(maxd<1e-14)
    # verify that my wavelet inverse works for 1d-db2
    def test_my_db2_1d_64(self, even_sx) :
        kernel = 'db2'; dim = 1; ft=tf.float64
        maxd = self.my_inv_kernel(kernel, dim, even_sx, float_type=ft)
        assert(maxd<1e-14) 
    # verify that my wavelet inverse works for 2d-haar
    def test_my_haar_2d_64(self, even_sx, even_sy) :
        kernel = 'haar'; dim = 2; ft=tf.float64
        maxd = self.my_inv_kernel(kernel, dim, even_sx, even_sy, float_type=ft)
        assert(maxd<1e-14) 
    # verify that my wavelet inverse works for 2d-db2
    def test_my_db2_2d_64(self, even_sx, even_sy) :
        kernel = 'db2'; dim = 2; ft=tf.float64
        maxd = self.my_inv_kernel(kernel, dim, even_sx, even_sy, float_type=ft)
        assert(maxd<1e-14) 
