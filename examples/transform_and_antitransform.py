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

import tensorflow as tf
from wavetf import WaveTFFactory


def run():
    # generate random tensor: block_size x dim_x x dim_y x channels
    bs = 10
    dx = 100
    dy = 80
    chans = 3
    kernel = "db2"  # 'db2' or 'haar'
    dim = 2  # 2D transform
    ten = tf.random.uniform([bs, dx, dy, chans])
    w = WaveTFFactory().build(kernel, dim=dim)
    print(f"Transforming from shape {ten.shape}... ", end="")
    direct = w.call(ten)
    print(f"to shape {direct.shape}")
    w_i = WaveTFFactory().build(kernel, dim=dim, inverse=True)
    print(f"Anti-transforming from shape {direct.shape}... ", end="")
    inv_dir = w_i.call(direct)
    print(f"to shape {inv_dir.shape}")
    delta = abs(ten - inv_dir)
    print(f"Precision error: {tf.math.reduce_max(delta)}")


if __name__ == "__main__":
    run()
