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
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Dense, Lambda, BatchNormalization, GlobalAveragePooling2D, Activation
from tensorflow.keras.models import Model
from wavetf import WaveTFFactory

def wavelet_cnn(input_shape, ks=3, baselev=4, wavelet=True,
                wave_kern='db2', hsv=True, convrep=2, num_classes=2):
    inputs = Input(input_shape)
    chans = input_shape[2] # number of channels, e.g., 3 if RGB
    bl = baselev

    # wavelet computation
    if (wavelet) :
        # convert RGB to HSV?
        if (hsv):
            wave0 = Lambda(lambda x: tf.image.rgb_to_hsv(x))(inputs)
        else:
            wave0 = inputs
        # compute 4 level of wavelet
        wave1 = WaveTFFactory.build(wave_kern)(wave0)
        # compute new wavelet features from LL componenents
        wave2 = WaveTFFactory.build(wave_kern)(wave1[:,:,:,:chans]) 
        wave3 = WaveTFFactory.build(wave_kern)(wave2[:,:,:,:chans])
        wave4 = WaveTFFactory.build(wave_kern)(wave3[:,:,:,:chans])
        # normalize
        waves = [wave1, wave2, wave3, wave4]
        for l in waves :
            l = BatchNormalization()(l)
    else :
        wave1 = wave2 = wave3 = wave4 = None

    kinit ='glorot_normal' # 'he_normal'

    def rep_conv(cnn, scale = 1) :
        for i in range(convrep) :
            cnn = Conv2D(scale * bl, ks, activation = 'relu', padding = 'same',
                         kernel_initializer = kinit)(cnn)
        return cnn

    def pool_down(cnn, mul):
        cnn = Conv2D(mul * bl, ks, activation = 'relu', padding = 'same',
                     kernel_initializer = kinit, strides=(2, 2))(cnn)
        return (cnn)
    
    cnn = inputs
    cnn = rep_conv(cnn, 1)

    for l in range(4) :
        cnn = pool_down(cnn, 2**(l+1))
        cnn = rep_conv(cnn, 2**(l+1))
        if (wavelet):
    	    cnn = Concatenate(axis=3)([cnn, waves[l]])

    # output
    cnn = Conv2D(2048, ks)(cnn)
    cnn = Activation('relu')(cnn)
    cnn = GlobalAveragePooling2D()(cnn)
    outputs = Dense(num_classes, activation='softmax')(cnn)
    model = Model(inputs = inputs, outputs = outputs)
    return model


