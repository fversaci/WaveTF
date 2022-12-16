# This is a wavelet-modified version of a U-Net.
#
# The original U-Net code (MIT licensed) is taken from
# https://github.com/VidushiBhatia/U-Net-Implementation
#
#
# Copyright 2022 CRS4 (http://www.crs4.it/)
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
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Concatenate,
    Dense,
    Lambda,
    BatchNormalization,
    GlobalAveragePooling2D,
    Activation,
    Conv2DTranspose,
)
from tensorflow.keras.models import Model
from wavetf import WaveTFFactory


def EncoderMiniBlock(
    inputs, n_filters=32, dropout_prob=0.3, max_pooling=True, wave_kern="db2"
):
    conv = Conv2D(
        n_filters,
        3,  # Kernel size
        activation="relu",
        padding="same",
        kernel_initializer="HeNormal",
    )(inputs)
    conv = Conv2D(
        n_filters,
        3,  # Kernel size
        activation="relu",
        padding="same",
        kernel_initializer="HeNormal",
    )(conv)
    conv = BatchNormalization()(conv, training=False)
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    if max_pooling:
        # next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
        next_layer = WaveTFFactory.build(wave_kern)(conv)
    else:
        next_layer = conv
    skip_connection = conv
    return next_layer, skip_connection


def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32, wave_kern="db2"):
    # up = Conv2DTranspose(
    #     n_filters, (3, 3), strides=(2, 2), padding="same"  # Kernel size
    # )(prev_layer_input)
    up = WaveTFFactory.build(wave_kern, inverse=True)(prev_layer_input)
    merge = Concatenate(axis=3)([up, skip_layer_input])
    conv = Conv2D(
        n_filters,
        3,  # Kernel size
        activation="relu",
        padding="same",
        kernel_initializer="HeNormal",
    )(merge)
    conv = Conv2D(
        n_filters,
        3,  # Kernel size
        activation="relu",
        padding="same",
        kernel_initializer="HeNormal",
    )(conv)
    return conv


def UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=3):
    inputs = Input(input_size)
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(
        cblock1[0], n_filters * 2, dropout_prob=0, max_pooling=True
    )
    cblock3 = EncoderMiniBlock(
        cblock2[0], n_filters * 4, dropout_prob=0, max_pooling=True
    )
    cblock4 = EncoderMiniBlock(
        cblock3[0], n_filters * 8, dropout_prob=0.3, max_pooling=True
    )
    cblock5 = EncoderMiniBlock(
        cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False
    )
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1], n_filters * 8)
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1], n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1], n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1], n_filters)
    conv9 = Conv2D(
        n_filters, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(ublock9)
    conv10 = Conv2D(n_classes, 1, padding="same")(conv9)
    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    return model


net = UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=3)
print(net.summary())
