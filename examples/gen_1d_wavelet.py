import tensorflow as tf
import pywt
from wavetf._general_conv import KerWaveLayer1D, InvKerWaveLayer1D
import numpy as np
import math


ker_db3 = [
    [0.3326705529509569,     0.035226291882100656 ],
    [0.8068915093133388,     0.08544127388224149  ],
    [0.4598775021193313,     -0.13501102001039084 ],
    [-0.13501102001039084,   -0.4598775021193313  ],
    [-0.08544127388224149,   0.8068915093133388   ],
    [0.035226291882100656,   -0.3326705529509569  ],
]

ker_db5 = [
    [0.16010239797419293,    0.0033357252854737712],
    [0.6038292697971896,     0.012580751999081999 ],
    [0.7243085284377729,     -0.006241490212798274],
    [0.13842814590132074,    -0.07757149384004572 ],
    [-0.24229488706638203,   -0.032244869584638375],
    [-0.032244869584638375,  0.24229488706638203  ],
    [0.07757149384004572,    0.13842814590132074  ],
    [-0.006241490212798274,  -0.7243085284377729  ],
    [-0.012580751999081999,  0.6038292697971896   ],
    [0.0033357252854737712,  -0.16010239797419293 ],
]

ker_db3 = tf.convert_to_tensor(ker_db3, dtype=tf.float64)
ker_db5 = tf.convert_to_tensor(ker_db5, dtype=tf.float64)

km = { 'db3': ker_db3, 'db5': ker_db5}

errors = False
for ker, tf_ker in km.items():
    # init Wavelet layer with user-provide kernel
    dir_t = KerWaveLayer1D(tf_ker)
    inv_t = InvKerWaveLayer1D(tf_ker)
    for sx in range(100, 120, 2):
        # create random numpy array
        ran = np.random.rand(5, sx).astype(np.float64)
        # run pywavelet equivalent
        pyw = pywt.dwt(ran, ker, mode='periodization')
        pyw = np.stack((pyw[0], pyw[1]), axis=-1)
        # convert numpy array to tensor
        ten = tf.convert_to_tensor(ran)
        ten = tf.reshape(ten, ten.shape + (1,))
        tfw = dir_t.call(ten)
        # compute difference
        diff = (tfw - pyw).numpy()
        bd = tf_ker.shape[0]//2
        # check inverse
        tfiw = inv_t.call(tfw)
        # compute max of difference
        maxd = abs(diff[:, bd:-bd]).max() # ignore border
        invd = abs((tfiw - ten).numpy()).max()
        mm = max(maxd, invd)
        if (mm>1e-5):
            print(ker)
            print(maxd, invd)
            errors = True

if (errors):            
    print('Error: wavelets differ too much')
else:
    print('Test completed without errors')
