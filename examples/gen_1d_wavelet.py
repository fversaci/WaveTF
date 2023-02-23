import tensorflow as tf
import pywt
from wavetf._general_conv import KerWaveLayer1D, InvKerWaveLayer1D
import numpy as np
import math


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


#ker_bior2_4 = [
#    [0.0,                   -0.0                ],
#    [0.03314563036811941,   0.0                 ],
#    [-0.06629126073623882,  -0.0                ],
#    [-0.1767766952966369,   0.3535533905932738  ],
#    [0.4198446513295126,    -0.7071067811865476 ],
#    [0.9943689110435825,    0.3535533905932738  ],
#    [0.4198446513295126,    -0.0                ],
#    [-0.1767766952966369,   0.0                 ],
#    [-0.06629126073623882,  -0.0                ],
#    [0.03314563036811941,   0.0                 ],
#]

ker_bior2_4 = [
    [0.0,                  0.0                 ],
    [0.0,                  -0.03314563036811941],
    [0.0,                  -0.06629126073623882],
    [0.3535533905932738,   0.1767766952966369  ],
    [0.7071067811865476,   0.4198446513295126  ],
    [0.3535533905932738,   -0.9943689110435825 ],
    [0.0,                  0.4198446513295126  ],
    [0.0,                  0.1767766952966369  ],
    [0.0,                  -0.06629126073623882],
    [0.0,                  -0.03314563036811941],
]

ker_db5 = tf.convert_to_tensor(ker_db5, dtype=tf.float64)
ker_bior2_4 = tf.convert_to_tensor(ker_bior2_4, dtype=tf.float64)

km = { 'db5': ker_db5, 'bior2.4': ker_bior2_4}

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
