WaveTF package
##############

WaveTF directly exposes only one class, which is a factory to create
Keras layers for the supported wavelet transformations:

.. autoclass:: wavetf.WaveTFFactory
   :members:

|
      
Keras layers
============

Here is the syntax of the wavelet Keras layers built by WaveTF,
depending on the number of dimensions they work on (1D vs 2D), and if
they are transforming or antitransforming:

1D direct transform
-------------------

.. autoclass:: wavetf._base_wavelets.DirWaveLayer1D
   :members: call
   :show-inheritance:

1D inverse transform
--------------------
      
.. autoclass:: wavetf._base_wavelets.InvWaveLayer1D
   :members: call
   :show-inheritance:

2D direct transform
-------------------
      
.. autoclass:: wavetf._base_wavelets.DirWaveLayer2D
   :members: call
   :show-inheritance:

2D inverse transform
--------------------

.. autoclass:: wavetf._base_wavelets.InvWaveLayer2D
   :members: call
   :show-inheritance:

