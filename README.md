# WaveTF: a 1D and 2D wavelet library for TensorFlow and Keras

## Overview

WaveTF is a TensorFlow library which implements 1D and 2D wavelet
transforms, making them available as Keras layers, which can thus be
easily plugged into machine learning workflows.

WaveTF can also be used outside of machine learning contexts, as a
parallel wavelet computation tool, running on CPUs, GPUs or Google
Cloud TPUs, and supporting, transparently at runtime, both 32- and
64-bit floats.

It accepts batched, multichannel inputs, e.g., for the 2D case, inputs
of shape [batch_size, dim_x, dim_y, channels]. Currently, Haar and
Daubechies-N=2 wavelet kernels are supported, and the input signal is
extended via anti-symmetric-reflect padding, which preserves its first
order finite difference at the border.

## Installation

Install with pip:

```bash
$ pip3 install .
```

## Requirements

WaveTF requires TensorFlow 2 to be installed. If you want to run the
tests you will also need pytest, numpy and PyWavelets.

## Documentation

API documentation for the latest WaveTF version is available via
[ReadTheDocs](https://wavetf.readthedocs.io/en/latest/).

### Generation via Sphinx

Alternatively, it can be generated locally via Sphinx.

To install Sphinx:
```bash
$ pip3 install sphinx sphinx_rtd_theme
```

To generate the html documentation (accessible at location `docs/build/html/index.html`):
```bash
$ make -C docs/ html
```

## Further details

An article describing in detail WaveTF's implementation and
performance has been presented at the [CADL
workshop](https://ailb-web.ing.unimore.it/cadl2020/) at [ICPR
2020](http://www.icpr2020.it/) and is available either via the
[Springer website](https://doi.org/10.1007/978-3-030-68763-2_46) or
the CRS4 publications repository [(direct link to
PDF)](http://publications.crs4.it/pubdocs/2021/Ver21/wavetf.pdf).

### Citation

```bibtex
@InProceedings{wavetf,
  author="Versaci, Francesco",
  title="WaveTF: A Fast 2D Wavelet Transform for Machine Learning in Keras",
  booktitle="Pattern Recognition. ICPR International Workshops and Challenges",
  year="2021",
  publisher="Springer International Publishing",
  pages="605--618",
  isbn="978-3-030-68763-2"
}
```

## Usage

WaveTF directly exposes a single class, which is a factory for Keras
layers which implement the Haar and Daubechies-N=2 wavelet transforms
and anti-transforms. Its use is pretty straightforward.

```python
import tensorflow as tf
from wavetf import WaveTFFactory

# input tensor
t0 = tf.random.uniform([32, 300, 200, 3])
# transform
w = WaveTFFactory().build('db2', dim=2)
t1 = w.call(t0)
# anti-transform
w_i = WaveTFFactory().build('db2', dim=2, inverse=True)
t2 = w_i.call(t1)
# compute difference
delta = abs(t2-t0)
print(f'Precision error: {tf.math.reduce_max(delta)}')
```

### Examples

Some basic examples, including a simple wavelet-enriched Convolutional
Neural Network (CNN), are available in the [examples directory](examples/).

## Author

`WaveTF` is developed by
  * Francesco Versaci, CRS4 <francesco.versaci@gmail.com>

## License

WaveTF is licensed under the Apache License, Version 2.0.
See LICENSE for further details.
