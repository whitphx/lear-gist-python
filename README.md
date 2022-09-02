# lear-gist-python
[![wercker status](https://app.wercker.com/status/5285318d112056b85e8f3643e8a4b9aa/s/master "wercker status")](https://app.wercker.com/project/byKey/5285318d112056b85e8f3643e8a4b9aa)


[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/D1D2ERWFG)

<a href="https://www.buymeacoffee.com/whitphx" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="180" height="50" ></a>

[![GitHub Sponsors](https://img.shields.io/github/sponsors/whitphx?label=Sponsor%20me%20on%20GitHub%20Sponsors&style=social)](https://github.com/sponsors/whitphx)

Python library to extract [A. Torralba's GIST descriptor](http://people.csail.mit.edu/torralba/code/spatialenvelope/).

This is just a wrapper for [Lear's GIST implementation](http://lear.inrialpes.fr/software) written in C. It supports both Python 2 and Python 3 and was tested under Python 2.7.15 and Python 3.6.6 on Linux.

## How to build and install

### Pre-requirements
Following packages must be installed before building and installing `lear-gist-python`.

##### numpy
```shell
$ pip install numpy
```

##### FFTW
[FFTW](http://www.fftw.org/) is required to build lear_gist.
Please download the source, then build and install like following. (Install guide is [here](http://www.fftw.org/fftw3_doc/Installation-on-Unix.html). Please refer for defail.)
Make sure `--enable-single` and `--enable-shared` options are set to `./configure`.
```shell
$ ./configure --enable-single --enable-shared
$ make
$ make install
```

Because:
- lear-gist requires *float version* FFTW to work with (`--enable-single`).
- lear-gist-python requires FFTW to be compiled with `-fPIC` option (`--enable-shared`).

### Build and install
Download lear_gist
```shell
$ ./download-lear.sh
```

Build and install
```shell
$ python setup.py build_ext
$ python setup.py install
```

If `fftw3f` is installed in non-standard path (for example, `$HOME/local`),
use `-I` and `-L` options:
```shell
$ python setup.py build_ext -I $HOME/local/include -L $HOME/local/lib
```

## Usage
```python
import gist
import numpy as np

img = ... # numpy array containing an image
descriptor = gist.extract(img)
```

## Scene classification sample
This sample uses [8 Scene Categories Dataset](http://people.csail.mit.edu/torralba/code/spatialenvelope/).

`scikit-learn` and `scikit-image` are required.
```shell
$ pip install scikit-learn scikit-image
```

```shell
cd sample
sh download-8scene.sh
# Extract GIST features from images in "spatial_envelope_256x256_static_8outdoorcategories" directory and save them into "features" directory
python feature_extraction.py spatial_envelope_256x256_static_8outdoorcategories features
# Train and test a multi-class linear classifier by features in "features" directory
python scene_classification.py features
```

## API
### `gist.extract(img, nblocks=4, orientations_per_scale=(8, 8, 4))`
* `img`: A numpy array (an instance of `numpy.ndarray`) which contains an image and whose shape is `(height, width, 3)`.
* `nblocks`: Use a grid of `nblocks * nblocks` cells.
* `orientations_per_scale`: Use `len(orientations_per_scale)` scales and compute `orientations_per_scale[i]` orientations for i-th scale.
