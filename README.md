# lear-gist-python
Python library to extract [A. Torralba's GIST descriptor](http://people.csail.mit.edu/torralba/code/spatialenvelope/).

This is just a wrapper for [Lear's GIST implementation](http://lear.inrialpes.fr/software) written in C. It supports both Python 2 and Python 3 and was tested under Python 2.7.10 and Python 3.4.3 on Linux.

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
$ sh download-lear.sh
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
