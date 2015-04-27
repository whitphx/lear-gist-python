# lear-gist-python
Python library to extract [A. Torralba's GIST descriptor](http://people.csail.mit.edu/torralba/code/spatialenvelope/).

This is just a wrapper of [Lear's GIST implementation](http://lear.inrialpes.fr/software) written in C.

## How to build and install

### Pre-requirements
[FFTW](http://www.fftw.org/) is required to build lear_gist.

### Build and install
Download lear_gist
```shell
sh download-lear.sh
```

Build and install
```shell
python setup.py build_ext
python install
```

If `fftw3f` is installed in non-standard path (for example, `$HOME/local`),
use `-I` and `-L` options:
```shell
python setup.py build_ext -I $HOME/local/include -L $HOME/local/lib
```

## Usage
```python
import gist
import numpy as np

img = ... # numpy array containing an image
descriptor = gist.extract(img)
```
