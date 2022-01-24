# pyfastms

**WIP**

Python bindings for the c++ implementation of [fastms](https://github.com/tum-vision/fastms); an efficient algorithm for finding piecewise smooth and constant approximations to images by minimizing the [Mumford-Shah](https://en.wikipedia.org/wiki/Mumford%E2%80%93Shah_functional) functional. ([Paper](https://vision.in.tum.de/_media/spezial/bib/strekalovskiy_cremers_eccv14.pdf)) 

## Use
```py
import numpy as np
import fastms

arr = np.zeros((100, 100, 3), dtype=np.float64) # ...
solver = fastms.FMSSolver()
res: np.ndarray = solver.run(arr) # run_float(), run_double(), ...
```

### Limitations & differences
* No CUDA support
* No OpenMP support
* Depends on Numpy, but not on Matlab mex/OpenCV

### Dependencies
* Cython 0.29.26
* Python 3.9.8
* Numpy 1.20.0
* Other stuff may work, that's what I'm using

### Build
```py
poetry build
```

### Install
```py
poetry install
```

### Test
```py
poetry run test
```
