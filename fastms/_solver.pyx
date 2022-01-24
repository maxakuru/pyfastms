cimport libc.stdlib
cimport cpython.bytes
cimport numpy as np
import numpy as np

from fastms._solver cimport Par, Solver, ArrayDim


ENGINE_CPU = 0
ENGINE_CUDA = 1

cdef class Parameters:
    cdef Par fms_par

    def __cinit__(self):
        self.fms_par = Par()

    def __init__(self,
        plambda: float = 0.1, 
        alpha: float = 20.0,
		temporal: float = 0.0,
		iterations: int = 10000,
		stop_eps: float = 5e-5,
		stop_k: int = 10,
		adapt_params: bool = False,
		weight: bool = False,
		edges: bool = False,
		use_double: bool = False,
		engine: int = ENGINE_CPU,
		verbose: bool = True
    ):
        self.plambda = plambda
        self.alpha = alpha
        self.temporal = temporal
        self.iterations = iterations
        self.stop_eps = stop_eps
        self.stop_k = stop_k
        self.adapt_params = adapt_params
        self.weight = weight
        self.edges = edges
        self.use_double = use_double
        self.engine = engine
        self.verbose = verbose

    @property
    def plambda(self):
        return self.fms_par.plambda
    @plambda.setter
    def plambda(self, plambda):
        self.fms_par.plambda = plambda

    @property
    def alpha(self):
        return self.fms_par.alpha
    @alpha.setter
    def alpha(self, alpha):
        self.fms_par.alpha = alpha

    @property
    def temporal(self):
        return self.fms_par.temporal
    @temporal.setter
    def temporal(self, temporal):
        self.fms_par.temporal = temporal

    @property
    def iterations(self):
        return self.fms_par.iterations
    @iterations.setter
    def iterations(self, iterations):
        self.fms_par.iterations = iterations

    @property
    def stop_eps(self):
        return self.fms_par.stop_eps
    @stop_eps.setter
    def stop_eps(self, stop_eps):
        self.fms_par.stop_eps = stop_eps

    @property
    def stop_k(self):
        return self.fms_par.stop_k
    @stop_k.setter
    def stop_k(self, stop_k):
        self.fms_par.stop_k = stop_k

    @property
    def adapt_params(self):
        return self.fms_par.adapt_params
    @adapt_params.setter
    def adapt_params(self, adapt_params):
        self.fms_par.adapt_params = adapt_params

    @property
    def weight(self):
        return self.fms_par.weight
    @weight.setter
    def weight(self, weight):
        self.fms_par.weight = weight

    @property
    def edges(self):
        return self.fms_par.edges
    @edges.setter
    def edges(self, edges):
        self.fms_par.edges = edges

    @property
    def use_double(self):
        return self.fms_par.use_double
    @use_double.setter
    def use_double(self, use_double):
        self.fms_par.use_double = use_double

    @property
    def engine(self):
        return self.fms_par.engine
    @engine.setter
    def engine(self, engine):
        self.fms_par.engine = engine

    @property
    def verbose(self):
        return self.fms_par.verbose
    @verbose.setter
    def verbose(self, verbose):
        self.fms_par.verbose = verbose

cdef class FMSSolver:
    cdef Solver *_solver
    cdef Parameters params
    cdef ArrayDim dim

    def __cinit__(self):
        self._solver = new Solver()

    def __init__(self, 
        plambda: float = 0.1, 
        alpha: float = 20.0,
		temporal: float = 0.0,
		iterations: int = 10000,
		stop_eps: float = 5e-5,
		stop_k: int = 10,
		adapt_params: bool = False,
		weight: bool = False,
		edges: bool = False,
		use_double: bool = True,
		engine: int = ENGINE_CPU,
		verbose: bool = True
    ):
        self.params = Parameters(plambda=plambda, alpha=alpha, 
            temporal=temporal, iterations=iterations, stop_eps=stop_eps, 
            stop_k=stop_k, adapt_params=adapt_params, weight=weight, edges=edges,
            use_double=use_double, engine=engine, verbose=verbose)

    def __dealloc__(self):
        # self.free_data()
        del self._solver
        
    def run_float(self, np.ndarray[float, ndim=3, mode="c"] in_img not None):
        cdef np.ndarray[float, ndim=3, mode="c"] in_buf = np.ascontiguousarray(in_img, dtype=np.float64)
        cdef np.ndarray[float, ndim=3, mode="c"] out_buf = np.ascontiguousarray(in_img.copy(), dtype=np.float64)
        # cdef np.float[:,:,::1] out_buf = np.ascontiguousarray(out_im, dtype = np.float)
        # cdef float* out = <float*> out_buf.data

        cdef float *ptr = &out_buf[0,0,0]
        self._solver.run(ptr, &in_buf[0,0,0], self.dim, self.params.fms_par)

        return out_buf

    def run_double_2d(self, np.ndarray[double, ndim=3, mode="c"] in_img not None):
        cdef np.ndarray[double, ndim=3, mode="c"] in_buf = np.ascontiguousarray(in_img, dtype=np.double)
        cdef np.ndarray[double, ndim=3, mode="c"] out_buf = in_img.copy()

        cdef double *ptr = &out_buf[0,0,0]
        self._solver.run(ptr, &in_buf[0,0,0], self.dim, self.params.fms_par)

        return out_buf

    def run_double(self, np.ndarray[double, ndim=3, mode="c"] in_img not None):
        cdef np.ndarray[double, ndim=3, mode="c"] in_buf = np.ascontiguousarray(in_img, dtype=np.double)
        cdef np.ndarray[double, ndim=3, mode="c"] out_buf = np.ascontiguousarray(in_img.copy(), dtype=np.double)

        cdef double *ptr = &out_buf[0,0,0]
        self._solver.run(ptr, &in_buf[0,0,0], self.dim, self.params.fms_par)

        return out_buf


    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, np.ndarray[double, ndim=3, mode="c"] value not None):
        cdef int x, y, c
        self._data = value

        w, h, c = self._data.shape[0], self._data.shape[1], self._data.shape[2]

        self.dim = ArrayDim(w, h, c)
