# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION


import subprocess
import shlex

from setuptools import Extension
from Cython.Build import cythonize

import numpy as np


def pkg_config(pkg_name, command):
    return subprocess.check_output(["pkg-config", command, pkg_name]).decode(
        "utf8"
    )


def build(setup_kwargs):
    """Needed for the poetry building interface."""

    # extra_compile_args = pkg_config("opencv", '--cflags')
    # extra_link_args = pkg_config("opencv", '--libs')
    # extra_compile_args = shlex.split(extra_compile_args)
    # extra_link_args = shlex.split(extra_link_args)

    extra_compile_args = ['-DDISABLE_OPENCV',
                          '-DDISABLE_CUDA',
                          '-DUSE_OPENMP',
                          '-DUSE_MEX']
    extra_link_args = []

    extensions = [

        Extension(
            "fastms._solver",
            sources=[
                "fastms/_solver.pyx",
                "src/libfastms/util/has_cuda.cpp",
                # "src/libfastms/util/image.cpp",
                "src/libfastms/solver/solver_host.cpp",
                "src/libfastms/solver/solver_base.cpp"
            ],
            include_dirs=[np.get_include(), "src/libfastms"],
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ]

    # extensions = cythonize(extensions)
    extensions = cythonize(extensions,
                           compiler_directives={'language_level': "3"})

    setup_kwargs.update(
        {
            "ext_modules": extensions,
            "include_dirs": [np.get_include()],
        }
    )
