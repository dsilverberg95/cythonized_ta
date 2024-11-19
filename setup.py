from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Check if we're in a development environment
use_cython = os.path.exists("cythonized_ta/cython_ta_funcs.pyx")

# Choose the appropriate file extension
ext = "pyx" if use_cython else "c"
extensions = [
    Extension(
        "cythonized_ta.cython_ta_funcs",
        [f"cythonized_ta/cython_ta_funcs.{ext}"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="cythonized_ta",
    ext_modules=cythonize(extensions) if use_cython else extensions,
    install_requires=[
        "Cython" if use_cython else "numpy",
        "numpy"
    ],
)