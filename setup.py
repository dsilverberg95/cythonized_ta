from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# Define the Cython extensions
extensions = [
    Extension(
        "cythonized_ta.cython_ta_funcs",  # Extension module name
        ["cythonized_ta/cython_ta_funcs.pyx"],  # Path to the .pyx file
        include_dirs=[numpy.get_include()],  # Include NumPy headers
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

# Setup configuration
setup(
    name="cythonized-ta",  # Distribution name (for pip)
    version="0.1.0",  # Package version
    description="A Cython-optimized library for technical analysis functions",
    author="Your Name",
    packages=find_packages(),  # Automatically find and include the Python package
    ext_modules=cythonize(extensions),  # Always Cythonize the extensions
    install_requires=["numpy", "Cython"],  # Dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Minimum Python version
)