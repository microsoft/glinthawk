import numpy as np
from Cython.Build import cythonize
from setuptools import setup, Extension

extensions = [
    Extension(
        "pipe_sim",  # Module name
        sources=["pipe_sim.pyx"],  # Cython source
        language="c++",  # Language
        include_dirs=[np.get_include()],  # Include numpy headers
        extra_compile_args=["-std=c++20"],  # Specify C++ standard
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
