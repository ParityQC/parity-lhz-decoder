import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [
            Extension(
                name="encircled_nodes_finder",
                sources=["src/parity_code/decoders/_cython/encircled_nodes_finder.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=["-w"],  # suppress all warnings
            ),
            Extension(
                name="divide_into_clusters_cython",
                sources=["src/parity_code/decoders/_cython/divide_into_clusters.pyx"],
                language="c++",
                extra_compile_args=["-w"],  # suppress all warnings
            ),
            Extension(
                name="compiled_post_processing",
                sources=["src/parity_code/decoders_compiled/_cython/compiled_post_processing.pyx"],
                extra_compile_args=["-w"],  # suppress all warnings
            ),
        ],
        compiler_directives={"language_level": "3"},
    )
)
