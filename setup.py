import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

sealcore = Extension(
    name="sealcore",
    sources=[
        'sealgw/calculation/cealcore.pyx',
        'cextern/chealpix/chealpix.c',
        'cextern/exponential_integral_Ei.c',
    ],
    libraries=['m', 'gsl', 'gslcblas', 'lal'],
    language='c',
    extra_compile_args=['-fopenmp', '-O3', '-lchealpix'],
    extra_link_args=['-fopenmp', '-O3', '-lchealpix'],
)

install_requires = [
    'numpy',
    'scipy',
    'cython',
    'matplotlib',
    'bilby',
    'ligo.skymap',
    'astropy',
    'spiir',
]

setup(
    name='sealgw',
    version="0.0.1",
    description='SealGW: SEmi-Analytical Localization for Gravitational Waves',
    author='Qian Hu',
    author_email='q.hu.2@research.gla.ac.uk',
    url='https://github.com/marinerq/sealgw',
    license='MIT',
    python_requires='>=3',
    packages=["sealgw", 'sealgw.calculation', 'sealgw.simulation'],
    install_requires=install_requires,
    include_dirs=[numpy.get_include(), 'cextern/', 'cextern/chealpix'],
    setup_requires=['numpy', 'cython', 'setuptools_scm'],
    entry_points={},
    ext_modules=cythonize([sealcore]),
)
