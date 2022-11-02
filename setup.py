from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy

sealcore = Extension(name = "sealcore",
                  sources=['sealgw/calculation/cealcore.pyx'],
                  #include_dirs=['/apps/skylake/software/GSL/2.5-GCC-9.2.0/include/','/fred/oz016/opt-pipe/include/'],
                  #library_dirs=['/apps/skylake/software/GSL/2.5-GCC-9.2.0/lib', '/fred/oz016/opt-pipe/lib'],
                  libraries=['m', 'gsl', 'gslcblas', 'lal'],
                  language='c',
                  extra_compile_args=['-fopenmp', '-O3'],  #'-fopenmp', '-O3','-lboost_python39'
                  extra_link_args=['-fopenmp', '-O3']
                  )

setup(
    name='sealgw',
    version="0.0.0",
    description='SealGW: SEmi-Analytical Localization for Gravitational Waves',
    author='Qian Hu',
    author_email='q.hu.2@research.gla.ac.uk',
    url='https://github.com/marinerq/sealgw',
    license='MIT',
    python_requires='>=3',
    packages=["sealgw", 'sealgw.calculation', 'sealgw.simulation'],
    install_requires=['numpy', 'scipy', 'cython', 'matplotlib', 'bilby', 'ligo.skymap', 'astropy', 'spiir'],
    include_dirs = [numpy.get_include()],
    setup_requires=['numpy', 'cython', 'setuptools_scm'],
    #package_data={"": ['*.c', '*.pyx', '*.pxd']},
    entry_points={},
    ext_modules=cythonize([sealcore])
    )

#rm -r build/ *.so
#python setup.py build_ext --inplace
#python setup.py clean --all


#python setup.py install --record sealgw_install_record.txt 
#xargs rm -rf < --sealgw_install_record.txt 


#pip install . 
#pip uninstall sealgw