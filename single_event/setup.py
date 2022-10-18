from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy

sealcore = Extension("sealcore",
                  sources=['sealcore.pyx'],
                  #include_dirs=[numpy.get_include(), '/apps/skylake/software/compiler/gcc/6.4.0/gsl/2.4/include/','/fred/oz016/opt-pipe/include/'],
                  include_dirs=[numpy.get_include(), '/apps/skylake/software/GSL/2.5-GCC-9.2.0/include/','/fred/oz016/opt-pipe/include/'],
                  #library_dirs=['/apps/skylake/software/compiler/gcc/6.4.0/gsl/2.4/lib', '/fred/oz016/opt-pipe/lib'],
                  library_dirs=['/apps/skylake/software/GSL/2.5-GCC-9.2.0/lib', '/fred/oz016/opt-pipe/lib'],
                  libraries=['m', 'gsl', 'gslcblas', 'lal'],
                  language='c',
                  extra_compile_args=['-fopenmp', '-O3'],
                  extra_link_args=[]
                  )

setup(name='sealcore',
      ext_modules=cythonize([sealcore]))

#python setup.py build_ext --inplace