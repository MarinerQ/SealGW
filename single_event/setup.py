from distutils.extension import Extension
from distutils.core import setup
from Cython.Build import cythonize
import numpy

sealcore = Extension(name = "sealcore",
                  sources=['cealcore.pyx'],
                  #sources=['test.pyx'],
                  #include_dirs=['/apps/skylake/software/GSL/2.5-GCC-9.2.0/include/','/fred/oz016/opt-pipe/include/'],
                  #library_dirs=['/apps/skylake/software/GSL/2.5-GCC-9.2.0/lib', '/fred/oz016/opt-pipe/lib'],
                  #include_dirs=['/Users/qianhu/opt/anaconda3/envs/igwn-py39/include/lal','/Users/qianhu/opt/anaconda3/envs/igwn-py39/include/gsl'],
                  #library_dirs=['/apps/skylake/software/GSL/2.5-GCC-9.2.0/lib', '/fred/oz016/opt-pipe/lib'],
                  libraries=['m', 'gsl', 'gslcblas', 'lal'],
                  language='c',
                  extra_compile_args=[],  #'-fopenmp', '-O3','-lboost_python39'
                  extra_link_args=[]
                  )

setup(ext_modules=cythonize([sealcore]))

#python setup.py build_ext --inplace
#python setup.py clean --all