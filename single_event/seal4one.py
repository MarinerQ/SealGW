# Semi-analytical localization for one event
#import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
#import ligo.skymap.plot
#import astropy_healpix as ah
#from ligo.skymap import postprocess
#from astropy import units as u
#from astropy.coordinates import SkyCoord
from ctypes import *
import os
#os.environ['LD_LIBRARY_PATH'] = '/apps/skylake/software/compiler/gcc/6.4.0/gsl/2.4/lib'
#os.environ['LD_LIBRARY_PATH'] = '/fred/oz016/opt-pipe/lib' 
import sealcore
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/fred/oz016/opt-pipe/lib


if __name__ == "__main__":

    event_info = np.loadtxt('data/event_info')

    #so_file = "sealcore.so"
    #sealcore = CDLL(so_file)
    #sealcore = cdll.LoadLibrary(os.path.abspath(so_file))
    print(sealcore.testfunc1(3,8))