# Semi-analytical localization for one event
import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
import ligo.skymap.plot
import astropy_healpix as ah
from ligo.skymap import postprocess
from astropy import units as u
from astropy.coordinates import SkyCoord
from ctypes import *
import os


if __name__ == "__main__":

    event_info = np.loadtxt('data/event_info')

    so_file = "sealcore.so"
    #sealcore = CDLL(so_file)
    sealcore = cdll.LoadLibrary(os.path.abspath(so_file))
    print(sealcore.testfunc1(3,8))