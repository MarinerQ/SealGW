# Semi-analytical localization for one event
import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
import ligo.skymap.plot
import astropy_healpix as ah
from ligo.skymap import postprocess
from astropy import units as u
from astropy.coordinates import SkyCoord



if __name__ == "__main__":

    with open('deta/event_info') as f:
        event_info = f.readlines()