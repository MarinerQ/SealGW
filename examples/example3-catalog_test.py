import bilby
import numpy as np 
import time
import sys
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
from functools import partial
from matplotlib.pyplot import MultipleLocator
from bilby.gw import conversion
from scipy.optimize import leastsq
import json
from pycbc.filter import matched_filter,matched_filter_core
from pycbc.types.timeseries import TimeSeries
from pycbc.types.frequencyseries import FrequencySeries
from lal import LIGOTimeGPS

import sealgw
from sealgw import seal
import sealgw.calculation as sealcal
import sealgw.simulation as sealsim


seal_O2 = seal.Seal('example_outputs/sealconfig_O2_lhv_BNS.txt')

Nsample = 1000
det_name_list = ['L1', 'H1', 'V1']
source_type = "BNS"
ncpu = 4
save_filename = 'example_outputs/catalog_statistics.txt' 
psd_files = ['example_inputs/L1_psd.txt','example_inputs/H1_psd.txt','example_inputs/V1_psd.txt']

catalog_stats = seal_O2.catalog_test(Nsample, det_name_list, source_type, ncpu, save_filename,duration = 320,
                      use_bilby_psd = False, custom_psd_path = psd_files)

