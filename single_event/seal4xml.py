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
import lal
import sealcore
import spiir
import time
import healpy as hp

def read_event_info(filepath):
    event_info = np.loadtxt(filepath)
    trigger_time = event_info[0]
    ndet = event_info[1]
    det_codes = np.int64(event_info[2:5])
    snrs = event_info[5:8]
    sigmas = event_info[8:]

    return trigger_time, ndet, det_codes, snrs, sigmas


def extract_info_from_xml(filepath):
    xmlfile = spiir.io.ligolw.coinc.load_coinc_xml(filepath)

    try:
        det_names = list(xmlfile['snrs'].keys())
    except KeyError as err:
        raise KeyError(
            f"snr array data not present {filepath} file. Please check your coinc.xml!"
        ) from err
    
    ndet = len(det_names)

    deff_array = np.array([])
    max_snr_array = np.array([])
    det_code_array = np.array([])
    ntimes_array = np.array([])
    snr_array = np.array([])
    time_array = np.array([])
    #snr_series_list = []
    #timestamps = {det: xmlfile["snrs"][det].index.values for det in det_names}
    lal_det_code = {'L1': 6, 'H1': 5, 'V1':2, 'K1': 14, 'I1': 15}  # new lal
    #ntime = len(xmlfile["snrs"][det_names[0]].index.values)
    #snr_array = np.zeros((ndet, 3, ntime ), dtype=np.dtype('d'))

    #i=0
    for det in det_names:
        deff_array = np.append(deff_array, xmlfile['tables']['postcoh']['deff_'+det])
        max_snr_array = np.append(max_snr_array, xmlfile['tables']['postcoh']['snglsnr_'+det])
        #snr_series_list.append( np.array([xmlfile["snrs"][det].index.values, np.real(xmlfile['snrs'][det]), 1*np.imag(xmlfile['snrs'][det])]).T )
        
        #snr_array[i] = np.array([xmlfile["snrs"][det].index.values, np.real(xmlfile['snrs'][det]), np.imag(xmlfile['snrs'][det])]) #.T
        snr_array = np.append(snr_array, xmlfile['snrs'][det] )
        time_array = np.append(time_array, xmlfile["snrs"][det].index.values)
        ntimes_array = np.append(ntimes_array, len(xmlfile['snrs'][det]))
        #i+=1
        det_code_array = np.append(det_code_array, int(lal_det_code[det]))
        #print("len, {}".format(len(xmlfile["snrs"][det].index.values)) )

    sigma_array = deff_array*max_snr_array

    trigger_time = xmlfile["tables"]["postcoh"]["end_time"].item()
    trigger_time += xmlfile["tables"]["postcoh"]["end_time_ns"].item() * 1e-9
    
    print("xml processing done. ")
    print(f'Trigger time: {trigger_time}')
    print(f'Detectors: {det_names}')
    print(f'SNRs: {max_snr_array}')
    print(f'sigmas: {sigma_array}')

    return trigger_time, ndet, ntimes_array.astype(c_int32), det_code_array.astype(c_int32), max_snr_array, sigma_array, time_array, snr_array



if __name__ == "__main__":
    #print(sealcore.pytest1(3,8))

    time0 = time.time()
    xmlfilepath = 'data/coinc_xml/H1L1V1_1187008882_3_806.xml'
    trigger_time, ndet, ntimes_array, det_code_array, max_snr_array, sigma_array, time_arrays, snr_arrays =\
        extract_info_from_xml(xmlfilepath)
    time1 = time.time()
    print("xml file processing done! Time cost {}s.".format(time1-time0))
    
    start_time = trigger_time-0.01
    end_time = trigger_time+0.01
    ntime_interp = 420

    nside = 16
    npix = hp.nside2npix(nside)
    theta, phi = hp.pixelfunc.pix2ang(nside,np.arange(npix),nest=True)
    ra = phi
    dec = -theta+np.pi/2

    max_net_snr = sum(ss**2 for ss in max_snr_array)**0.5
    a = 0.000466361
    b = 0.00036214
    c = 0.00032248
    d = 0.0005245
    prior_mu = a*max_net_snr + b
    prior_sigma = c*max_net_snr +d

    coh_skymap_bicorr = np.zeros(npix)

    time2 = time.time()
    sealcore.Pycoherent_skymap_bicorr(coh_skymap_bicorr, time_arrays,  snr_arrays,
        det_code_array, sigma_array, ntimes_array,
        ndet, ra, dec, npix,
        start_time, end_time, ntime_interp,
        prior_mu,prior_sigma)
    time3 = time.time()

    print("Skymap calculation done! Time cost {}s.".format(time3-time2))
    print(coh_skymap_bicorr[0])
    

    