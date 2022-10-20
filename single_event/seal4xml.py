# Semi-analytical localization for one event
#import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
#import ligo.skymap.plot
#import astropy_healpix as ah
#from ligo.skymap import postprocess
#from astropy import units as u
#from astropy.coordinates import SkyCoord
#from ctypes import *
import os
import lal
import sealcore
import spiir

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
    #snr_series_list = []
    #timestamps = {det: xmlfile["snrs"][det].index.values for det in det_names}
    lal_det_code = {'L1': 6, 'H1': 5, 'V1':2, 'K1': 14, 'I1': 15}  # new lal
    ntime = len(xmlfile["snrs"][det_names[0]].index.values)
    data_array = np.zeros((ndet, 3, ntime ))

    i=0
    for det in det_names:
        deff_array = np.append(deff_array, xmlfile['tables']['postcoh']['deff_'+det])
        max_snr_array = np.append(max_snr_array, xmlfile['tables']['postcoh']['snglsnr_'+det])
        #snr_series_list.append( np.array([xmlfile["snrs"][det].index.values, np.real(xmlfile['snrs'][det]), 1*np.imag(xmlfile['snrs'][det])]).T )
        
        data_array[i] = np.array([xmlfile["snrs"][det].index.values, np.real(xmlfile['snrs'][det]), np.imag(xmlfile['snrs'][det])]) #.T
        i+=1
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

    return trigger_time, ndet, ntime, det_code_array, max_snr_array, sigma_array, data_array



if __name__ == "__main__":
    print(sealcore.pytest1(3,8))

    xmlfilepath = 'data/coinc_xml/H1L1V1_1187008882_3_806.xml'
    trigger_time, ndet, ntime, det_code_array, max_snr_array, sigma_array, data_array =\
        extract_info_from_xml(xmlfilepath)

    '''
    delta_t = 1/2048
    lal_data_list = []
    for i in range(ndet):
        ep = lal.LIGOTimeGPS(snr_series_list[i][0])
        lal_data = lal.CreateCOMPLEX8TimeSeries("test"+str(i),ep,0,delta_t,lal.DimensionlessUnit,len(snr_series_list[i]))
        lal_data.data.data[:] = snr_series_list[i][:,1] + snr_series_list[i][:,1]*1j
        lal_data_list.append(lal_data)
    '''
    data_array_1d = np.reshape(data_array, ndet*ntime*3)
    print(data_array_1d.shape)
    
    sealcore.PytestLALseries(data_array_1d, ndet, ntime)
    

    