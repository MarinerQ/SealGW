# source /fred/oz016/qian/spiirenv/bin/activate

# Targets:
# 1. Read .xml file
# 2. Store SNR, sigma as txt files

# The process may be faster if we do the above in C code. 

from pathlib import Path
import pandas as pd
import numpy as np
import spiir  # quite slow to import?


if __name__ == "__main__":

    #data_dir = Path("data/coinc_xml/")
    data_dir = Path("/fred/oz016/qian/test/")
    coinc_xml = data_dir / "H1L1V1_1187008882_3_806.xml"  # or H1L1_...
    
    print(f'Processing coinc.xml file from {coinc_xml}...')

    xmlfile = spiir.io.ligolw.coinc.load_coinc_xml(coinc_xml)

    try:
        det_names = list(xmlfile['snrs'].keys())
    except KeyError as err:
        raise KeyError(
            f"snr array data not present {coinc_xml} file. Please check your coinc.xml!"
        ) from err
    
    ndet = len(det_names)

    # Calculate sigma = sqrt((h|h)) = deff/SNR
    deff_array = np.array([])
    max_snr_array = np.array([])
    for det in det_names:
        deff_array = np.append(deff_array, xmlfile['tables']['postcoh']['deff_'+det])
        max_snr_array = np.append(max_snr_array, xmlfile['tables']['postcoh']['snglsnr_'+det])
    sigma_array = deff_array*max_snr_array


    # Is trigger time stored in xml file? 
    timestamp = xmlfile['snrs'][det].index.to_numpy()
    snr_timeseries_dict = xmlfile['snrs']
    netsnr_timeseries = sum([abs(snr_timeseries_dict[det]) ** 2 for det in det_names])
    trigger_time = timestamp[np.argmax(netsnr_timeseries)]
    #trigger_time = timestamp[np.argmax(abs(snr_timeseries_dict['L1']))]

    # Save event info
    # trigger_time, ndet,    detcode1, ..., detcodeN,    max_snr1, ..., max_snrN,    sigma1, ..., sigmaN
    # 1+1+N+N+N = 3N+2 elements
    event_info = np.array([trigger_time, ndet])

    # Here detector code is for old version of LAL(6.49.0), see:
    # /fred/oz016/opt-pipe/include/lal/LALDetectors.h
    # Be careful when chenge to new version - the code is different, see:
    # https://lscsoft.docs.ligo.org/lalsuite/lal/_l_a_l_detectors_8h_source.html#l00168
    lal_det_code = {'L1': 5, 'H1': 4, 'V1':1, 'K1': 16, 'I1': 17}  
    det_code_array = np.array([])
    for det in det_names:
        event_info = np.append(event_info, lal_det_code[det])
    event_info = np.append(event_info, max_snr_array)
    event_info = np.append(event_info, sigma_array)
    np.savetxt('data/event_info', event_info)


    # Save SNR
    for det in det_names:
        snr_to_save = np.array([timestamp, np.real(snr_timeseries_dict[det]), 1*np.imag(snr_timeseries_dict[det])]).T 
        np.savetxt(f'data/snr_data/snr_det{lal_det_code[det]}', snr_to_save)


    print(f'Trigger time: {trigger_time}')
    print(f'Detectors: {det_names}')
    print(f'SNRs: {max_snr_array}')
    print(f'sigmas: {sigma_array}')
    print('SNR and event info have been saved. \n')