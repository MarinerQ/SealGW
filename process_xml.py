# source /fred/oz016/qian/spiirenv/bin/activate

# Targets:
# 1. Read .xml file
# 2. Store SNR, sigma as txt files

# The process may be faster if we do the above in C code. 

print('\nProcessing .xml file...')
from pathlib import Path
import pandas as pd
import numpy as np
import spiir  # quite slow to import?

data_dir = Path("data/coinc_xml/")
coinc_xml = data_dir / "H1L1V1_1344284404_273_202.xml"

xmlfile = spiir.io.ligolw.coinc.load_coinc_xml(coinc_xml)

det_names = list(xmlfile['snrs'].keys())
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



# Save SNR
for det in det_names:
    snr_to_save = np.array([timestamp, np.real(snr_timeseries_dict[det]), np.imag(snr_timeseries_dict[det])]).T 
    np.savetxt('data/snr_data/snr_'+det, snr_to_save)

# Save event info
# trigger_time, ndet,    detname1, ..., detnameN,    max_snr1, ..., max_snrN,    sigma1, ..., sigmaN
# 1+1+N+N+N = 3N+2 elements
event_info = np.array([trigger_time, ndet])

# https://lscsoft.docs.ligo.org/lalsuite/lal/_l_a_l_detectors_8h_source.html#l00168
lal_det_code = {'L1': 6, 'H1': 5, 'V1':2, 'K1': 14, 'I1': 15}  # Are "K1", "I1" the right names for SPIIR?
det_code_array = np.array([])
for det in det_names:
    event_info = np.append(event_info, lal_det_code[det])
event_info = np.append(event_info, max_snr_array)
event_info = np.append(event_info, sigma_array)
np.savetxt('data/event_info', event_info)


print('Trigger time: ',trigger_time)
print('Detectors: ', det_names)
print('SNRs: ', max_snr_array)
print("sigmas: ", sigma_array)
print('SNR and event info have been saved. \n')