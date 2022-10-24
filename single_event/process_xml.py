# source /fred/oz016/qian/spiirenv/bin/activate

# Targets:
# 1. Read .xml file
# 2. Store SNR, sigma as txt files

# The process may be faster if we do the above in C code. 

from pathlib import Path

import click
import numpy as np
import spiir  # quite slow to import?


@click.command
@click.argument("xml", type=str) # default="data/coinc_xml/H1L1V1_1187008882_3_806.xml"
@click.option("--out", type=str, default="data", help="Output data directory")
def main(xml: str, out: str = "data"):
    print(f'Processing coinc.xml file from {xml}...')
    xmlfile = spiir.io.ligolw.coinc.load_coinc_xml(xml)
    
    # specify folder to save output data
    out_path = Path(out)
    out_path.mkdir(exist_ok=True, parents=True)
    (out_path / "snr_data").mkdir(exist_ok=True)  # make output subfolder for SNR data

    try:
        det_names = list(xmlfile['snrs'].keys())
    except KeyError as err:
        raise KeyError(
            f"snr array data not present {xml} file. Please check your coinc.xml!"
        ) from err
    
    ndet = len(det_names)

    # Calculate sigma = sqrt((h|h)) = deff/SNR
    deff_array = np.array([])
    max_snr_array = np.array([])
    for det in det_names:
        deff_array = np.append(deff_array, xmlfile['tables']['postcoh']['deff_'+det])
        max_snr_array = np.append(max_snr_array, xmlfile['tables']['postcoh']['snglsnr_'+det])
    sigma_array = deff_array*max_snr_array

    trigger_time = xmlfile["tables"]["postcoh"]["end_time"].item()
    trigger_time += xmlfile["tables"]["postcoh"]["end_time_ns"].item() * 1e-9

    # Is trigger time stored in xml file?
    timestamps = {det: xmlfile["snrs"][det].index.values for det in det_names}
    # timestamp = xmlfile['snrs'][det].index.to_numpy()
    # snr_timeseries_dict = xmlfile['snrs']
    # netsnr_timeseries = sum([abs(snr_timeseries_dict[det]) ** 2 for det in det_names])
    
    # trigger_time = timestamp[np.argmax(netsnr_timeseries)]
    #trigger_time = timestamp[np.argmax(abs(snr_timeseries_dict['L1']))]

    # Save event info
    # trigger_time, ndet,    detcode1, ..., detcodeN,    max_snr1, ..., max_snrN,    sigma1, ..., sigmaN
    # 1+1+N+N+N = 3N+2 elements
    event_info = np.array([trigger_time, ndet])

    # Here detector code is for old version of LAL(6.49.0), see:
    # /fred/oz016/opt-pipe/include/lal/LALDetectors.h
    # Be careful when chenge to new version - the code is different, see:
    # https://lscsoft.docs.ligo.org/lalsuite/lal/_l_a_l_detectors_8h_source.html#l00168
    #lal_det_code = {'L1': 5, 'H1': 4, 'V1':1, 'K1': 16, 'I1': 17}  # old lal
    lal_det_code = {'L1': 6, 'H1': 5, 'V1':2, 'K1': 14, 'I1': 15}  # new lal
    det_code_array = np.array([])
    for det in det_names:
        event_info = np.append(event_info, lal_det_code[det])
    event_info = np.append(event_info, max_snr_array)
    event_info = np.append(event_info, sigma_array)
    np.savetxt(out_path / 'event_info', event_info)


    # Save SNR
    for det in det_names:
        snr_to_save = np.array([timestamps[det], np.real(xmlfile['snrs'][det]), 1*np.imag(xmlfile['snrs'][det])]).T 
        np.savetxt(out_path / "snr_data" / f"snr_det{lal_det_code[det]}", snr_to_save)


    print(f'Trigger time: {trigger_time}')
    print(f'Detectors: {det_names}')
    print(f'SNRs: {max_snr_array}')
    print(f'sigmas: {sigma_array}')
    print('SNR and event info have been saved. \n')

if __name__ == "__main__":
    main()