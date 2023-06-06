from pathlib import Path
from typing import Optional, Sequence, Union

import click
import numpy as np

import sealgw
import sealgw.simulation as sealsim
from sealgw import seal


def ifos2detnamelist(ifostr):
    return ifostr.split('_')

def make_fitting_plots(low_snr_cutoff, high_snr_cutoff, fitting_samples, source_type, seal_label, label, output_dir):
    snr_steps = np.arange(low_snr_cutoff, high_snr_cutoff, 2)
    a, b, c, d, mu_list, sigma_list = sealsim.prior_fitting.fitting_abcd(
        fitting_samples, snr_steps, source_type
    )
    linear_fitting_plot_filename = f'{output_dir}/linear_fitting_{seal_label}_{label}.png'
    sealsim.prior_fitting.linear_fitting_plot(
        snr_steps, mu_list, sigma_list, a, b, c, d, linear_fitting_plot_filename
    )
    bimodal_fitting_plot_filename = f'{output_dir}/bimodal_fitting_{seal_label}_{label}.png'
    sealsim.prior_fitting.bimodal_fitting_plot(
        fitting_samples,
        a,
        b,
        c,
        d,
        [10, 15, 20, 25],
        [15, 20, 25, 30],
        save_filename=bimodal_fitting_plot_filename,
    )



@click.command()
@click.argument('ifostr')
@click.argument('psdpath')
@click.argument('output-dir')
@click.argument('label')
@click.option('--low-snr-cutoff', default=9, prompt='Lower SNR cutoff: ')
@click.option('--high-snr-cutoff', default=35, prompt='Higher SNR cutoff: ')
@click.option('--nsample', default=30000, prompt='Number of samples: ')
@click.option(
    '--ncpu', default=1, prompt='ncpu: ', help='Number of CPUs for simulation.'
)
def main(
    ifostr: str,
    psdpath: str,
    output_dir: str,
    label: str,
    low_snr_cutoff: float,
    high_snr_cutoff: float,
    nsample: int,
    ncpu: int,
):
    det_name_list = ifos2detnamelist(ifostr)

    seallist = []
    seal_label = []

    premerger_time_start_list = np.arange(2,61,2)[::-1] * 60.
    premerger_time_end_list = np.arange(0,59,2)[::-1] * 60.
    #sampling_frequency_list = np.array([64,64,64,64,64,64,256,512,2048,4096])
    sampling_frequency_list = np.zeros(len(premerger_time_start_list)) + 2048

    for i in range(len(premerger_time_start_list)):
        seal2train = seal.SealBNSEW()

        premerger_time_start = premerger_time_start_list[i]
        premerger_time_end = premerger_time_end_list[i]
        seal2train.premerger_time_start = premerger_time_start
        seal2train.premerger_time_end = premerger_time_end
        sampling_frequency = sampling_frequency_list[i]

        seallabel = f'BNS_EW_{int(premerger_time_start)}to{int(premerger_time_end)}'
        print(f"\nTraining {seallabel} seal model...")
        
        fitting_result, _ = seal2train.fitting_mu_sigma_snr_relation(Nsample=nsample, 
                                                                        det_name_list=det_name_list,
                                                                        source_type='BNS_EW_FD',
                                                                        ncpu=ncpu,
                                                                        use_bilby_psd=True,
                                                                        #dmax=horizons[i]/3, #horizons[i]/3,
                                                                        fmin=5,
                                                                        duration=120,
                                                                        sampling_frequency=sampling_frequency,
                                                                        low_snr_cutoff=low_snr_cutoff,
                                                                        high_snr_cutoff=high_snr_cutoff,
                                                                        )
        make_fitting_plots(low_snr_cutoff, high_snr_cutoff, fitting_result, 'BNS_EW_FD', seallabel, label, output_dir)
        seal2train.description = f"Config file for BNS EW {int(premerger_time_start)} to {int(premerger_time_end)}s {label}"
        seallist.append(seal2train)
        seal_label.append(seallabel)

    # save
    config_filename = f'{output_dir}/config_{label}.json'
    sealsim.prior_fitting.save_configs(seal_label, seallist, config_filename)


if __name__ == '__main__':
    main()
