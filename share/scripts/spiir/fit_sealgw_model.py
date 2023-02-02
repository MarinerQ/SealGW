from pathlib import Path
from typing import Optional, Sequence, Union

import click
import numpy as np

import sealgw
import sealgw.simulation as sealsim
from sealgw import seal


def ifos2detnamelist(ifostr):
    det_name_list = []
    for name in ifostr:
        det_name_list.append(name + '1')
    return det_name_list


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
    source_types = ["BNS", "BBH", "NSBH"]
    seallist = []

    for source_type in source_types:
        print(f"\nTraining {source_type} seal model...")
        seal2train = seal.Seal()

        # fitting
        fitting_samples = seal2train.fitting_mu_sigma_snr_relation(
            Nsample=nsample,
            det_name_list=det_name_list,
            source_type=source_type,
            ncpu=ncpu,
            use_bilby_psd=False,
            custom_psd_path=psdpath,
            low_snr_cutoff=low_snr_cutoff,
            high_snr_cutoff=high_snr_cutoff,
        )
        
        # make fitting plots
        snr_steps = np.arange(low_snr_cutoff, high_snr_cutoff, 2)
        a, b, c, d, mu_list, sigma_list = sealsim.prior_fitting.fitting_abcd(
            fitting_samples, snr_steps, source_type
        )
        linear_fitting_plot_filename = f'{output_dir}/linear_fitting_{source_type}_{label}.png'
        sealsim.prior_fitting.linear_fitting_plot(
            snr_steps, mu_list, sigma_list, a, b, c, d, linear_fitting_plot_filename
        )
        bimodal_fitting_plot_filename = f'{output_dir}/bimodal_fitting_{source_type}_{label}.png'
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

        seal2train.description = f"Config file for {source_type} {label}"
        seallist.append(seal2train)

    # save
    config_filename = f'{output_dir}/config_{label}.json'
    sealsim.prior_fitting.save_configs(source_types, label, seallist, config_filename)


if __name__ == '__main__':
    main()
