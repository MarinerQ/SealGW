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

    ######### Train BNS #########
    seallabel = 'BNS'
    print(f"Training {seallabel} seal model...")
    seal2train = seal.Seal()
    fitting_samples, _ = seal2train.fitting_mu_sigma_snr_relation(
        Nsample=nsample,
        det_name_list=det_name_list,
        source_type='BNS',
        ncpu=ncpu,
        use_bilby_psd=False,
        duration=100,
        custom_psd_path=psdpath,
        low_snr_cutoff=low_snr_cutoff,
        high_snr_cutoff=high_snr_cutoff,
    )
    make_fitting_plots(low_snr_cutoff, high_snr_cutoff, fitting_samples, 'BNS', seallabel, label, output_dir)
    seal2train.description = f"Config file for BNS {label}"
    seallist.append(seal2train)
    seal_label.append(seallabel)

    ######### Train BNS EW 10s and 30s #########
    ew_times = [10, 30]
    for ew_time in ew_times:
        seallabel = f'BNS_EW_{int(ew_time)}'
        print(f"\nTraining {seallabel} seal model...")
        seal2train = seal.SealBNSEW()
        seal2train.premerger_time_start = 100
        seal2train.premerger_time_end = ew_time
        fitting_samples, _ = seal2train.fitting_mu_sigma_snr_relation(
            Nsample=nsample,
            det_name_list=det_name_list,
            source_type='BNS_EW_FD',
            ncpu=ncpu,
            use_bilby_psd=False,
            duration=100-ew_time,
            custom_psd_path=psdpath,
            low_snr_cutoff=low_snr_cutoff,
            high_snr_cutoff=high_snr_cutoff,
        )
        make_fitting_plots(low_snr_cutoff, high_snr_cutoff, fitting_samples, 'BNS', seallabel, label, output_dir)
        seal2train.description = f"Config file for BNS EW {ew_time}s {label}"
        seallist.append(seal2train)
        seal_label.append(seallabel)

    '''
    ######### Train BBH foe different fixed chirp masses #########
    bbh_fixed_mc_list = [10,20,30,40,50,60,70]
    for bbh_fixed_mc in bbh_fixed_mc_list:
        seallabel = f'BBH_Mc_{bbh_fixed_mc}'
        print(f"Training {seallabel} seal model...")
        seal2train = seal.Seal()
        fitting_samples, _ = seal2train.fitting_mu_sigma_snr_relation(
            Nsample=nsample,
            det_name_list=det_name_list,
            source_type='BBH',
            ncpu=ncpu,
            use_bilby_psd=False,
            custom_psd_path=psdpath,
            low_snr_cutoff=low_snr_cutoff,
            high_snr_cutoff=high_snr_cutoff,
            fixed_mc=bbh_fixed_mc,
        )
        make_fitting_plots(low_snr_cutoff, high_snr_cutoff, fitting_samples, 'BBH', seallabel, label, output_dir)
        seal2train.description = f"Config file for BBH with Mc fixed at {bbh_fixed_mc} {label}"
        seallist.append(seal2train)
        seal_label.append(seallabel)
    '''


    ######### Train BBH #########
    seallabel='BBH'
    print(f"Training {seallabel} seal model...")
    seal2train = seal.Seal()
    fitting_samples, _ = seal2train.fitting_mu_sigma_snr_relation(
        Nsample=nsample,
        det_name_list=det_name_list,
        source_type='BBH',
        ncpu=ncpu,
        use_bilby_psd=False,
        custom_psd_path=psdpath,
        low_snr_cutoff=low_snr_cutoff,
        high_snr_cutoff=high_snr_cutoff,
    )
    make_fitting_plots(low_snr_cutoff, high_snr_cutoff, fitting_samples, 'BBH', seallabel, label, output_dir)
    seal2train.description = f"Config file for BBH {label}"
    seallist.append(seal2train)
    seal_label.append(seallabel)

    ######### Train NSBH #########
    seallabel='NSBH'
    print(f"Training {seallabel} seal model...")
    seal2train = seal.Seal()
    fitting_samples, _ = seal2train.fitting_mu_sigma_snr_relation(
        Nsample=nsample,
        det_name_list=det_name_list,
        source_type='NSBH',
        ncpu=ncpu,
        use_bilby_psd=False,
        custom_psd_path=psdpath,
        low_snr_cutoff=low_snr_cutoff,
        high_snr_cutoff=high_snr_cutoff,
    )
    make_fitting_plots(low_snr_cutoff, high_snr_cutoff, fitting_samples, 'NSBH', seallabel, label, output_dir)
    seal2train.description = f"Config file for NSBH {label}"
    seallist.append(seal2train)
    seal_label.append(seallabel)

    # save
    config_filename = f'{output_dir}/config_{label}.json'
    sealsim.prior_fitting.save_configs(seal_label, seallist, config_filename)


if __name__ == '__main__':
    main()
