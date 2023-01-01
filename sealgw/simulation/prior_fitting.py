import matplotlib.pyplot as plt
import numpy as np
from bilby.gw import WaveformGenerator
from bilby.gw.conversion import (
    convert_to_lal_binary_black_hole_parameters,
    convert_to_lal_binary_neutron_star_parameters,
)
from bilby.gw.source import lal_binary_black_hole, lal_binary_neutron_star
from scipy.optimize import leastsq

from .generating_data import (
    generate_random_injection_parameters,
    zip_injection_parameters,
)


# fitting functions
def select_aij_according_to_snr(file, low, high):

    # select
    selected_a11 = file[np.where((file[:, 0] >= low) * (file[:, 1] < high))][:, 1]
    selected_a12 = file[np.where((file[:, 0] >= low) * (file[:, 1] < high))][:, 2]
    selected_a21 = file[np.where((file[:, 0] >= low) * (file[:, 1] < high))][:, 3]
    selected_a22 = file[np.where((file[:, 0] >= low) * (file[:, 1] < high))][:, 4]

    # put aij together
    return np.array([selected_a11, selected_a12, selected_a21, selected_a22])


def f(x, mu, sigma):
    # Normalized Gaussian PDF
    return np.exp(-((x - mu) ** 2) / 2 / sigma**2) / np.sqrt(2 * np.pi) / sigma


def initial_estimate(snr):
    # design noise
    mu = 0.00029915 * snr - 0.0001853
    # sigma = 0.0001759*snr + 3.75904e-05
    sigma = mu / np.sqrt(2.71828 * np.log(2))

    return mu, sigma


def error_v1new(paras, x, n_i):
    """
    paras = [mu,sigma]
    """
    mu, sigma = paras
    ff = (f(x, mu, sigma) + f(x, -mu, sigma)) / 2
    return ff - n_i


def bins_to_center_values(bins):
    return np.array([((bins[i] + bins[i + 1]) / 2) for i in range(0, len(bins) - 1)])


def ls_fit_bi(snr, samples):  # fit 2 Gaussian prior
    # TODO: histograms should be computed with numpy instead of matplotlib
    n, bins, patches = plt.hist(samples, bins="auto", density=True)
    x = bins_to_center_values(bins)
    mu_init, sigma_init = initial_estimate(snr)
    paras0 = [mu_init, sigma_init]
    paras_fitted = leastsq(error_v1new, paras0, args=(x, n))[0]
    return paras_fitted


def para_conversion(d_L, iota, psi, phase):
    cos_iota = np.cos(iota)
    cos_phic = np.cos(phase)
    sin_phic = np.sin(phase)
    cos_2psi = np.cos(2 * psi)
    sin_2psi = np.sin(2 * psi)

    A11 = (
        (1 + cos_iota * cos_iota) / 2 * cos_phic * cos_2psi
        - cos_iota * sin_phic * sin_2psi
    ) / d_L
    A12 = (
        (1 + cos_iota * cos_iota) / 2 * sin_phic * cos_2psi
        + cos_iota * cos_phic * sin_2psi
    ) / d_L
    A21 = (
        -(1 + cos_iota * cos_iota) / 2 * cos_phic * sin_2psi
        - cos_iota * sin_phic * cos_2psi
    ) / d_L
    A22 = (
        -(1 + cos_iota * cos_iota) / 2 * sin_phic * sin_2psi
        + cos_iota * cos_phic * cos_2psi
    ) / d_L

    return A11, A12, A21, A22


def calculate_snr_kernel(sample_ID, samples, ifos, wave_gen, results):
    inj_para = zip_injection_parameters(samples[sample_ID])
    h_dict = wave_gen.frequency_domain_strain(parameters=inj_para)

    net_snr_sq = 0
    for det in ifos:
        signal = det.get_detector_response(h_dict, inj_para)
        net_snr_sq += det.optimal_snr_squared(signal)

    results[sample_ID] = np.sqrt(abs(net_snr_sq))


def fitting_abcd(simulation_result, snr_steps):
    mu_list = []
    sigma_list = []

    n_list = []
    bin_list = []
    for i in range(len(snr_steps)):
        snr_step = snr_steps[i]
        Aijs = select_aij_according_to_snr(
            simulation_result, snr_step - 1, snr_step + 1
        )

        # TODO: histograms should be computed with numpy instead of matplotlib
        n, bins, patches = plt.hist(Aijs, bins="auto", density=True)
        n_list.append(n)
        bin_list.append(bins)

        paras_fit = ls_fit_bi(snr_step, Aijs)
        plt.clf()
        mu_list.append(paras_fit[0])
        sigma_list.append(paras_fit[1])

    a, b = np.polyfit(snr_steps, mu_list, 1)
    c, d = np.polyfit(snr_steps, sigma_list, 1)

    return a, b, c, d, mu_list, sigma_list


def get_wave_gen(source_type, fmin, duration, sampling_frequency):

    if source_type == "BNS":
        waveform_arguments = dict(
            waveform_approximant="TaylorF2",
            reference_frequency=50.0,
            minimum_frequency=fmin,
        )
        waveform_generator = WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=lal_binary_neutron_star,
            parameter_conversion=convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=waveform_arguments,
        )

    elif source_type == "BBH":
        waveform_arguments = dict(
            waveform_approximant="IMRPhenomPv2",
            reference_frequency=50.0,
            minimum_frequency=fmin,
        )
        waveform_generator = WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=lal_binary_black_hole,
            parameter_conversion=convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments,
        )

    elif source_type == "NSBH":
        waveform_arguments = dict(
            waveform_approximant="IMRPhenomNSBH",
            reference_frequency=50.0,
            minimum_frequency=fmin,
        )
        waveform_generator = WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=lal_binary_neutron_star,
            waveform_arguments=waveform_arguments,
        )

    else:
        raise ValueError(f"Invalid source {source_type}! Must be one of BBH, BNS, NSBH")

    return waveform_generator


def get_fitting_source_para_sample(source_type, Nsample):
    if source_type == "BNS":
        samples = generate_random_injection_parameters(
            Nsample=Nsample,
            dmin=0,
            dmax=200,
            m1_low=1.1,
            m1_high=2,
            q_low=0.8,
            a_max=0.1,
            m2_low=1.1,
        )

    elif source_type == "BBH":
        samples = generate_random_injection_parameters(
            Nsample=Nsample,
            dmin=0,
            dmax=4000,
            m1_low=6,
            m1_high=90,
            q_low=0.25,
            a_max=0.1,
            m2_low=6,
        )

    elif source_type == "NSBH":
        samples = generate_random_injection_parameters(
            Nsample=Nsample,
            dmin=0,
            dmax=500,
            m1_low=6,
            m1_high=90,
            q_low=0.1,
            a_max=0.1,
            m2_low=1.1,
        )

    else:
        raise ValueError(f"Invalid source {source_type}! Must be one of BBH, BNS, NSBH")

    return samples


def linear_fitting_plot(snr_steps, mu_list, sigma_list, a, b, c, d, save_filename):
    labelsize = 22
    ticksize = 18
    legendsize = "large"

    fig, ax = plt.subplots(figsize=(10, 7.5))

    # plt.text(7.8, 9.5, r'x 10$^{-3}$',fontsize=16)
    ax.set_yticks(size=ticksize)
    ax.set_xticks(size=ticksize)
    ax.scatter(
        snr_steps,
        1e3 * np.array(mu_list),
        marker="x",
        color="orangered",
        label=r"$\mu$",
    )
    ax.plot(
        snr_steps,
        1e3 * (a * snr_steps + b),
        color="royalblue",
        label=r"Linear fitting of $\mu$",
    )
    ax.scatter(
        snr_steps,
        1e3 * np.array(sigma_list),
        marker="o",
        color="darkorange",
        label=r"$\sigma$",
    )
    ax.plot(
        snr_steps,
        1e3 * (c * snr_steps + d),
        color="forestgreen",
        label=r"Linear fitting of $\sigma$",
    )
    ax.set_xlabel("Signal-to-noise Ratio", size=labelsize)
    ax.set_ylabel(r"$\mu, \sigma \times$ 1000", size=labelsize)
    ax.legend(loc="best", ncol=2, fontsize=legendsize)
    ax.grid()

    fig.savefig(save_filename)


def bimodal_fitting_plot(result, a, b, c, d, save_filename):
    labelsize = 18
    ticksize = 16
    legendsize = "x-large"

    A_range = np.linspace(-0.04, 0.04, 200)
    color_bar = "cornflowerblue"
    color_line = "red"
    test_snr_low = [12, 16, 20, 24]
    test_snr_high = [16, 20, 24, 30]

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 10))
    for i in range(4):
        ax = axes[i // ncols, i % ncols]
        ax.set_yticks(size=ticksize)
        ax.set_xticks(size=ticksize)
        snr_low = test_snr_low[i - 1]
        snr_high = test_snr_high[i - 1]
        snr_middle = (snr_high + snr_low) / 2
        mu = a * snr_middle + b
        sigma = c * snr_middle + b
        theo_pdf = (f(A_range, mu, sigma) + f(A_range, -mu, sigma)) / 2
        ax.hist(
            select_aij_according_to_snr(result, snr_low, snr_high),
            bins="auto",
            density=True,
            label=f"SNR {snr_low}-{snr_high}",
            color=color_bar,
        )
        ax.plot(A_range, theo_pdf, color=color_line)
        ax.set_ylabel("Probability density", size=labelsize)
        ax.set_xlim(-0.05, 0.05)
        ax.legend(loc="best", fontsize=legendsize)

    fig.savefig(save_filename)
