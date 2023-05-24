import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.optimize import leastsq


# fitting functions
def select_aij_according_to_snr(file, low, high):

    # select
    selected_a11 = file[np.where((file[:, 0] >= low) * (file[:, 0] < high))][:, 1]
    selected_a12 = file[np.where((file[:, 0] >= low) * (file[:, 0] < high))][:, 2]
    selected_a21 = file[np.where((file[:, 0] >= low) * (file[:, 0] < high))][:, 3]
    selected_a22 = file[np.where((file[:, 0] >= low) * (file[:, 0] < high))][:, 4]

    # put aij together
    alist = np.array([])
    alist = np.append(alist, selected_a11)
    alist = np.append(alist, selected_a12)
    alist = np.append(alist, selected_a21)
    alist = np.append(alist, selected_a22)
    return np.array(alist)
    # return np.array([selected_a11, selected_a12, selected_a21, selected_a22])


def normalizedgaussian(x, mu, sigma):
    # Normalized Gaussian PDF
    return np.exp(-((x - mu) ** 2) / 2 / sigma**2) / np.sqrt(2 * np.pi) / sigma


def initial_estimate(snr, source_type):
    # design noise
    mu = 0.00029915 * snr - 0.0001853
    # sigma = 0.0001759*snr + 3.75904e-05
    sigma = mu / np.sqrt(2.71828 * np.log(2))

    if source_type == 'BBH':
        mu = mu / 10.0
        sigma = sigma / 10.0
    return mu, sigma


def error_v1new(paras, x, n_i):
    """
    paras = [mu,sigma]
    """
    mu, sigma = paras
    ff = (normalizedgaussian(x, mu, sigma) + normalizedgaussian(x, -mu, sigma)) / 2
    return ff - n_i


def bins_to_center_values(bins):
    return np.array([((bins[i] + bins[i + 1]) / 2) for i in range(0, len(bins) - 1)])


def ls_fit_bi(snr, samples, source_type):  # fit 2 Gaussian prior
    # TODO: histograms should be computed with numpy instead of matplotlib
    n, bins, patches = plt.hist(samples, bins="auto", density=True)
    x = bins_to_center_values(bins)
    mu_init, sigma_init = initial_estimate(snr, source_type)
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


def fitting_abcd(simulation_result, snr_steps, source_type="BNS"):
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

        paras_fit = ls_fit_bi(snr_step, Aijs, source_type)
        plt.clf()
        mu_list.append(abs(paras_fit[0]))
        sigma_list.append(paras_fit[1])

    a, b = np.polyfit(snr_steps, mu_list, 1)
    c, d = np.polyfit(snr_steps, sigma_list, 1)

    return a, b, c, d, mu_list, sigma_list


def linear_fitting_plot(
    snr_steps,
    mu_list,
    sigma_list,
    a,
    b,
    c,
    d,
    save_filename=None,
) -> Figure:
    labelsize = 22
    ticksize = 18
    legendsize = "large"

    fig, ax = plt.subplots(figsize=(10, 7.5))

    # plt.text(7.8, 9.5, r'x 10$^{-3}$',fontsize=16)
    # ax.set_yticks(size=ticksize)
    # ax.set_xticks(size=ticksize)
    plt.yticks(size=ticksize)
    plt.xticks(size=ticksize)
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

    if save_filename:
        fig.savefig(save_filename)
    return fig


def bimodal_fitting_plot(
    result, a, b, c, d, test_snr_low, test_snr_high, save_filename=None
) -> Figure:
    labelsize = 18
    ticksize = 16
    legendsize = 'x-large'

    A_max = np.percentile(abs(result[:, 1]), 99)
    A_range = np.linspace(-1.1 * A_max, 1.1 * A_max, 200)
    color_bar = 'cornflowerblue'
    color_line = 'red'

    nrows = int(np.ceil(len(test_snr_low) / 2))
    ncols = 2
    figsize = (10, 5 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i in range(len(test_snr_low)):
        ax = axes[i // ncols, i % ncols]
        # ax.set_yticks(size=ticksize)
        # ax.set_xticks(size=ticksize)
        plt.yticks(size=ticksize)
        plt.xticks(size=ticksize)
        snr_low = test_snr_low[i]
        snr_high = test_snr_high[i]
        snr_middle = (snr_high + snr_low) / 2
        mu = a * snr_middle + b
        sigma = c * snr_middle + b
        theo_pdf = (
            normalizedgaussian(A_range, mu, sigma)
            + normalizedgaussian(A_range, -mu, sigma)
        ) / 2
        ax.hist(
            select_aij_according_to_snr(result, snr_low, snr_high),
            bins='auto',
            density=True,
            label=f"SNR {snr_low}-{snr_high}",
            color=color_bar,
        )
        ax.plot(A_range, theo_pdf, color=color_line)
        ax.set_ylabel("Probability density", size=labelsize)
        ax.set_xlim(-1.1 * A_max, 1.1 * A_max)
        ax.legend(loc="best", fontsize=legendsize)

    if save_filename:
        fig.savefig(save_filename)
    return fig


def find_horizon(ifos, waveform_generator, example_injection_parameter):
    example_injection_parameter['luminosity_distance'] = 1
    h_dict = waveform_generator.frequency_domain_strain(example_injection_parameter)

    netsnrsq = 0
    for det in ifos:
        # signal = det.get_detector_response(h_dict, example_injection_parameter)
        netsnrsq += det.optimal_snr_squared(h_dict['plus'])

    netsnr = np.real(netsnrsq) ** 0.5

    # set snr=8 as detection threshold
    return netsnr / 8


def save_configs(source_types, seallist, outputfilename):
    if len(source_types) != len(seallist):
        raise Exception('len(source_types) != len(seallist).')

    config_dict_dict = dict()
    for i in range(len(source_types)):
        source_type = source_types[i]
        tempseal = seallist[i]
        config_dict = {
            'description': tempseal.description,
            'a': tempseal.prior_coef_a,
            'b': tempseal.prior_coef_b,
            'c': tempseal.prior_coef_c,
            'd': tempseal.prior_coef_d,
        }
        config_dict_dict[source_type] = config_dict

    with open(outputfilename, 'w') as file:
        file.write(json.dumps(config_dict_dict))
