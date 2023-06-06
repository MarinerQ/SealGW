import ctypes
import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import astropy_healpix as ah
import healpy as hp
import ligo.skymap.plot
import numpy as np
import scipy
import sealcore
from astropy import units as u
from astropy.coordinates import SkyCoord
from ligo.skymap import postprocess
from matplotlib import figure as Figure
from matplotlib import pyplot as plt
import time
from pycbc.waveform import get_fd_waveform
import bilby
from gstlal import chirptime

# export OMP_NUM_THREADS=8
LAL_DET_MAP = dict(L1=6, H1=5, V1=2, K1=14, I1=15, CE=5, CEL=6, ET1=16, ET2=17, ET3=18)


def cythontestfunc(ra, dec, gpstime, detcode):
    return sealcore.pytest1(ra, dec, gpstime, detcode)


'''
def lal_et_response_function(ra, dec, gpstime, psi, det_name, mode):
    mode201 = {'plus': 0, 'cross': 1}
    #name2code = {'ET1': 16, 'ET2': 17, 'ET3': 18}

    mode_code = mode201[mode]
    #det_code = name2code[det_name]
    det_code = LAL_DET_MAP[det_name]

    return sealcore.Pylal_resp_func(ra, dec, gpstime, psi, det_code, mode_code)


def lal_ce_response_function(ra, dec, gpstime, psi, mode):
    mode201 = {'plus': 0, 'cross': 1}
    mode_code = mode201[mode]

    det_code = 10

    return sealcore.Pylal_resp_func(ra, dec, gpstime, psi, det_code, mode_code)
'''


def lal_response_function(ra, dec, gpstime, psi, det_name, mode):
    mode201 = {'plus': 0, 'cross': 1}
    # name2code = {'ET1': 16, 'ET2': 17, 'ET3': 18}

    mode_code = mode201[mode]
    # det_code = name2code[det_name]
    det_code = LAL_DET_MAP[det_name]

    return sealcore.Pylal_resp_func(ra, dec, gpstime, psi, det_code, mode_code)


def lal_dt_function(ra, dec, gpstime, det_name):
    det_code = LAL_DET_MAP[det_name]

    return sealcore.Pylal_dt_func(ra, dec, gpstime, det_code)


logger = logging.getLogger(__name__)


def read_event_info(filepath):
    event_info = np.loadtxt(filepath)
    trigger_time = event_info[0]
    ndet = event_info[1]
    det_codes = np.int64(event_info[2:5])
    snrs = event_info[5:8]
    sigmas = event_info[8:]

    return trigger_time, ndet, det_codes, snrs, sigmas


def extract_info_from_xml(
    filepath, return_names=False, use_timediff=True, recalculate_sigmas=True
):
    import spiir.io.ligolw

    xmlfile = spiir.io.ligolw.load_coinc_xml(filepath)

    try:
        det_names = list(xmlfile["snrs"].keys())
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

    postcoh = xmlfile["tables"]["postcoh"]
    trigger_time = 0
    for det in det_names:
        deff_array = np.append(deff_array, postcoh[f"deff_{det}"])
        max_snr_array = np.append(max_snr_array, postcoh[f"snglsnr_{det}"])
        snr_array = np.append(snr_array, xmlfile["snrs"][det])
        time_array = np.append(time_array, xmlfile["snrs"][det].index.values)
        ntimes_array = np.append(ntimes_array, len(xmlfile["snrs"][det]))
        det_code_array = np.append(det_code_array, int(LAL_DET_MAP[det]))

        trigger_time += postcoh[f'end_time_sngl_{det}'].item()
        trigger_time += postcoh[f'end_time_ns_sngl_{det}'].item() * 1e-9

    sigma_array = deff_array * max_snr_array
    if recalculate_sigmas:
        sigma_dict = calculate_template_norms(
            postcoh['mass1'][0],
            postcoh['mass2'][0],
            postcoh['spin1z'][0],
            postcoh['spin2z'][0],
            postcoh['template_duration'][0],
            xmlfile['psds'],
            postcoh['f_final'][0],
        )

        # key sequence in xmlfile['psds'] may be different from det_names
        for i, det in enumerate(det_names):
            sigma_array[i] = sigma_dict[det]

    trigger_time = trigger_time / len(det_names)  # mean of trigger times of each det
    if use_timediff:
        trigger_time = time_array[np.argmax(abs(snr_array))]
    logger.debug("xml processing done. ")
    logger.debug(f"Trigger time: {trigger_time}")
    logger.debug(f"Detectors: {det_names}")
    logger.debug(f"SNRs: {max_snr_array}")
    logger.debug(f"sigmas: {sigma_array}")

    if return_names:
        return (
            trigger_time,
            ndet,
            ntimes_array.astype(ctypes.c_int32),
            det_names,
            max_snr_array,
            sigma_array,
            time_array,
            snr_array,
        )
    else:
        return (
            trigger_time,
            ndet,
            ntimes_array.astype(ctypes.c_int32),
            det_code_array.astype(ctypes.c_int32),
            max_snr_array,
            sigma_array,
            time_array,
            snr_array,
        )


def calculate_template_norms(m1, m2, a1, a2, duration, psd_dict, f_final=1024):
    sigma_dict = {}

    mc = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
    if mc > 1.73:
        approximant = 'IMRPhenomD'
    else:
        approximant = 'TaylorF2'  # TaylorF2
    for detname, psdarray in psd_dict.items():
        if len(sigma_dict) == 0:
            delta_f = (
                1 / duration
            )  # psd_dict[detname].index[1]-psd_dict[detname].index[0]
            f_final = min(psd_dict[detname].index[-1], f_final)
            # remove biased PSD from 1000Hz in SPIIR trigger
            if f_final > 972:
                f_final = 972
            hp, hc = get_fd_waveform(
                approximant=approximant,  # SEOBNRv4_ROM TaylorF2
                mass1=m1,
                mass2=m2,
                distance=1,
                inclination=0,
                coa_phase=0,
                spin1x=0,
                spin1y=0,
                spin1z=a1,
                spin2x=0,
                spin2y=0,
                spin2z=a2,
                delta_f=delta_f,
                f_lower=20,
                f_final=f_final,
            )

        hp_farray = hp.sample_frequencies.numpy()
        mask = hp_farray < f_final
        psd_interp = np.interp(
            hp.sample_frequencies.numpy()[mask],
            psd_dict[detname].index,
            psd_dict[detname].values,
        )
        sigma = bilby.gw.utils.noise_weighted_inner_product(
            hp.numpy()[mask], hp.numpy()[mask], psd_interp, 1 / delta_f
        )
        sigma = np.sqrt(np.real(sigma))
        sigma_dict[detname] = sigma

    return sigma_dict


def generate_healpix_grids(nside):
    npix = hp.nside2npix(nside)  # equiv to 12*nside^2
    theta, phi = hp.pixelfunc.pix2ang(nside, np.arange(npix), nest=True)
    ra = phi
    dec = -theta + np.pi / 2

    return ra, dec


'''
def calculate_template_norm(mass_1, mass_2, ):
    sigmas = []

    mass_1
    hp, hc = get_fd_waveform(
        approximant='TaylorF2',
        mass1=mass_1,
        mass2=mass_2,
        distance=1,
        inclination=0,
        coa_phase=0,
        lambda1=lambda_1,
        lambda2=lambda_2,
        spin1x=0,
        spin1y=0,
        spin1z=a_1,
        spin2x=0,
        spin2y=0,
        spin2z=a_2,
        delta_f=deltaf,
        f_lower=20,
        f_final=1024,
        f_ref=50.0,
    )

    return np.array(sigmas)
'''


def deg2perpix(nlevel):
    """
    nside_base:  16
    nlevel = 0, nside = 16, npix = 3072, deg2 per pixel = 13.4287109375
    nlevel = 1, nside = 32, npix = 12288, deg2 per pixel = 3.357177734375
    nlevel = 2, nside = 64, npix = 49152, deg2 per pixel = 0.83929443359375
    nlevel = 3, nside = 128, npix = 196608, deg2 per pixel = 0.2098236083984375
    nlevel = 4, nside = 256, npix = 786432, deg2 per pixel = 0.052455902099609375
    nlevel = 5, nside = 512, npix = 3145728, deg2 per pixel = 0.013113975524902344
    nlevel = 6, nside = 1024, npix = 12582912, deg2 per pixel = 0.003278493881225586
    nlevel = 7, nside = 2048, npix = 50331648, deg2 per pixel = 0.0008196226755777994
    """
    nside_base = 16
    nside = nside_base * 2**nlevel
    npix = 12 * nside**2
    deg2perpix = 41252.96 / npix

    return deg2perpix


def normalize_log_probabilities(log_probs: np.ndarray) -> np.ndarray:
    """Converts log probabilities into a normalized probability array."""
    max_log_prob = np.max(log_probs)
    np.subtract(log_probs, max_log_prob, out=log_probs)
    np.exp(log_probs, out=log_probs)
    sum_probs = np.sum(log_probs)
    np.divide(log_probs, sum_probs, out=log_probs)
    return log_probs


def seal_with_adaptive_healpix(
    nlevel,
    time_arrays,
    snr_arrays,
    det_code_array,
    sigma_array,
    ntimes_array,
    ndet,
    start_time,
    end_time,
    interp_factor,
    prior_mu,
    prior_sigma,
    nthread,
    max_snr_det_id,
    interp_order=0,
    use_timediff=True,
    prior_type=0,
    premerger_time=0,
):

    # Healpix: The Astrophysical Journal, 622:759–771, 2005. See its Figs. 3 & 4.
    # Adaptive healpix: see Bayestar paper (and our seal paper).

    nside_base = 16
    nside_final = nside_base * 2**nlevel  # 16 *  [1,4,16,...,2^6]
    npix_final = 12 * nside_final**2  # 12582912 for nlevel=6

    # ra, dec = generate_healpix_grids(nside_final)
    skymap_multires = np.zeros(npix_final)

    dts = [
        time_arrays[ntimes_array[detid] + 1] - time_arrays[ntimes_array[detid]]
        for detid in range(ndet)
    ]
    ntime_interp = int(interp_factor * (end_time - start_time) / min(dts))
    if use_timediff:
        use_timediff = 1
    else:
        use_timediff = 0
    sealcore.Pycoherent_skymap_multires(
        skymap_multires,
        time_arrays,
        snr_arrays,
        det_code_array,
        sigma_array,
        ntimes_array,
        ndet,
        start_time,
        end_time,
        ntime_interp,
        prior_mu,
        prior_sigma,
        nthread,
        interp_order,
        max_snr_det_id,
        nlevel,
        use_timediff,
        prior_type,
        premerger_time,
    )

    # return normalize_log_probabilities(skymap_multires)
    return skymap_multires


def get_det_code_array(det_name_list):
    """
    Transfer detector name to detector code in LAL, see
    https://lscsoft.docs.ligo.org/lalsuite/lal/_l_a_l_detectors_8h_source.html#l00168

    Note in history version of LAL, the codes may be different.
    """

    return np.array([LAL_DET_MAP[det] for det in det_name_list])


def plot_skymap(
    probs: np.ndarray,
    save_filename: Optional[str] = None,
    true_ra: float = None,
    true_dec: float = None,
    zoomin_truth: bool = False,
    zoomin_maxprob: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 7),
) -> Figure:
    """Plots a localisation skymap using from a probability density array."""
    import spiir.search.skymap

    # add ground truth marker if both ra and dec are not None
    ground_truth = None
    if true_ra is not None and true_dec is not None:
        ground_truth = SkyCoord(true_ra, true_dec, unit="rad")

    inset_kwargs = []
    if zoomin_truth:
        inset_kwargs.append({'center': ground_truth, 'radius': 5})
    if zoomin_maxprob:
        inset_kwargs.append({'center': "max", 'radius': 5})
    if len(inset_kwargs) == 0:
        inset_kwargs = None

    fig = spiir.search.skymap.plot_skymap(
        probs,
        contours=[50, 90],
        ground_truth=ground_truth,
        colorbar=False,
        figsize=figsize,
        inset_kwargs=inset_kwargs,
        title=title,
    )

    if save_filename is not None:
        fig.savefig(save_filename)
        logger.info(f"Skymap saved to {save_filename}")

    return fig


def apply_fudge_factor(probs: np.ndarray, fudge_percent: float) -> np.ndarray:
    """
    Apply a fudge factor to a probability healpix skymap to renormalize the probabilities.
    The fudge factor is determined by experiment to adjust 90% area.

    e.g., if 95% area gets 90% simulations right, the fudge_percent 0.95. The top 95% of prob skymap
    will be multiplied by 90/95 so that it becomes 90% area.

    """
    # Find the top values that summed up to 0.9 in probs
    sorted_probs = np.sort(probs)[::-1]
    cumulative_probs = np.cumsum(sorted_probs)
    index_90 = np.searchsorted(cumulative_probs, fudge_percent)
    top_90_probs_mask = probs >= sorted_probs[index_90]

    # Apply fudge factor
    fudge_factor = 0.9 / fudge_percent
    fudge_facto4norm = 0.1 / (1 - fudge_percent)
    fudge_factored_probs = np.where(
        top_90_probs_mask, probs * fudge_factor, probs * fudge_facto4norm
    )
    fudge_factored_probs /= np.sum(fudge_factored_probs)

    return fudge_factored_probs


def confidence_area(probs, confidence_level):
    """
    skymap: probability skymap
    confidence_level: float or array

    returns confidence area at given confidence lvel.
    """
    nside = ah.npix_to_nside(len(probs))
    deg2perpix = ah.nside_to_pixel_area(nside).to_value(u.deg**2)

    credible_levels = 100 * postprocess.find_greedy_credible_levels(probs)
    area = np.searchsorted(np.sort(credible_levels), confidence_level) * deg2perpix

    return area


def cumulative_percentage(probs, ra, dec):
    """
    Find cumulative percentage at (ra, dec).

    Small number means high prob area because we counts pixels from the highest one.

    When reach the lowest prob pixel, prob accumulate to one.
    """
    theta = np.pi / 2 - dec
    phi = ra

    nside = ah.npix_to_nside(len(probs))  # confirm this is the correct nside
    pixel_index = hp.pixelfunc.ang2pix(nside, theta, phi, nest=True)

    prob_here = probs[pixel_index]
    larger_prob_index = np.where(probs > prob_here)[0]

    percentage = sum(probs[larger_prob_index])

    return percentage


def search_area(probs, ra, dec):
    """
    Find cumulative search area at (ra, dec), search from the highest pixel.

    When reach the lowest prob pixel, prob accumulate to one.
    """
    nside = ah.npix_to_nside(len(probs))
    deg2perpix = ah.nside_to_pixel_area(nside).to_value(u.deg**2)

    theta = np.pi / 2 - dec
    phi = ra
    pixel_index = hp.pixelfunc.ang2pix(nside, theta, phi, nest=True)

    prob_here = probs[pixel_index]
    larger_prob_index = np.where(probs > prob_here)[0]

    area = len(larger_prob_index) * deg2perpix

    return area


def catalog_test_statistics(probs, ra_inj, dec_inj):
    """
    return all statistics that catalog test needs.
    """
    nside = ah.npix_to_nside(len(probs))
    deg2perpix = ah.nside_to_pixel_area(nside).to_value(u.deg**2)

    theta = np.pi / 2 - dec_inj
    phi = ra_inj
    pixel_index = hp.pixelfunc.ang2pix(nside, theta, phi, nest=True)

    prob_here = probs[pixel_index]
    larger_prob_index = np.where(probs > prob_here)[0]

    inj_point_cumulative_percentage = sum(probs[larger_prob_index])
    search_area = len(larger_prob_index) * deg2perpix

    credible_levels = 100 * postprocess.find_greedy_credible_levels(probs)
    levels = [50, 90]
    confidence_areas = np.searchsorted(np.sort(credible_levels), levels) * deg2perpix

    return confidence_areas, search_area, inj_point_cumulative_percentage


def confidence_band(nsamples, alpha=0.95):
    n = nsamples
    k = np.arange(0, n + 1)
    p = k / n
    ci_lo, ci_hi = scipy.stats.beta.interval(alpha, k + 1, n - k + 1)
    return p, ci_lo, ci_hi
