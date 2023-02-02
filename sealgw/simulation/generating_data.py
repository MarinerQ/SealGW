import bilby
import numpy as np
import spiir
from bilby.gw.conversion import (
    chirp_mass_and_mass_ratio_to_component_masses,
    component_masses_to_chirp_mass,
    convert_to_lal_binary_black_hole_parameters,
    convert_to_lal_binary_neutron_star_parameters,
)
from lal import LIGOTimeGPS
from pycbc.filter import matched_filter
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.types.timeseries import TimeSeries
from pycbc.waveform import get_fd_waveform, get_td_waveform

# fmt: off
_PARAMETERS = [
    'chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
    'theta_jn', 'psi', 'phase', 'ra', 'dec', 'luminosity_distance', 'geocent_time'
]
PARAMETERS = {"BBH": _PARAMETERS}
for NS_SOURCE in ["BNS", "NSBH", "BNS_EW_FD", "BNS_EW_TD"]:
    PARAMETERS[NS_SOURCE] = _PARAMETERS + ["lambda_1", "lambda_2"]
# fmt: on


def generate_random_spin(Nsample):
    """
    a random point in unit sphere
    (r,theta,phi) is the sphere coordinate
    """
    r = np.random.random(Nsample)
    phi = 2 * np.pi * np.random.random(Nsample)
    cos_theta = 2 * np.random.random(Nsample) - 1.0

    sin_theta = np.sqrt(1 - cos_theta**2)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    spin_x = r * sin_theta * cos_phi
    spin_y = r * sin_theta * sin_phi
    spin_z = r * cos_theta

    return spin_x, spin_y, spin_z


def generate_random_mass(Nsample, m1_low, m1_high, q_low, m2_low):
    m1 = np.random.uniform(low=m1_low, high=m1_high, size=Nsample)

    if m1_low > 5 and m2_low < 3:  # is NSBH
        m2 = np.random.uniform(low=m2_low, high=2, size=Nsample)
        mass_ratio = m2 / m1
        index = np.where(mass_ratio < q_low)[0]
        while len(index) != 0:
            corrected_mass_1 = np.random.uniform(
                low=m1_low, high=m1_high, size=len(index)
            )
            m1[index] = corrected_mass_1
            mass_ratio = m2 / m1
            index = np.where(mass_ratio < q_low)[0]
    else:
        mass_ratio = np.random.uniform(low=q_low, high=1, size=Nsample)
        m2 = m1 * mass_ratio
        index = np.where(m2 < m2_low)[0]
        while len(index) != 0:
            corrected_mass_ratio = np.random.uniform(low=q_low, high=1, size=len(index))
            m2[index] = m1[index] * corrected_mass_ratio
            index = np.where(m2 < m2_low)[0]

    return m1, m2


def generate_random_angle(Nsample, flag, low=0, high=2 * np.pi):
    """
    flag='cos' works for iota, whose cosine is uniform in [-1,1]
    flag='sin' works for dec, whose sine is uniform in [-1,1]
    flag='flat' works for psi (0-pi), phase (0-2pi), ra (0-2pi)
    """
    if flag == "cos":
        cos_angle = np.random.uniform(low=-1, high=1, size=Nsample)
        random_angle = np.arccos(cos_angle)
    elif flag == "sin":
        sindec = np.random.uniform(low=-1, high=1, size=Nsample)
        random_angle = np.arcsin(sindec)
    elif flag == "flat":
        random_angle = np.random.uniform(low=low, high=high, size=Nsample)

    return random_angle


def generate_random_distance(Nsample, low, high):
    r"""
    genreate distance that is uniform in space,
    i.e., prob density function p(r) \propto r^2
    unit: Mpc
    """

    # check: plt.hist(np.random.power(a=3, size=10000), bins=50)
    random_dl = (high - low) * np.random.power(a=3, size=Nsample) + low

    return random_dl


def generate_random_inject_paras(
    Nsample,
    dmin,
    dmax,
    m1_low,
    m1_high,
    q_low,
    a_max,
    m2_low,
    source_type,
    spin_type='aligned',
    pre_t=None,
    flow=None,
):

    # mass: 2 parameters
    mass_1, mass_2 = generate_random_mass(
        Nsample, m1_low=m1_low, m1_high=m1_high, q_low=q_low, m2_low=m2_low
    )
    chirp_mass = component_masses_to_chirp_mass(mass_1, mass_2)
    mass_ratio = mass_2 / mass_1

    # spin + theta_jn: 7 parameters
    assert spin_type == "aligned", "Only aligned spins supported for this algorithm."
    a_1 = np.random.uniform(low=0, high=a_max, size=Nsample)
    a_2 = np.random.uniform(low=0, high=a_max, size=Nsample)
    phi_jl = np.zeros(Nsample)
    tilt_1 = np.zeros(Nsample)
    tilt_2 = np.zeros(Nsample)
    phi_12 = np.zeros(Nsample)
    iota = generate_random_angle(Nsample, "cos")

    # extrinsics: 6 parameters
    psi = generate_random_angle(Nsample, "flat", low=0, high=np.pi)
    phase = generate_random_angle(Nsample, "flat", low=0, high=2 * np.pi)
    ra = generate_random_angle(Nsample, "flat", low=0, high=2 * np.pi)
    dec = generate_random_angle(Nsample, "sin")
    luminosity_distance = generate_random_distance(Nsample, low=dmin, high=dmax)
    geocent_time = np.random.uniform(low=0, high=3.14e7, size=Nsample)

    parameter_arrays = [
        chirp_mass, mass_ratio, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, iota,
        psi, phase, ra, dec, luminosity_distance, geocent_time,
    ]  # fmt: skip

    # additional parameters for non-BBH source types
    if source_type == 'BNS':
        lambda_1 = np.random.uniform(low=400, high=450, size=Nsample)
        lambda_2 = np.random.uniform(low=400, high=450, size=Nsample)
        parameter_arrays += [lambda_1, lambda_2]

    elif source_type == 'NSBH':
        lambda_1 = np.zeros(Nsample)
        lambda_2 = np.random.uniform(low=400, high=450, size=Nsample)
        parameter_arrays += [lambda_1, lambda_2]

    elif source_type in ['BNS_EW_FD', 'BNS_EW_TD']:
        lambda_1 = np.random.uniform(low=400, high=450, size=Nsample)
        lambda_2 = np.random.uniform(low=400, high=450, size=Nsample)
        # premerger_time = np.zeros(Nsample) + pre_t
        # flows = np.zeros(Nsample) + flow
        parameter_arrays += [lambda_1, lambda_2]

        # para_list = [chirp_mass,mass_ratio,a_1,a_2,tilt_1,tilt_2,phi_12,phi_jl,
        #        iota, psi, phase, ra, dec, luminosity_distance, geocent_time,
        #        lambda_1, lambda_2, premerger_time, flows]

    else:
        # raise Exception('Source type error!')
        pass

    return np.stack(parameter_arrays, axis=1)


def zip_injection_parameters(values, source_type, names=None):
    return dict(zip(names or PARAMETERS[source_type], values))


def get_inj_paras(
    parameter_values,
    source_type,
    parameter_names=[
        'chirp_mass',
        'mass_ratio',
        'a_1',
        'a_2',
        'tilt_1',
        'tilt_2',
        'phi_12',
        'phi_jl',
        'theta_jn',
        'psi',
        'phase',
        'ra',
        'dec',
        'luminosity_distance',
        'geocent_time',
    ],
):
    inj_paras = dict()
    if source_type in ['BNS', 'NSBH']:
        parameter_names = [
            'chirp_mass',
            'mass_ratio',
            'a_1',
            'a_2',
            'tilt_1',
            'tilt_2',
            'phi_12',
            'phi_jl',
            'theta_jn',
            'psi',
            'phase',
            'ra',
            'dec',
            'luminosity_distance',
            'geocent_time',
            'lambda_1',
            'lambda_2',
        ]
    elif source_type in ['BNS_EW_FD', 'BNS_EW_TD']:
        parameter_names = [
            'chirp_mass',
            'mass_ratio',
            'a_1',
            'a_2',
            'tilt_1',
            'tilt_2',
            'phi_12',
            'phi_jl',
            'theta_jn',
            'psi',
            'phase',
            'ra',
            'dec',
            'luminosity_distance',
            'geocent_time',
            'lambda_1',
            'lambda_2',
        ]
    for i in range(len(parameter_names)):
        inj_paras[parameter_names[i]] = parameter_values[i]
    return inj_paras


def premerger_time_to_freq(pre_t, m1, m2):
    '''
    Maggiore, Gravtational wave, Vol. 1, eqs.(4.20) and (4.21)
    '''
    mc = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
    freq = 134 * (1 / pre_t) ** (3 / 8) * (1.21 / mc) ** (5 / 8)
    return freq


def bns_truncated_fd_bilbypara(
    farray,
    chirp_mass,
    mass_ratio,
    a_1,
    a_2,
    luminosity_distance,
    phase,
    theta_jn,
    ra,
    dec,
    psi,
    geocent_time,
    lambda_1,
    lambda_2,
    premerger_time,
    flow,
    **kwargs
):
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(
        chirp_mass, mass_ratio
    )
    fhigh = np.ceil(premerger_time_to_freq(premerger_time, mass_1, mass_2))
    deltaf = farray[1] - farray[0]
    approx = 'TaylorF2'  # TaylorF2 IMRPhenomPv2 IMRPhenomPv2_NRTidalv2
    waveform_polarizations = {}
    waveform_polarizations['plus'], waveform_polarizations['cross'] = get_fd_waveform(
        approximant=approx,
        mass1=mass_1,
        mass2=mass_2,
        distance=luminosity_distance,
        inclination=theta_jn,
        coa_phase=phase,
        lambda1=lambda_1,
        lambda2=lambda_2,
        spin1x=0,
        spin1y=0,
        spin1z=a_1,
        spin2x=0,
        spin2y=0,
        spin2z=a_2,
        delta_f=deltaf,
        f_lower=flow,
        f_final=fhigh,
        f_ref=50.0,
    )

    N_gw = len(waveform_polarizations['plus'])
    N_wave_gen = len(farray)
    zero_array = np.zeros(N_wave_gen - N_gw)
    for mode in waveform_polarizations.keys():
        waveform_polarizations[mode] = np.append(
            waveform_polarizations[mode], zero_array
        )
        # waveform_polarizations[mode] = waveform_polarizations[mode]

    return waveform_polarizations


def bns_truncated_td_bilbypara(
    tarray,
    chirp_mass,
    mass_ratio,
    a_1,
    a_2,
    luminosity_distance,
    phase,
    theta_jn,
    ra,
    dec,
    psi,
    geocent_time,
    lambda_1,
    lambda_2,
    premerger_time,
    flow,
    **kwargs
):
    mass_1, mass_2 = chirp_mass_and_mass_ratio_to_component_masses(
        chirp_mass, mass_ratio
    )
    duration = np.ceil(tarray[-1] - tarray[0])
    if duration < premerger_time:
        raise ValueError('duration < premerger_time!')

    # to reduce computing time
    rough_estimation_flow = premerger_time_to_freq(duration, mass_1, mass_2)
    # rough_estimation_flow -= 0.5

    deltat = tarray[1] - tarray[0]
    approx = 'TaylorT4'  # TaylorF2 IMRPhenomPv2 IMRPhenomPv2_NRTidalv2
    waveform_polarizations = {}
    waveform_polarizations['plus'], waveform_polarizations['cross'] = get_td_waveform(
        approximant=approx,
        mass1=mass_1,
        mass2=mass_2,
        distance=luminosity_distance,
        inclination=theta_jn,
        coa_phase=phase,
        lambda1=lambda_1,
        lambda2=lambda_2,
        spin1x=0,
        spin1y=0,
        spin1z=0,
        spin2x=0,
        spin2y=0,
        spin2z=0,
        delta_t=deltat,
        f_lower=rough_estimation_flow,
    )

    zero_index = np.where(
        waveform_polarizations['plus'].sample_times > -1 * premerger_time
    )[0]
    zero_index2 = np.where(waveform_polarizations['plus'].sample_times < -1 * duration)[
        0
    ]
    for mode in waveform_polarizations.keys():
        waveform_polarizations[mode] = waveform_polarizations[mode].numpy()
        waveform_polarizations[mode][zero_index] = np.zeros(len(zero_index))
        waveform_polarizations[mode] = np.delete(
            waveform_polarizations[mode], zero_index2
        )

    return waveform_polarizations


def bns_truncated_fd(
    farray,
    mass_1,
    mass_2,
    spin_1z,
    spin_2z,
    luminosity_distance,
    phase,
    iota,
    ra,
    dec,
    psi,
    geocent_time,
    lambda_1,
    lambda_2,
    premerger_time,
    flow,
    **kwargs
):
    fhigh = np.ceil(premerger_time_to_freq(premerger_time, mass_1, mass_2))
    deltaf = farray[1] - farray[0]
    approx = 'TaylorF2'  # TaylorF2 IMRPhenomPv2 IMRPhenomPv2_NRTidalv2
    waveform_polarizations = {}
    waveform_polarizations['plus'], waveform_polarizations['cross'] = get_fd_waveform(
        approximant=approx,
        mass1=mass_1,
        mass2=mass_2,
        distance=luminosity_distance,
        inclination=iota,
        coa_phase=phase,
        lambda1=lambda_1,
        lambda2=lambda_2,
        spin1x=0,
        spin1y=0,
        spin1z=spin_1z,
        spin2x=0,
        spin2y=0,
        spin2z=spin_2z,
        delta_f=deltaf,
        f_lower=flow,
        f_final=fhigh,
        f_ref=50.0,
    )

    N_gw = len(waveform_polarizations['plus'])
    N_wave_gen = len(farray)
    zero_array = np.zeros(N_wave_gen - N_gw)
    for mode in waveform_polarizations.keys():
        waveform_polarizations[mode] = np.append(
            waveform_polarizations[mode], zero_array
        )
        waveform_polarizations[mode] = waveform_polarizations[mode]

    return waveform_polarizations


def bns_truncated_td(
    tarray,
    mass_1,
    mass_2,
    spin_1z,
    spin_2z,
    luminosity_distance,
    phase,
    iota,
    ra,
    dec,
    psi,
    geocent_time,
    lambda_1,
    lambda_2,
    premerger_time,
    flow,
    **kwargs
):
    duration = np.ceil(tarray[-1] - tarray[0])
    if duration < premerger_time:
        raise ValueError('duration < premerger_time!')

    # to reduce computing time
    rough_estimation_flow = premerger_time_to_freq(duration, mass_1, mass_2)
    rough_estimation_flow -= 0.5

    deltat = tarray[1] - tarray[0]
    approx = 'TaylorT4'  # TaylorF2 IMRPhenomPv2 IMRPhenomPv2_NRTidalv2
    waveform_polarizations = {}
    waveform_polarizations['plus'], waveform_polarizations['cross'] = get_td_waveform(
        approximant=approx,
        mass1=mass_1,
        mass2=mass_2,
        distance=luminosity_distance,
        inclination=iota,
        coa_phase=phase,
        lambda1=lambda_1,
        lambda2=lambda_2,
        spin1x=0,
        spin1y=0,
        spin1z=0.0,
        spin2x=0,
        spin2y=0,
        spin2z=0.0,
        delta_t=deltat,
        f_lower=rough_estimation_flow,
    )

    zero_index = np.where(
        waveform_polarizations['plus'].sample_times > -1 * premerger_time
    )[0]
    zero_index2 = np.where(waveform_polarizations['plus'].sample_times < -1 * duration)[
        0
    ]
    for mode in waveform_polarizations.keys():
        waveform_polarizations[mode] = waveform_polarizations[mode].numpy()
        waveform_polarizations[mode][zero_index] = np.zeros(len(zero_index))
        waveform_polarizations[mode] = np.delete(
            waveform_polarizations[mode], zero_index2
        )

    return waveform_polarizations


def get_wave_gen(source_type, fmin, duration, sampling_frequency):

    if source_type == 'BNS':
        waveform_arguments = dict(
            waveform_approximant='TaylorF2',
            reference_frequency=50.0,
            minimum_frequency=fmin,
        )
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
            parameter_conversion=convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=waveform_arguments,
        )

    elif source_type == 'BBH':
        waveform_arguments = dict(
            waveform_approximant='IMRPhenomPv2',
            reference_frequency=50.0,
            minimum_frequency=fmin,
        )
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments,
        )

    elif source_type == 'NSBH':
        waveform_arguments = dict(
            waveform_approximant='IMRPhenomNSBH',
            reference_frequency=50.0,
            minimum_frequency=fmin,
        )
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
            waveform_arguments=waveform_arguments,
        )

    elif source_type == 'BNS_EW_FD':
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bns_truncated_fd_bilbypara,
            parameter_conversion=convert_to_lal_binary_neutron_star_parameters,
        )

    elif source_type == 'BNS_EW_TD':
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration,
            sampling_frequency=sampling_frequency,
            time_domain_source_model=bns_truncated_td_bilbypara,
            parameter_conversion=convert_to_lal_binary_neutron_star_parameters,
        )

    else:
        raise Exception('Source type error!')

    return waveform_generator


def get_example_injpara(source_type):
    '''
    This function is used to generate an example injection parameter for (lower bound) horizon estimation.
    It therefore gives lighter masses.
    '''
    example_injection_parameter = dict()

    if source_type == 'BNS':
        mc = bilby.gw.conversion.component_masses_to_chirp_mass(1.2, 1.2)
        example_injection_parameter['chirp_mass'] = mc
        example_injection_parameter['mass_ratio'] = 1
        example_injection_parameter['a_1'] = 0
        example_injection_parameter['a_2'] = 0
        example_injection_parameter['tilt_1'] = 0
        example_injection_parameter['tilt_2'] = 0
        example_injection_parameter['phi_12'] = 0
        example_injection_parameter['phi_jl'] = 0
        example_injection_parameter['lambda_1'] = 425
        example_injection_parameter['lambda_2'] = 425

        example_injection_parameter['theta_jn'] = 0
        example_injection_parameter['psi'] = 0
        example_injection_parameter['phase'] = 0
        example_injection_parameter['ra'] = 0
        example_injection_parameter['dec'] = 0
        example_injection_parameter['luminosity_distance'] = 1
        example_injection_parameter['geocent_time'] = 0

    elif source_type == 'BBH':
        mc = bilby.gw.conversion.component_masses_to_chirp_mass(10, 10)
        example_injection_parameter['chirp_mass'] = mc
        example_injection_parameter['mass_ratio'] = 1
        example_injection_parameter['a_1'] = 0
        example_injection_parameter['a_2'] = 0
        example_injection_parameter['tilt_1'] = 0
        example_injection_parameter['tilt_2'] = 0
        example_injection_parameter['phi_12'] = 0
        example_injection_parameter['phi_jl'] = 0

        example_injection_parameter['theta_jn'] = 0
        example_injection_parameter['psi'] = 0
        example_injection_parameter['phase'] = 0
        example_injection_parameter['ra'] = 0
        example_injection_parameter['dec'] = 0
        example_injection_parameter['luminosity_distance'] = 1
        example_injection_parameter['geocent_time'] = 0

    elif source_type == 'NSBH':
        mc = bilby.gw.conversion.component_masses_to_chirp_mass(10, 1.4)
        example_injection_parameter['chirp_mass'] = mc
        example_injection_parameter['mass_ratio'] = 1.4 / 10
        example_injection_parameter['a_1'] = 0
        example_injection_parameter['a_2'] = 0
        example_injection_parameter['tilt_1'] = 0
        example_injection_parameter['tilt_2'] = 0
        example_injection_parameter['phi_12'] = 0
        example_injection_parameter['phi_jl'] = 0
        example_injection_parameter['lambda_1'] = 0
        example_injection_parameter['lambda_2'] = 425

        example_injection_parameter['theta_jn'] = 0
        example_injection_parameter['psi'] = 0
        example_injection_parameter['phase'] = 0
        example_injection_parameter['ra'] = 0
        example_injection_parameter['dec'] = 0
        example_injection_parameter['luminosity_distance'] = 1
        example_injection_parameter['geocent_time'] = 0

    elif source_type == 'BNS_EW_FD':
        raise Exception('This function is under development!')

    elif source_type == 'BNS_EW_TD':
        raise Exception('This function is under development!')

    else:
        raise Exception('Source type error!')

    return example_injection_parameter


def get_ifos(det_name_list, duration, sampling_frequency, custom_psd_path):
    ifos = bilby.gw.detector.InterferometerList(det_name_list)

    # set detector paramaters
    for i in range(len(ifos)):
        det = ifos[i]
        det.duration = duration
        det.sampling_frequency = sampling_frequency
        # psd_file = 'psd/{}/{}_psd.txt'.format(psd_label, det_name_list[i])
        if custom_psd_path:  # otherwise auto-set by bilby
            if type(custom_psd_path) == str and '.xml' in custom_psd_path:
                temppsd = spiir.io.ligolw.array.load_psd_series_from_xml(
                    custom_psd_path
                )[det_name_list[i]]
                psdarray = temppsd.to_numpy()
                freqarray = temppsd.index.to_numpy()
                psd = bilby.gw.detector.PowerSpectralDensity(
                    frequency_array=freqarray, psd_array=psdarray
                )
            elif type(custom_psd_path) == list:
                psd_file = custom_psd_path[i]
                psd = bilby.gw.detector.PowerSpectralDensity(psd_file=psd_file)
            det.power_spectral_density = psd
    return ifos


def get_fitting_source_para_sample(source_type, Nsample, **kwargs):
    if source_type == 'BNS':
        dmax = 200
        if 'dmax' in kwargs.keys():
            dmax = kwargs['dmax']
        samples = generate_random_inject_paras(
            Nsample=Nsample,
            dmin=0,
            dmax=dmax,
            m1_low=1.1,
            m1_high=2,
            q_low=0.8,
            a_max=0.1,
            m2_low=1.1,
            source_type=source_type,
        )

    elif source_type == 'BBH':
        dmax = 4000
        if 'dmax' in kwargs.keys():
            dmax = kwargs['dmax']
        samples = generate_random_inject_paras(
            Nsample=Nsample,
            dmin=0,
            dmax=dmax,
            m1_low=6,
            m1_high=90,
            q_low=0.25,
            a_max=0.1,
            m2_low=6,
            source_type=source_type,
        )

    elif source_type == 'NSBH':
        dmax = 500
        if 'dmax' in kwargs.keys():
            dmax = kwargs['dmax']
        samples = generate_random_inject_paras(
            Nsample=Nsample,
            dmin=0,
            dmax=dmax,
            m1_low=6,
            m1_high=90,
            q_low=0.05,
            a_max=0.1,
            m2_low=1.1,
            source_type=source_type,
        )

    elif source_type in ['BNS_EW_FD', 'BNS_EW_TD']:
        dmax = 5000
        if 'dmax' in kwargs.keys():
            dmax = kwargs['dmax']
        samples = generate_random_inject_paras(
            Nsample=Nsample,
            dmin=0,
            dmax=dmax,
            m1_low=1.1,
            m1_high=2,
            q_low=0.8,
            a_max=0.1,
            m2_low=1.1,
            source_type=source_type,
        )  # , pre_t=kwargs['pre_t'], flow=kwargs['flow']

    else:
        raise Exception('Source type error!')

    return samples


def snr_generator(ifos, waveform_generator, injection_parameter):
    """
    Generate SNR timeseries and sigmas (waveform normalization factor).

    Input:
    ifos: bilby ifos
    waveform_generator: bilby waveform_generator
    injection_parameter: dict of injection parameters, as in bilby

    return: two lists, as the sequence in input ifos
    snr_timeseries_list: a list of snr timeseries (pycbc timeseries)
    sigma_list: a list of sigmas
    """
    injection_parameters_copy = injection_parameter.copy()
    injection_parameters_copy["theta_jn"] = 0
    injection_parameters_copy["luminosity_distance"] = 1

    snr_list = []
    sigma_list = []

    for det in ifos:
        freq_mask = det.frequency_mask
        delta_t = 1.0 / det.strain_data.sampling_frequency
        delta_f = det.frequency_array[1] - det.frequency_array[0]
        epoch = LIGOTimeGPS(det.strain_data.start_time)

        d_pycbc = det.strain_data.to_pycbc_timeseries()
        hc = waveform_generator.time_domain_strain(injection_parameters_copy)['plus']
        hc_pycbc = TimeSeries(hc, delta_t=delta_t, epoch=epoch)
        # if 'premerger_time' in injection_parameters_cs.keys():
        #     cyctimeshift = (
        #         injection_parameters_cs['geocent_time']
        #         - injection_parameters_cs['yctimeshift = (
        #         injection_parameters_cs['geocent_time']
        #         - injection_parameters_cs['premerger_time']
        #     )premerger_time']
        #     )
        # else:
        #    cyctimeshift = injection_parameters_cs['geocent_time']
        # hc_pycbc.cyclic_time_shift(cyctimeshift)
        psd_pycbc = FrequencySeries(
            det.power_spectral_density_array, delta_f=delta_f, epoch=epoch
        )

        snr = matched_filter(
            hc_pycbc,
            d_pycbc,
            psd=psd_pycbc,
            low_frequency_cutoff=det.frequency_array[freq_mask][0],
            high_frequency_cutoff=det.frequency_array[freq_mask][-1],
        )

        hc_fd = waveform_generator.frequency_domain_strain(injection_parameters_copy)
        sigma = bilby.gw.utils.noise_weighted_inner_product(
            hc_fd["plus"], hc_fd["plus"], det.power_spectral_density_array, det.duration
        )
        sigma = np.sqrt(np.real(sigma))

        snr_list.append(snr)
        sigma_list.append(sigma)

    return snr_list, sigma_list
