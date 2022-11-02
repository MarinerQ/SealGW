import bilby
import numpy as np 
import time
from bilby.gw import conversion
from pycbc.filter import matched_filter,matched_filter_core
from pycbc.types.timeseries import TimeSeries
from pycbc.types.frequencyseries import FrequencySeries
from lal import LIGOTimeGPS


def generate_random_spin(Nsample):
    ''' 
    a random point in unit sphere
    (r,theta,phi) is the sphere coordinate
    '''
    r = np.random.random(Nsample)
    phi = 2*np.pi*np.random.random(Nsample)
    cos_theta = 2*np.random.random(Nsample)-1.0
    
    sin_theta = np.sqrt(1-cos_theta**2)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    spin_x = r*sin_theta*cos_phi
    spin_y = r*sin_theta*sin_phi
    spin_z = r*cos_theta
    
    return spin_x, spin_y, spin_z

def generate_random_mass(Nsample,m1_low,m1_high,q_low,m2_low):
    m1 = np.random.uniform(low=m1_low, high=m1_high, size=Nsample)
    mass_ratio = np.random.uniform(low=q_low, high=1, size=Nsample)

    m2 = m1 * mass_ratio
    index = np.where(m2<m2_low)[0]
    while len(index)!=0:
        corrected_mass_ratio = np.random.uniform(low=q_low, high=1, size=len(index))
        m2[index] = m1[index] * corrected_mass_ratio
        index = np.where(m2<m2_low)[0]

    return m1, m2
    
def generate_random_angle(Nsample, flag, low=0, high=2*np.pi):
    '''
    flag='cos' works for iota, whose cosine is uniform in [-1,1]
    flag='sin' works for dec, whose sine is uniform in [-1,1]
    flag='flat' works for psi (0-pi), phase (0-2pi), ra (0-2pi)
    '''
    if flag=="cos":
        cos_angle =  np.random.uniform(low=-1, high=1, size=Nsample)
        random_angle = np.arccos(cos_angle)
    elif flag=="sin":
        sindec = np.random.uniform(low=-1, high=1, size=Nsample)
        random_angle = np.arcsin(sindec)
    elif flag=="flat":
        random_angle = np.random.uniform(low=low, high=high, size=Nsample)

    return random_angle

def generate_random_distance(Nsample, low, high):
    '''
    genreate distance that is uniform in space, i.e., prob density function p(r) \propto r^2
    unit: Mpc
    '''
    # check: plt.hist(np.random.power(a=3, size=10000), bins=50)
    random_dl = (high-low)*np.random.power(a=3, size=Nsample) + low

    return random_dl


def generate_random_inject_paras(Nsample, dmin, dmax, m1_low,m1_high,q_low, a_max, m2_low, spin_type='aligned'):

    # mass:2
    mass_1, mass_2 = generate_random_mass(Nsample,m1_low=m1_low,m1_high=m1_high,q_low=q_low,m2_low=m2_low)
    chirp_mass = conversion.component_masses_to_chirp_mass(mass_1,mass_2)
    mass_ratio = mass_2/mass_1

    # spin+thetajn:7
    '''
    if spin_type == 'precessing': # doesn't work for our localization algorithm. Won't use it.
        spin_1x, spin_1y, spin_1z = generate_random_spin(Nsample)
        spin_2x, spin_2y, spin_2z = generate_random_spin(Nsample)
        iota = generate_random_angle(Nsample, 'cos')

        fref_list = np.zeros(Nsample)+50.0
        phiref_list = np.zeros(Nsample)
        converted_spin = pespin.spin_angles(mass_1,mass_2,iota , spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,spin_2z, fref_list,phiref_list)

        theta_jn = converted_spin[:,0]
        phi_jl = converted_spin[:,1]
        tilt_1 = converted_spin[:,2]
        tilt_2 = converted_spin[:,3]
        phi_12 = converted_spin[:,4]
        a_1 = converted_spin[:,5]
        a_2 = converted_spin[:,6]
    '''
    if spin_type == 'aligned':
        a_1 = np.random.uniform(low=0, high=a_max, size=Nsample)
        a_2 = np.random.uniform(low=0, high=a_max, size=Nsample)
        phi_jl = np.zeros(Nsample)
        tilt_1 = np.zeros(Nsample)
        tilt_2 = np.zeros(Nsample)
        phi_12 = np.zeros(Nsample)
        iota = generate_random_angle(Nsample, 'cos')

    
    # others:6
    psi = generate_random_angle(Nsample, 'flat', low=0, high=np.pi)
    phase = generate_random_angle(Nsample, 'flat', low=0, high=2*np.pi)
    ra = generate_random_angle(Nsample, 'flat', low=0, high=2*np.pi)
    dec = generate_random_angle(Nsample, 'sin')
    luminosity_distance = generate_random_distance(Nsample, low=dmin, high=dmax)
    geocent_time = np.random.uniform(low=0, high=3.14e7, size=Nsample)

    para_list = [chirp_mass,mass_ratio,a_1,a_2,tilt_1,tilt_2,phi_12,phi_jl,
                iota, psi, phase, ra, dec, luminosity_distance, geocent_time]
    samples = np.zeros(shape=(Nsample,len(para_list)) )
    for i in range(len(para_list)):
        samples[:,i] = para_list[i] 
    return samples


def get_inj_paras(parameter_values, parameter_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl', 'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']):
    inj_paras = dict()
    for i in range(len(parameter_names)):
            inj_paras[parameter_names[i]] = parameter_values[i]
    return inj_paras 

def calculate_snr_kernel(sample_ID, samples, ifos, waveform_generator, results):
    inj_para = get_inj_paras(samples[sample_ID])
    #inj_para = bilby.gw.conversion.generate_all_bbh_parameters(inj_para)
    h_dict = waveform_generator.frequency_domain_strain(parameters=inj_para)

    net_snr_sq = 0
    for det in ifos:
        signal = det.get_detector_response(h_dict, inj_para)
        net_snr_sq += det.optimal_snr_squared(signal)

    results[sample_ID] = np.sqrt(abs(net_snr_sq))
    #return np.sqrt(net_snr_sq)


def snr_generator(ifos, waveform_generator, injection_parameter):
    ''' 
    Generate SNR timeseries and sigmas (waveform normalization factor).

    Input:
    ifos: bilby ifos
    waveform_generator: bilby waveform_generator
    injection_parameter: dict of injection parameters, as in bilby

    return: two lists, as the sequence in input ifos
    snr_timeseries_list: a list of snr timeseries (pycbc timeseries)
    sigma_list: a list of sigmas
    '''
    injection_parameters_cs = injection_parameter.copy()
    injection_parameters_cs['theta_jn'] = 0
    injection_parameters_cs['luminosity_distance'] = 1

    snr_list = []
    sigma_list = []
    #for i in range(len(ifos)):
    for det in ifos:
        #det = ifos[i]
        freq_mask = det.frequency_mask
        delta_t = 1. / det.strain_data.sampling_frequency
        delta_f = det.frequency_array[1] - det.frequency_array[0]
        epoch=LIGOTimeGPS(det.strain_data.start_time)
        
        d_pycbc = det.strain_data.to_pycbc_timeseries()
        hc = waveform_generator.time_domain_strain(injection_parameters_cs)['plus']
        hc_pycbc = TimeSeries(hc, delta_t=delta_t,epoch=epoch)
        hc_pycbc.cyclic_time_shift(injection_parameters_cs['geocent_time'])
        psd_pycbc = FrequencySeries(det.power_spectral_density_array, delta_f=delta_f,epoch=epoch)
        
        snr = matched_filter(hc_pycbc, d_pycbc, psd=psd_pycbc,
                        low_frequency_cutoff=det.frequency_array[freq_mask][0],high_frequency_cutoff=det.frequency_array[freq_mask][-1])
        
        hc_fd = waveform_generator.frequency_domain_strain(injection_parameters_cs)['plus']
        sigma = bilby.gw.utils.noise_weighted_inner_product(hc_fd, hc_fd, det.power_spectral_density_array, det.duration)
        sigma = np.sqrt( np.real(sigma) )
        
        snr_list.append(snr)
        sigma_list.append(sigma)
    
    return snr_list, sigma_list