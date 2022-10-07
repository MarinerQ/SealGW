import bilby
import numpy as np 
import time
import sys
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
from functools import partial
from matplotlib.pyplot import MultipleLocator
from bilby.gw import conversion
from scipy.optimize import leastsq

#from pesummary.gw.conversions import spins as pespin

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

# fitting functions
def select_aij_according_to_snr(file, low, high):
    
    # select
    selected_a11 = file[np.where( (file[:,0]>=low) * (file[:,1]<high) )][:,1]
    selected_a12 = file[np.where( (file[:,0]>=low) * (file[:,1]<high) )][:,2]
    selected_a21 = file[np.where( (file[:,0]>=low) * (file[:,1]<high) )][:,3]
    selected_a22 = file[np.where( (file[:,0]>=low) * (file[:,1]<high) )][:,4]
    
    # put aij together
    alist = np.array([])
    alist = np.append(alist,selected_a11)
    alist = np.append(alist,selected_a12)
    alist = np.append(alist,selected_a21)
    alist = np.append(alist,selected_a22)
    return np.array(alist)

def f(x,mu,sigma):
    # Normalized Gaussian PDF
    return np.exp(-(x-mu)**2/2/sigma**2)/np.sqrt(2*np.pi)/sigma

def initial_estimate(snr):
    # design noise
    mu = 0.00029915*snr - 0.0001853
    #sigma = 0.0001759*snr + 3.75904e-05
    sigma = mu/np.sqrt(2.71828*np.log(2))

    return mu, sigma

def error_v1new(paras,x,n_i):
    '''
    paras = [mu,sigma]
    '''
    mu,sigma = paras
    ff = (f(x,mu,sigma)+f(x,-mu,sigma))/2
    return ff - n_i


def bins_to_center_values(bins):
    center_values = []
    for i in range(0,len(bins)-1):
        center_values.append( (bins[i]+bins[i+1])/2 )
    center_values = np.array(center_values)
    return center_values

def ls_fit_bi(snr,samples):  # fit 2 Gaussian prior
    n,bins,patches = plt.hist(samples,bins='auto',density=True)
    x = bins_to_center_values(bins)
    mu_init, sigma_init = initial_estimate(snr)
    paras0=[mu_init,sigma_init]
    paras_fitted = leastsq(error_v1new,paras0,args=(x,n))[0]
    return paras_fitted


####################
def get_inj_paras(parameter_values, parameter_names = ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl',
                'theta_jn','psi','phase','ra','dec','luminosity_distance','geocent_time']):
    inj_paras = dict()
    for i in range(len(parameter_names)):
            inj_paras[parameter_names[i]] = parameter_values[i]
    return inj_paras 

def calculate_snr_kernel(sample_ID, samples, ifos, wave_gen, results):
    inj_para = get_inj_paras(samples[sample_ID])
    #inj_para = bilby.gw.conversion.generate_all_bbh_parameters(inj_para)
    h_dict = wave_gen.frequency_domain_strain(parameters=inj_para)

    net_snr_sq = 0
    for det in ifos:
        signal = det.get_detector_response(h_dict, inj_para)
        net_snr_sq += det.optimal_snr_squared(signal)

    results[sample_ID] = np.sqrt(abs(net_snr_sq))
    #return np.sqrt(net_snr_sq)
    


if __name__ == '__main__':
    time_start = time.time()
    
    # python musigma_fitting.py 30000 LHV BNS 170817 4
    Nsample = int(sys.argv[1])  # e.g. 50000
    det_flag = str(sys.argv[2])  # e.g. LHVK
    source_type = str(sys.argv[3])  # e.g. BNS
    psd_label = str(sys.argv[4])  # e.g. O4 170817 ... 
    core_num = int(sys.argv[-1])
    
    print('Start calculation.')
    print('Nsample: {}, Detector: {}, Source type: {}, PSD: {}, Core number: {}'.format(Nsample,det_flag,source_type,psd_label,core_num))
    print("Generating parameters...")
    # generate samples
    if source_type == 'BNS':
        samples = generate_random_inject_paras(Nsample=Nsample,dmin = 0, dmax=200,m1_low=1.1,m1_high=2, q_low = 0.8, a_max=0.1, m2_low=1.1)
    elif source_type == 'BBH':
        samples = generate_random_inject_paras(Nsample=Nsample,dmin = 0, dmax=4000,m1_low=6,m1_high=90, q_low = 0.25, a_max=0.1, m2_low=6)
    elif source_type == 'NSBH':
        samples = generate_random_inject_paras(Nsample=Nsample,dmin = 0, dmax=500,m1_low=6,m1_high=90, q_low = 0.1, a_max=0.1, m2_low=1.1)
    else:
        raise('Source type error!')

    # choose detector
    det_name_list = []
    for letter in det_flag:
        if letter in ['L', 'H', 'V', 'K', 'I']:
            det_name_list.append(letter+'1')
            fmin=20
        elif letter in ['C', 'E']:
            if letter=='C':
                det_name_list.append('CE')
            else:
                det_name_list.append('ET')
            fmin=10
        else:
            raise('Detector name error!')
    
    ifos = bilby.gw.detector.InterferometerList(det_name_list)

    duration = 32
    sampling_frequency = 4096
    # set detector paramaters
    for i in range(len(ifos)):
        det = ifos[i]
        det.duration = duration
        det.sampling_frequency=sampling_frequency
        psd_file = 'data/psd/{}/{}_psd.txt'.format(psd_label, det_name_list[i])
        psd = power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file=psd_file)
        det.power_spectral_density = psd 
    
    # waveform generator
    if source_type == 'BNS':
        waveform_arguments = dict(waveform_approximant='TaylorF2',
                                    reference_frequency=50., minimum_frequency=fmin)
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=waveform_arguments)

    elif source_type == 'BBH':
        waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                                    reference_frequency=50., minimum_frequency=fmin)
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments)
    
    elif source_type == 'NSBH':
        waveform_arguments = dict(waveform_approximant='IMRPhenomNSBH',
                                    reference_frequency=50., minimum_frequency=fmin)
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
            waveform_arguments=waveform_arguments)

    
    # Calculate SNR
    manager = multiprocessing.Manager()
    snrs = manager.Array('d', range(Nsample))
    partial_work = partial(calculate_snr_kernel,samples=samples, ifos=ifos, wave_gen=waveform_generator, results=snrs)

    print("Computing SNR...")
    with Pool(core_num) as p:
        p.map(partial_work, range(Nsample) )

    print('Fitting mu-sigma-SNR relation...')
    # Calculate Aij
    # ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl',
    #            'iota','psi','phase','ra','dec','luminosity_distance','geocent_time']
    d_L = samples[:,13]
    iota = samples[:,8]
    psi = samples[:,9]
    phase = samples[:,10]

    cos_iota = np.cos(iota)
    cos_phic = np.cos(phase)
    sin_phic = np.sin(phase)
    cos_2psi = np.cos(2*psi)
    sin_2psi = np.sin(2*psi)

    A11 = ( (1+cos_iota*cos_iota)/2*cos_phic*cos_2psi - cos_iota*sin_phic*sin_2psi ) / d_L
    A12 = ( (1+cos_iota*cos_iota)/2*sin_phic*cos_2psi + cos_iota*cos_phic*sin_2psi ) / d_L
    A21 = ( -(1+cos_iota*cos_iota)/2*cos_phic*sin_2psi - cos_iota*sin_phic*cos_2psi ) / d_L
    A22 = ( -(1+cos_iota*cos_iota)/2*sin_phic*sin_2psi + cos_iota*cos_phic*cos_2psi ) / d_L

    result = np.vstack([snrs, A11, A12, A21, A22]).T
    np.savetxt('data/psd/{}/snr_A_{}_{}.txt'.format(psd_label, source_type, det_flag), result)


    #snr_steps = [0,8,10,12,14,16,18,20,22,24,26,28,30,32,114514]
    mu_list=[]
    sigma_list=[]

    sample_list = []
    n_list = []
    bin_list = []
    snr_steps = np.arange(9,31,2)
    for i in range(len(snr_steps)):
        snr_step = snr_steps[i]
        Aijs = select_aij_according_to_snr(result, snr_step-1, snr_step+1)
        n,bins,patches = plt.hist(Aijs,bins='auto',density=True)
        n_list.append(n)
        bin_list.append(bins)

        paras_fit = ls_fit_bi(snr_step,Aijs)
        plt.clf() 
        mu_list.append(paras_fit[0])
        sigma_list.append(paras_fit[1])

    a,b = np.polyfit(snr_steps,mu_list,1)
    c,d = np.polyfit(snr_steps,sigma_list,1)
    np.savetxt('data/psd/{}/abcd_{}_{}.txt'.format(psd_label, source_type, det_flag), [a,b,c,d])


    # test plots
    ####### linear fitting #######
    labelsize=22
    ticksize=18
    legendsize = 'large'

    plt.figure(figsize=(10,7.5))
    plt.rcParams.update({"font.size":16})

    #plt.text(7.8, 9.5, r'x 10$^{-3}$',fontsize=16)
    plt.yticks(size = ticksize)
    plt.xticks(size = ticksize)
    plt.scatter(snr_steps, 1e3*np.array(mu_list),marker='x',color='orangered',label=r"$\mu$")
    plt.plot(snr_steps,1e3*(a*snr_steps+b),color='royalblue',label=r"Linear fitting of $\mu$")
    plt.scatter(snr_steps, 1e3*np.array(sigma_list),marker='o',color='darkorange',label=r"$\sigma$")
    plt.plot(snr_steps,1e3*(c*snr_steps+d),color='forestgreen',label=r"Linear fitting of $\sigma$")
    plt.xlabel("Signal-to-noise Ratio",size = labelsize)
    plt.ylabel(r'$\mu, \sigma \times$ 1000',size = labelsize)
    plt.legend(loc = 'best',ncol=2, fontsize=legendsize)
    plt.grid()

    plt.savefig('data/psd/{}/linear_fit_{}_{}.pdf'.format(psd_label, source_type, det_flag))
    ####### end of linear fitting #######
    ####### bimodal fitting #######


    labelsize=18
    ticksize=16
    legendsize = 'x-large'

    A_range = np.linspace(-0.04,0.04,200)
    plt.figure(figsize=(14,10))
    color_bar = 'cornflowerblue'
    color_line = 'red'
    test_snr_low = [12, 16, 20, 24]
    test_snr_high = [16, 20, 24, 30]
    for i in [1,2,3,4]:
        plt.subplot(2,2,i)
        plt.yticks(size = ticksize)
        plt.xticks(size = ticksize)
        snr_low = test_snr_low[i-1]
        snr_high = test_snr_high[i-1]
        snr_middle = (snr_high+snr_low)/2
        mu = a*snr_middle + b
        sigma = c*snr_middle +b
        theo_pdf = (f(A_range,mu,sigma)+f(A_range,-mu,sigma))/2
        plt.hist(select_aij_according_to_snr(result, snr_low, snr_high),bins='auto',density=True, label='SNR {}-{}'.format(snr_low,snr_high),color=color_bar)
        plt.plot(A_range,theo_pdf,color=color_line)
        plt.ylabel('Probability density',size=labelsize)
        plt.xlim(-0.05,0.05)
        plt.legend()

    plt.savefig('data/psd/{}/bimodal_fit_{}_{}.pdf'.format(psd_label, source_type, det_flag))
    ####### end of bimodal fitting #######


    time_end = time.time()
    timecost = time_end-time_start

    print('Simulation done. Time cost: {}s.'.format(timecost))


