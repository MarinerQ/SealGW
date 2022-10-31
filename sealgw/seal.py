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
import json

from .simulation.generating_data import *
from .simulation.prior_fitting import *


class Seal():
    ''' 
    The inferface of SealGW. 
    
    You can use the Seal object to train, exam the localization algorithm, and to localize GW events.
    '''
    def __init__(self,config_dict=None):
        if config_dict is None:
            self.initialized = False
            self.description = "An uninitialized seal."

        elif type(config_dict) == dict:
            self.prior_coef_a = config_dict['a']
            self.prior_coef_b = config_dict['b']
            self.prior_coef_c = config_dict['c']
            self.prior_coef_d = config_dict['d']
            self.description = config_dict['description']

            self.initialized = True

        elif type(config_dict) == str:
            with open(config_dict) as f:
                data = f.read()
            config_dict_from_file = json.loads(data)
            self.prior_coef_a = config_dict_from_file['a']
            self.prior_coef_b = config_dict_from_file['b']
            self.prior_coef_c = config_dict_from_file['c']
            self.prior_coef_d = config_dict_from_file['d']
            self.description = config_dict_from_file['description']

            self.initialized = True

            
        
    def save_config_dict(self, filename):
        if self.initialized == False:
            raise Exception("Seal not initialized!")

        config_dict = {
            'description': self.description,
            'a':self.prior_coef_a,
            'b':self.prior_coef_b,
            'c':self.prior_coef_c,
            'd':self.prior_coef_d
            }

        with open(filename, 'w') as file:
            file.write(json.dumps(config_dict))
        


    def fitting_mu_sigma_snr_relation(self, Nsample, det_name_list, source_type, ncpu, fmin=20, duration = 32, sampling_frequency = 4096, use_bilby_psd = True, custom_psd_path = None, plotsave=None):
        if self.initialized:
            raise Exception("This seal is already initialized!")
        if use_bilby_psd and custom_psd_path:
            raise Exception("You can use only one of them: bilby PSD or your own PSD. Disable one of them. ")
        if custom_psd_path:
            if len(custom_psd_path) != len(det_name_list):
                raise Exception("Number of PSDs does not match with number of detectors.")
        

        ifos = bilby.gw.detector.InterferometerList(det_name_list)

        # set detector paramaters
        for i in range(len(ifos)):
            det = ifos[i]
            det.duration = duration
            det.sampling_frequency=sampling_frequency
            #psd_file = 'psd/{}/{}_psd.txt'.format(psd_label, det_name_list[i])
            if custom_psd_path:  # otherwise auto-set by bilby
                psd_file = custom_psd_path[i]
                psd = bilby.gw.detector.PowerSpectralDensity(psd_file=psd_file)
                det.power_spectral_density = psd 
        

        # waveform generator and samples
        waveform_generator = get_wave_gen(source_type, fmin,duration,sampling_frequency)
        samples = get_fitting_source_para_sample(source_type,Nsample)
        
        
        manager = multiprocessing.Manager()
        snrs = manager.Array('d', range(Nsample))
        partial_work = partial(calculate_snr_kernel,samples=samples, ifos=ifos, wave_gen=waveform_generator, results=snrs)

        print("Computing SNR...")
        with Pool(ncpu) as p:
            p.map(partial_work, range(Nsample) )

        print('Fitting mu-sigma-SNR relation...')
        # Calculate Aij
        # ['chirp_mass','mass_ratio','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl',
        #            'iota','psi','phase','ra','dec','luminosity_distance','geocent_time']
        d_L = samples[:,13]
        iota = samples[:,8]
        psi = samples[:,9]
        phase = samples[:,10]

        A11, A12, A21, A22 = para_conversion(d_L, iota, psi, phase)

        simulation_result = np.vstack([snrs, A11, A12, A21, A22]).T

        snr_steps = np.arange(9,31,2)
        a,b,c,d,mu_list,sigma_list = fitting_abcd(simulation_result, snr_steps)

        self.prior_coef_a = a
        self.prior_coef_b = b
        self.prior_coef_c = c
        self.prior_coef_d = d
        self.description = 'Seal trained with '
        if custom_psd_path:
            self.description += 'own PSD for '
        else:
            self.description += 'bilby PSD for '

        for i in range(len(det_name_list)):
            self.description += (det_name_list[i] + ' ')

        self.initialized = True
        print("Fitting done!\na = {}\nb = {}\nc = {}\nd = {}".format(a,b,c,d))

        if plotsave:
            linear_fitting_plot(snr_steps, mu_list, sigma_list, a, b, c, d, plotsave[0])
            bimodal_fitting_plot(simulation_result, a, b, c, d, plotsave[1])
            
    
    def localize(self, snr_timeseries, sigmas):
        return

        
