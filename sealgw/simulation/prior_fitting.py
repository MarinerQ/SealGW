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