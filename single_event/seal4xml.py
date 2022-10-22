# Semi-analytical localization for one event
import healpy as hp
import numpy as np
from matplotlib import pyplot as plt
import ligo.skymap.plot
import astropy_healpix as ah
from ligo.skymap import postprocess
from astropy import units as u
from astropy.coordinates import SkyCoord
import ctypes 
import os
import lal
import sealcore
import spiir
import time

def read_event_info(filepath):
    event_info = np.loadtxt(filepath)
    trigger_time = event_info[0]
    ndet = event_info[1]
    det_codes = np.int64(event_info[2:5])
    snrs = event_info[5:8]
    sigmas = event_info[8:]

    return trigger_time, ndet, det_codes, snrs, sigmas


def extract_info_from_xml(filepath):
    xmlfile = spiir.io.ligolw.coinc.load_coinc_xml(filepath)

    try:
        det_names = list(xmlfile['snrs'].keys())
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
    #snr_series_list = []
    #timestamps = {det: xmlfile["snrs"][det].index.values for det in det_names}
    lal_det_code = {'L1': 6, 'H1': 5, 'V1':2, 'K1': 14, 'I1': 15}  # new lal
    #ntime = len(xmlfile["snrs"][det_names[0]].index.values)
    #snr_array = np.zeros((ndet, 3, ntime ), dtype=np.dtype('d'))

    #i=0
    for det in det_names:
        deff_array = np.append(deff_array, xmlfile['tables']['postcoh']['deff_'+det])
        max_snr_array = np.append(max_snr_array, xmlfile['tables']['postcoh']['snglsnr_'+det])
        #snr_series_list.append( np.array([xmlfile["snrs"][det].index.values, np.real(xmlfile['snrs'][det]), 1*np.imag(xmlfile['snrs'][det])]).T )
        
        #snr_array[i] = np.array([xmlfile["snrs"][det].index.values, np.real(xmlfile['snrs'][det]), np.imag(xmlfile['snrs'][det])]) #.T
        snr_array = np.append(snr_array, xmlfile['snrs'][det] )
        time_array = np.append(time_array, xmlfile["snrs"][det].index.values)
        ntimes_array = np.append(ntimes_array, len(xmlfile['snrs'][det]))
        #i+=1
        det_code_array = np.append(det_code_array, int(lal_det_code[det]))
        #print("len, {}".format(len(xmlfile["snrs"][det].index.values)) )

    sigma_array = deff_array*max_snr_array

    trigger_time = xmlfile["tables"]["postcoh"]["end_time"].item()
    trigger_time += xmlfile["tables"]["postcoh"]["end_time_ns"].item() * 1e-9
    
    print("xml processing done. ")
    print(f'Trigger time: {trigger_time}')
    print(f'Detectors: {det_names}')
    print(f'SNRs: {max_snr_array}')
    print(f'sigmas: {sigma_array}')

    return trigger_time, ndet, ntimes_array.astype(ctypes.c_int32), det_code_array.astype(ctypes.c_int32), max_snr_array, sigma_array, time_array, snr_array


def generate_healpix_grids(nside):
    npix = hp.nside2npix(nside)  # equiv to 12*nside^2
    theta, phi = hp.pixelfunc.pix2ang(nside,np.arange(npix),nest=True)
    ra = phi
    dec = -theta+np.pi/2

    return ra,dec


def seal_with_adaptive_healpix(nlevel,time_arrays,snr_arrays,det_code_array,sigma_array,ntimes_array,ndet,
                                start_time, end_time, ntime_interp, prior_mu,prior_sigma):

    # Healpix: The Astrophysical Journal, 622:759–771, 2005. See its Figs. 3 & 4.
    # Adaptive healpix: see Bayestar paper (and our seal paper).
    nside_base = 16
    npix_base  = 12 * nside_base**2 # 12*16*16=3072

    nside_final = nside_base * 2**nlevel  # 16 *  [1,4,16,...,2^6]
    npix_final =12 * nside_final**2  # 12582912 for nlevel=6

    ra, dec = generate_healpix_grids(nside_final)
    skymap_multires = np.zeros(npix_final)

    # Initialize argsort
    argsort = np.arange(npix_base/4)
    argsort_pix_id = np.arange(npix_base)

    for ilevel in range(nlevel+1):
        iside = nside_base * 2**ilevel
        ipix = 12 * iside**2
        ra_for_this_level, dec_for_this_level = generate_healpix_grids(iside) # len(ra_for_this_level) = ipix
        
        # Calculate the first 1/4 most probable grids
        ra_to_calculate = ra_for_this_level[argsort_pix_id]  # len(ra_to_calculate) = npix_base
        dec_to_calculate = dec_for_this_level[argsort_pix_id]

        # Calculate skymap (of log prob density)
        coh_skymap_for_this_level = np.zeros(npix_base)
        sealcore.Pycoherent_skymap_bicorr(coh_skymap_for_this_level, time_arrays,  snr_arrays,
            det_code_array, sigma_array, ntimes_array,
            ndet, ra_to_calculate, dec_to_calculate, npix_base, 
            start_time, end_time, ntime_interp,
            prior_mu,prior_sigma)

        # Update skymap
        nfactor = 4**(nlevel-ilevel)  # map a pixel of this level to multiple pixels in the final level
        for i in range(npix_base):  # Can we avoid this loop? It's 3072 times. Runs fast though. 
            index = argsort_pix_id[i]
            skymap_multires[index*nfactor:(index+1)*nfactor] = coh_skymap_for_this_level[i]
        
        # Update argsort
        argsort = np.argsort(coh_skymap_for_this_level)[::-1][: int(npix_base/4)]
        argsort_temp = argsort_pix_id[argsort]
        for j in range(4):
            argsort_pix_id[j::4] = argsort_temp*4+j

        return skymap_multires


def plot_skymap(skymap, save_filename=None, true_ra = None, true_dec = None):
    ''' Input: log_prob_density_skymap'''
    skymap = np.exp(skymap)
    skymap /= sum(skymap)

    npix = len(skymap)
    nside = int(np.sqrt(npix/12.0))

    contour = [50,90]

    nside = ah.npix_to_nside(len(skymap))

    # Convert sky map from probability to probability per square degree.
    deg2perpix = ah.nside_to_pixel_area(nside).to_value(u.deg**2)
    probperdeg2 = skymap / deg2perpix

    #Calculate contour levels
    cls = 100 * postprocess.find_greedy_credible_levels(skymap)

    #plt.figure(figsize = (10,6))
    plt.figure()
    #Initialize skymap grid
    ax = plt.axes(projection='astro hours mollweide')
    ax.grid()
    if (true_ra is not None) and (true_dec is not None):
        ax.plot_coord(SkyCoord(true_ra, true_dec, unit='rad'), 'x', color='green', markersize=8)

    #Plot skymap with labels
    vmax = probperdeg2.max()
    vmin = probperdeg2.min()
    img = ax.imshow_hpx(probperdeg2, cmap='cylon', nested=True, vmin=vmin, vmax=vmax)
    cs = ax.contour_hpx((cls, 'ICRS'), nested=True, linewidths=0.5, levels=contour,colors='k')
    v = np.linspace(vmin, vmax, 2, endpoint=True)
    cb = plt.colorbar(img, orientation='horizontal', ticks=v, fraction=0.045)
    cb.set_label(r'probability per deg$^2$',fontsize=11)


    text=[]
    pp = np.round(contour).astype(int)
    ii = np.round(np.searchsorted(np.sort(cls), contour) *
                            deg2perpix).astype(int)
    for i, p in zip(ii, pp):
        text.append('{:d}% area: {:,d} deg²'.format(p, i))
    ax.text(1, 1, '\n'.join(text), transform=ax.transAxes, ha='right')

    if save_filename is not None:
        plt.savefig(save_filename)
        print('Skymap saved to '+ save_filename)


if __name__ == "__main__":
    #print(sealcore.pytest1(3,8))

    time0 = time.time()
    xmlfilepath = 'data/coinc_xml/H1L1V1_1187008882_3_806.xml'
    trigger_time, ndet, ntimes_array, det_code_array, max_snr_array, sigma_array, time_arrays, snr_arrays =\
        extract_info_from_xml(xmlfilepath)
    time1 = time.time()
    print("xml file processing done! Time cost {}s.".format(time1-time0))
    
    start_time = trigger_time-0.01
    end_time = trigger_time+0.01
    ntime_interp = 420

    nside = 16*16
    npix = hp.nside2npix(nside)
    theta, phi = hp.pixelfunc.pix2ang(nside,np.arange(npix),nest=True)
    ra = phi
    dec = -theta+np.pi/2

    max_net_snr = sum(ss**2 for ss in max_snr_array)**0.5
    a = 0.000466361
    b = 0.00036214
    c = 0.00032248
    d = 0.0005245
    prior_mu = a*max_net_snr + b
    prior_sigma = c*max_net_snr +d


    ########### Reload some event info because xml has some errors ###########
    trigger_time = 1187008882.43
    start_time = trigger_time-0.01
    end_time = trigger_time+0.01
    sigma_array = np.array([142*8.0,213*8.0,60*8.0])
    print("Time prior and horizon are manually corrected.")
    ########### Delete above whem xml issues are fixed ###########

    time2 = time.time()
    '''
    coh_skymap = np.zeros(npix)
    sealcore.Pycoherent_skymap_bicorr(coh_skymap, time_arrays,  snr_arrays,
        det_code_array, sigma_array, ntimes_array,
        ndet, ra, dec, npix,
        start_time, end_time, ntime_interp,
        prior_mu,prior_sigma)
    '''
    nlevel = 6
    coh_skymap_multires = seal_with_adaptive_healpix(nlevel,time_arrays,snr_arrays,det_code_array,sigma_array,ntimes_array,ndet, start_time, end_time, ntime_interp, prior_mu,prior_sigma)
    

    time3 = time.time()
    print("Skymap calculation done! Time cost {}s.".format(time3-time2))
    #print(coh_skymap_multires[0])

    plot_skymap(coh_skymap_multires, save_filename='test.pdf', true_ra = 3.446, true_dec = -0.408)
    


    