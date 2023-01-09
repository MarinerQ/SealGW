
cdef extern from "sealcore.h":
    double testfunc1(double ra, double dec, double gpstime, int detcode);

def pytest1(double ra, double dec, double gpstime , int detcode):
    return testfunc1(ra, dec, gpstime,detcode )


cdef extern from "sealcore.h":
    double et_resp_func(double ra, double dec, double gpstime, double psi, int detcode, int mode);

def Pyet_resp_func(double ra, double dec, double gpstime, double psi, int detcode, int mode):
    return et_resp_func(ra, dec, gpstime, psi, detcode, mode )


cdef extern from "sealcore.h":
    #void testdoubleseries(double *snr_array, int ndet, int ntime);
    void coherent_skymap_bicorr(
                double *coh_skymap_bicorr,
				const double *time_arrays, 
				const double complex *snr_arrays,  
				const int *detector_codes, 
				const double *sigmas, 
				const int *ntimes,
				const int Ndet,
				const double *ra_grids, 
				const double *dec_grids, 
				const int ngrid, 
				const double start_time, 
				const double end_time, 
				const int ntime_interp,
                const double prior_mu,
                const double prior_sigma,
				const int nthread);


def Pycoherent_skymap_bicorr(
                double[:] coh_skymap_bicorr,
                double[:] time_arrays, 
				double complex[:] snr_arrays,  
				int[:] detector_codes, 
				double[:] sigmas, 
				int[:] ntimes,
				int Ndet,
				double[:] ra_grids, 
				double[:] dec_grids, 
				int ngrid, 
				double start_time, 
				double end_time, 
				int ntime_interp,
                double prior_mu,
                double prior_sigma,
				int nthread): # 'arr' is a one-dimensional numpy array

    #if not coh_skymap_bicorr.flags['C_CONTIGUOUS']:
    #    coh_skymap_bicorr = np.ascontiguousarray(coh_skymap_bicorr) # Makes a contiguous copy of the numpy array.

    cdef double[:] coh_skymap_bicorr_memview = coh_skymap_bicorr

    coherent_skymap_bicorr(&coh_skymap_bicorr_memview[0], &time_arrays[0], &snr_arrays[0],  &detector_codes[0], &sigmas[0], &ntimes[0], Ndet, &ra_grids[0], &dec_grids[0], ngrid, start_time, end_time, ntime_interp, prior_mu, prior_sigma, nthread)

    return coh_skymap_bicorr
