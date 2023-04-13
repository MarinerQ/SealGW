
cdef extern from "sealcore.h":
    double testfunc1(double ra, double dec, double gpstime, int detcode);

def pytest1(double ra, double dec, double gpstime , int detcode):
    return testfunc1(ra, dec, gpstime,detcode )


cdef extern from "sealcore.h":
    double lal_resp_func(double ra, double dec, double gpstime, double psi, int detcode, int mode);

def Pylal_resp_func(double ra, double dec, double gpstime, double psi, int detcode, int mode):
    return lal_resp_func(ra, dec, gpstime, psi, detcode, mode )


cdef extern from "sealcore.h":
    double lal_dt_func(double ra, double dec, double gpstime, int detcode);

def Pylal_dt_func(double ra, double dec, double gpstime, int detcode):
    return lal_dt_func(ra, dec, gpstime, detcode )


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
				const int nthread,
				const int interp_order);


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
				int nthread,
				int interp_order): # 'arr' is a one-dimensional numpy array

    #if not coh_skymap_bicorr.flags['C_CONTIGUOUS']:
	# Makes a contiguous copy of the numpy array.
    #    coh_skymap_bicorr = np.ascontiguousarray(coh_skymap_bicorr)

    cdef double[:] coh_skymap_bicorr_memview = coh_skymap_bicorr

    coherent_skymap_bicorr(
			&coh_skymap_bicorr_memview[0],
			&time_arrays[0],
			&snr_arrays[0],
			&detector_codes[0],
			&sigmas[0],
			&ntimes[0],
			Ndet,
			&ra_grids[0],
			&dec_grids[0],
			ngrid,
			start_time,
			end_time,
			ntime_interp,
			prior_mu,
			prior_sigma,
			nthread,
			interp_order
	)

    return coh_skymap_bicorr


cdef extern from "sealcore.h":
    void coherent_skymap_bicorr_usetimediff(
                double *coh_skymap_bicorr,
				const double *time_arrays,
				const double complex *snr_arrays,
				const int *detector_codes,
				const double *sigmas,
				const int *ntimes,
				const int Ndet,
				#const double *ra_grids,
				#const double *dec_grids,
				const int *argsort_pix_id,
				const int nside,
				const int ngrid,
				const double start_time,
				const double end_time,
				const int ntime_interp,
                const double prior_mu,
                const double prior_sigma,
				const int nthread,
				const int interp_order,
				const int max_snr_det_id);


def Pycoherent_skymap_bicorr_usetimediff(
                double[:] coh_skymap_bicorr,
                double[:] time_arrays,
				double complex[:] snr_arrays,
				int[:] detector_codes,
				double[:] sigmas,
				int[:] ntimes,
				int Ndet,
				#double[:] ra_grids,
				#double[:] dec_grids,
				int[:] argsort_pix_id,
				int nside,
				int ngrid,
				double start_time,
				double end_time,
				int ntime_interp,
                double prior_mu,
                double prior_sigma,
				int nthread,
				int interp_order,
				int max_snr_det_id): # 'arr' is a one-dimensional numpy array

    #if not coh_skymap_bicorr.flags['C_CONTIGUOUS']:
	# Makes a contiguous copy of the numpy array.
    #    coh_skymap_bicorr = np.ascontiguousarray(coh_skymap_bicorr)

    cdef double[:] coh_skymap_bicorr_memview = coh_skymap_bicorr

    coherent_skymap_bicorr_usetimediff(
			&coh_skymap_bicorr_memview[0],
			&time_arrays[0],
			&snr_arrays[0],
			&detector_codes[0],
			&sigmas[0],
			&ntimes[0],
			Ndet,
			#&ra_grids[0],
			#&dec_grids[0],
			&argsort_pix_id[0],
			nside,
			ngrid,
			start_time,
			end_time,
			ntime_interp,
			prior_mu,
			prior_sigma,
			nthread,
			interp_order,
			max_snr_det_id
	)

    return coh_skymap_bicorr



cdef extern from "sealcore.h":
    void coherent_skymap_multires_bicorr(
		double *coh_skymap_multires_bicorr,
		const double *time_arrays,
		const double complex *snr_arrays,
		const int *detector_codes,
		const double *sigmas,
		const int *ntimes,
		const int Ndet,
		const double start_time,
		const double end_time,
		const int ntime_interp,
		const double prior_mu,
		const double prior_sigma,
		const int nthread,
		const int interp_order,
		const int max_snr_det_id,
		const int nlevel,
		const int use_timediff);


def Pycoherent_skymap_multires_bicorr(
                double[:] coh_skymap_multires_bicorr,
                double[:] time_arrays,
				double complex[:] snr_arrays,
				int[:] detector_codes,
				double[:] sigmas,
				int[:] ntimes,
				int Ndet,
				double start_time,
				double end_time,
				int ntime_interp,
                double prior_mu,
                double prior_sigma,
				int nthread,
				int interp_order,
				int max_snr_det_id,
				const int nlevel,
				const int use_timediff): # 'arr' is a one-dimensional numpy array

    #if not coh_skymap_bicorr.flags['C_CONTIGUOUS']:
	# Makes a contiguous copy of the numpy array.
    #    coh_skymap_bicorr = np.ascontiguousarray(coh_skymap_bicorr)

    cdef double[:] coh_skymap_multires_bicorr_memview = coh_skymap_multires_bicorr

    coherent_skymap_multires_bicorr(
			&coh_skymap_multires_bicorr_memview[0],
			&time_arrays[0],
			&snr_arrays[0],
			&detector_codes[0],
			&sigmas[0],
			&ntimes[0],
			Ndet,
			start_time,
			end_time,
			ntime_interp,
			prior_mu,
			prior_sigma,
			nthread,
			interp_order,
			max_snr_det_id,
			nlevel,
			use_timediff
	)

    return coh_skymap_multires_bicorr
