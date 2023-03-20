#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <string.h>
#include <time.h>

#include <gsl/gsl_matrix.h> // /apps/skylake/software/compiler/gcc/6.4.0/gsl/2.4/include/
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>

#include <lal/LALDetectors.h> // /fred/oz016/opt-pipe/include/
#include <lal/LALSimulation.h>
#include <lal/TimeDelay.h>
#include <lal/LALDatatypes.h>
#include <lal/Date.h>
#include <lal/Units.h>

COMPLEX16TimeSeries * 	XLALCreateCOMPLEX16TimeSeries (const CHAR *name, const LIGOTimeGPS *epoch, REAL8 f0, REAL8 deltaT, const LALUnit *sampleUnits, size_t length);

void XLALDestroyCOMPLEX16TimeSeries (COMPLEX16TimeSeries *series);

//cc -fPIC -shared -o sealcore.so sealcore.c -llal -lgsl

static double max_in_4(double loga, double logb, double logc, double logd){
	double temp = loga;
	if(temp<logb) temp = logb;
	if(temp<logc) temp = logc;
	if(temp<logd) temp = logd;
	return temp;
}

static double logsumexp(double loga, double logb)
{
	if(loga>logb){
		if(loga>logb+40) return loga;
		else return loga + log(1+exp(logb-loga));
	}
	else{
		if(logb>loga+40) return logb;
		else return logb + log(1+exp(loga-logb));
	}
}

static double logsumexp4(double loga, double logb, double logc, double logd)
{
	double max_log = max_in_4(loga,logb,logc,logd);

	if(max_log == loga){
		return loga + log(1 + exp(logb-loga) + exp(logc-loga) + exp(logd-loga));
	}
	else if (max_log == logb){
		return logb + log(1 + exp(loga-logb) + exp(logc-logb) + exp(logd-logb));
	}
	else if (max_log == logc){
		return logc + log(1 + exp(loga-logc) + exp(logb-logc) + exp(logd-logc));
	}
	else if (max_log == logd){
		return logd + log(1 + exp(loga-logd) + exp(logb-logd) + exp(logc-logd));
	}
}

static void gsl_matrix_mult(const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *C)
{
	unsigned long i,j,k;
	double data=0;

	if(A->size2 != B->size1){
		printf("A and B do not fit! \n");
		exit(-2);
	}
	else if(C->size1!=A->size1 || C->size2!=B->size2){
		printf("C doesn't fit with A or B! \n");
		exit(-2);
	}

	for(i=0;i<A->size1;i++){
		for(k=0;k<B->size2;k++){
			data=0;
			for(j=0;j<A->size2;j++){
				data += gsl_matrix_get(A,i,j)*gsl_matrix_get(B,j,k);
			}
			gsl_matrix_set(C,i,k,data);
		}
	}
}

static double quadratic_form(double a1, double a2, double m11, double m12, double m21, double m22)
{
	double temp1,temp2,temp3;
	temp1 = m11*a1*a1;
	temp2 = (m12 + m21)*a1*a2;
	temp3 = m22*a2*a2;
	return temp1+temp2+temp3;
}

double complex interpolate_time_series(COMPLEX16TimeSeries *lal_array, double time, int interp_order){
	double start_time = lal_array->epoch.gpsSeconds + lal_array->epoch.gpsNanoSeconds*1E-9;
	double end_time   = start_time + lal_array->data->length * lal_array->deltaT;
	double deltat = lal_array->deltaT;

	if(time < start_time){
		printf("interpolate time < start time of time series!\n");
		exit(-1);
	}
	else if(time > end_time){
		printf("interpolate time > end time of time series!\n");
		exit(-1);
	}

	int index = (int)((time-start_time)/deltat);
	double diff = (time-start_time)/deltat-(double)index;
	double complex int_data;
	double real, imag;

	if (interp_order==0)
	{ // step interpolation
		if(diff<0.5)
		{
			real = creal(lal_array->data->data[index]);
			imag = cimag(lal_array->data->data[index]);
			int_data = real+I*imag;
	}
		else
		{
			real = creal(lal_array->data->data[index+1]);
			imag = cimag(lal_array->data->data[index+1]);
			int_data = real+I*imag;
		}
	}
	else if(interp_order == 1)
	{ // linear interpolation
		real = creal(lal_array->data->data[index])*(1-diff) + creal(lal_array->data->data[index+1])*diff;
		imag = cimag(lal_array->data->data[index])*(1-diff) + cimag(lal_array->data->data[index+1])*diff;
		int_data = real+I*imag;
	}
	else if(interp_order == 2)
	{ // quadratic interpolation
		double complex y_1,y_2,y_3;
		double x;

		if(diff<0.5)
		{
			x = diff;
			y_1 = lal_array->data->data[index-1];
			y_2 = lal_array->data->data[index];
			y_3 = lal_array->data->data[index+1];
		}
		else
		{
			x = 1 - diff;
			y_1 = lal_array->data->data[index];
			y_2 = lal_array->data->data[index+1];
			y_3 = lal_array->data->data[index+2];
		}

		real = creal(y_1)*x*(x-1.0)/(2.0) + creal(y_2)*(x+1.0)*(x-1.0)/(-1.0) + creal(y_3)*(x+1)*x/(2.0);
		imag = cimag(y_1)*x*(x-1.0)/(2.0) + cimag(y_2)*(x+1.0)*(x-1.0)/(-1.0) + cimag(y_3)*(x+1)*x/(2.0);
		int_data = real+I*imag;
	}
	else{
		printf("Wrong interp order!\n");
		exit(-1);
	}

	return int_data;
}



static void ComputeDetAMResponse(
		double *fplus,          /**< Returned value of F+ */
		double *fcross,         /**< Returned value of Fx */
		const REAL4 D[3][3],    /**< Detector response 3x3 matrix */
		const double ra,        /**< Right ascention of source (radians) */
		const double dec,       /**< Declination of source (radians) */
		const double psi,       /**< Polarization angle of source (radians) */
		const double gmst       /**< Greenwich mean sidereal time (radians) */
		)
{
	int i;
	double X[3];
	double Y[3];

	/* Greenwich hour angle of source (radians). */
	const double gha = gmst - ra;

	/* pre-compute trig functions */
	const double cosgha = cos(gha);
	const double singha = sin(gha);
	const double cosdec = cos(dec);
	const double sindec = sin(dec);
	const double cospsi = cos(psi);
	const double sinpsi = sin(psi);

	/* Eq. (B4) of [ABCF].  Note that dec = pi/2 - theta, and gha =
	 * -phi where theta and phi are the standard spherical coordinates
	 * used in that paper. */
	X[0] = -cospsi * singha - sinpsi * cosgha * sindec;
	X[1] = -cospsi * cosgha + sinpsi * singha * sindec;
	X[2] =  sinpsi * cosdec;

	/* Eq. (B5) of [ABCF].  Note that dec = pi/2 - theta, and gha =
	 * -phi where theta and phi are the standard spherical coordinates
	 * used in that paper. */
	Y[0] =  sinpsi * singha - cospsi * cosgha * sindec;
	Y[1] =  sinpsi * cosgha + cospsi * singha * sindec;
	Y[2] =  cospsi * cosdec;

	/* Now compute Eq. (B7) of [ABCF] for each polarization state, i.e.,
	 * with s+=1 and sx=0 to get F+, with s+=0 and sx=1 to get Fx */
	*fplus = *fcross = 0.0;
	for(i = 0; i < 3; i++) {
		const double DX = D[i][0] * X[0] + D[i][1] * X[1] + D[i][2] * X[2];
		const double DY = D[i][0] * Y[0] + D[i][1] * Y[1] + D[i][2] * Y[2];
		*fplus  += X[i] * DX - Y[i] * DY;
		*fcross += X[i] * DY + Y[i] * DX;
	}
}

static void getGpc(const LALDetector detector, double ra, double de, double gps_time, double *Gpc)
{
	double fplus,fcross,gmst;
	LIGOTimeGPS gps_time_ligo;
	gps_time_ligo.gpsSeconds = (int)gps_time;
	gps_time_ligo.gpsNanoSeconds = (int)(gps_time-(int)gps_time)*1000000000;//a probable mistake here
	gmst = XLALGreenwichMeanSiderealTime(&gps_time_ligo);
	ComputeDetAMResponse(&fplus,&fcross,detector.response,ra,de,0.0,gmst);//psi = 0

	if(Gpc==NULL){
		printf("Gpc is a NULL pointer\n");
		return;
	}
	else{
		Gpc[0] = fplus;
		Gpc[1] = fcross;
	}
}

static void getGsigma_matrix(const LALDetector *detectors,const double *sigma, int Ndet, double ra, double dec, double gps_time,double *Gsigma)
{
	int i;
	double Gpc[2];

	if(detectors==NULL){
		printf("You need to create the detector first! \n");
	}
	else if(sigma==NULL){
		printf("You need to assign the value to sigma! \n");
	}

	for(i=0;i<Ndet;i++){
		getGpc(detectors[i],ra,dec,gps_time,Gpc);
		Gsigma[i*2]   = Gpc[0]*sigma[i];
		Gsigma[i*2+1] = Gpc[1]*sigma[i];
	}

}

double et_resp_func(double ra, double dec, double gpstime, double psi, int detcode, int mode){
	/*
	Bilby has different geometry for ET in bilby and LAL. This function calculate ET's response functions in LAL.
	*/
	LALDetector tempdet;
	double fplus,fcross,gmst;
	LIGOTimeGPS gps_time_ligo;
	gps_time_ligo.gpsSeconds = (int)gpstime;
	gps_time_ligo.gpsNanoSeconds = (int)(gpstime-(int)gpstime)*1000000000;
	gmst = XLALGreenwichMeanSiderealTime(&gps_time_ligo);

	tempdet = lalCachedDetectors[detcode];
	ComputeDetAMResponse(&fplus,&fcross,tempdet.response,ra,dec,psi,gmst);

	if(mode==0)
	{
		return fplus;
	}
	else if (mode==1)
	{
		return fcross;
	}

}

double testfunc1(double ra, double dec, double gpstime, int detcode){
	int det_id, Ndet=3;
    //int detector_codes[3] = {16, 17, 18};
	LALDetector tempdet;
	double fplus,fcross,gmst;
	LIGOTimeGPS gps_time_ligo;
	gps_time_ligo.gpsSeconds = (int)gpstime;
	gps_time_ligo.gpsNanoSeconds = (int)(gpstime-(int)gpstime)*1000000000;
	gmst = XLALGreenwichMeanSiderealTime(&gps_time_ligo);

	tempdet = lalCachedDetectors[detcode];
	ComputeDetAMResponse(&fplus,&fcross,tempdet.response,ra,dec,0.0,gmst);
	return fplus;
	/*
    for(det_id=0; det_id<Ndet; det_id++){
		tempdet = lalCachedDetectors[detector_codes[det_id]];
        ComputeDetAMResponse(&fplus,&fcross,tempdet.response,ra,dec,0.0,gmst);
        printf("ET%d, ra=%f, dec=%f, gmst=%f\n", detector_codes[det_id]-15, ra, dec, gmst);
        printf("fp=%f, fc=%f\n",fplus, fcross );
	}*/
	//return 0.0;
}


COMPLEX16TimeSeries ** CreateCOMPLEX16TimeSeriesList(const double *time_arrays, const double complex *snr_arrays, const int ndet, const int *ntimes){

	int i,ntime,previous_ntime;
	LIGOTimeGPS epoch;
    COMPLEX16TimeSeries **lalsnr_array = (COMPLEX16TimeSeries**)malloc(sizeof(COMPLEX16TimeSeries*)*ndet);
	double start_time, deltat;

	previous_ntime=0;
	for ( i = 0; i < ndet; i++)
	{
		ntime = ntimes[i]; // allow different numbers of data samples for different detectors
		start_time = time_arrays[previous_ntime];
		epoch.gpsSeconds = (int)start_time;
		epoch.gpsNanoSeconds = (start_time-epoch.gpsSeconds)*1E9;
		deltat = time_arrays[previous_ntime+1] - time_arrays[previous_ntime]; // allow different sampling rates for different detectors
		lalsnr_array[i] = XLALCreateCOMPLEX16TimeSeries("",&epoch,0.0,deltat,&lalDimensionlessUnit,ntime);
		lalsnr_array[i]->data->data = (COMPLEX16 *) &snr_arrays[previous_ntime];
		lalsnr_array[i]->data->length = ntime;
		previous_ntime += ntime;
		//printf("%d-th lal series created.\n", i);
	}
	return lalsnr_array;
}

void DestroyCOMPLEX16TimeSeriesList(COMPLEX16TimeSeries **lalsnr_array, int ndet){
	int i;
	for ( i = 0; i > 1; i++)  // only need to free the first one??
	{
		//printf("destroy %d-th lal series\n", i);
		XLALDestroyCOMPLEX16TimeSeries(lalsnr_array[i]);
		//printf("destroy lal series done\n");
	}
}

/*
Coherent localization skymap with bimodal correlated-digonal prior.
See arXiv:2110.01874.
*/
void coherent_skymap_bicorr(
				double *coh_skymap_bicorr, // The probability skymap we want to return
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
				const int interp_order)
{
	int grid_id,time_id,det_id;

	double dt = (end_time-start_time)/ntime_interp;
	double ref_gps_time = (start_time + end_time)/2.0;

	LALDetector tempdet, detectors[Ndet];
	for(det_id=0; det_id<Ndet; det_id++){
		tempdet = lalCachedDetectors[detector_codes[det_id]];
		detectors[det_id] = tempdet;
	}
	COMPLEX16TimeSeries ** snr_list = CreateCOMPLEX16TimeSeriesList(time_arrays, snr_arrays, Ndet, ntimes);

	LIGOTimeGPS ligo_gps_time;
	ligo_gps_time.gpsSeconds = (int)(ref_gps_time);
	ligo_gps_time.gpsNanoSeconds = (ref_gps_time-(int)(ref_gps_time)) * 1E9;


	double mu_multimodal = prior_mu;
	double sigma_multimodal = prior_sigma;
	double xi = 1/sigma_multimodal/sigma_multimodal;
	double alpha = mu_multimodal*xi;

	clock_t t;
	double time_taken;
	#pragma omp parallel num_threads(nthread) private(time_id,det_id)  shared(coh_skymap_bicorr,snr_list,detectors)
	{
	#pragma omp for
	for(grid_id=0;grid_id<ngrid;grid_id+=1){

    	t = clock();
		double Gsigma[2*Ndet];

		//gsl_matrix *detector_real_streams = gsl_matrix_calloc(ntime_interp,Ndet);
		//gsl_matrix *detector_imag_streams = gsl_matrix_calloc(ntime_interp,Ndet);

		gsl_matrix *M_prime = gsl_matrix_alloc(2,2);

		gsl_matrix *G_sigma = gsl_matrix_alloc(Ndet,2); //same as previouly defined
		gsl_matrix *G_sigma_transpose = gsl_matrix_alloc(2,Ndet);

		gsl_matrix *J_real_streams   = gsl_matrix_calloc(ntime_interp,2);
		gsl_matrix *J_imag_streams   = gsl_matrix_calloc(ntime_interp,2);

		t = clock() - t;
    	time_taken = ((double)t)/CLOCKS_PER_SEC;;
		if (grid_id==0){
			printf("Time cost of allocating: %f\n", time_taken);
		}


		//set parameters
		double ra  = ra_grids[grid_id];
		double dec = dec_grids[grid_id];

		/*
		//time shift the data
		for(det_id=0;det_id<Ndet;det_id++){
			double time_shift = XLALTimeDelayFromEarthCenter((detectors[det_id]).location,ra,dec,&ligo_gps_time);

			for(time_id=0;time_id<ntime_interp;time_id++){
				double complex data = interpolate_time_series(snr_list[det_id], start_time + time_id*dt + time_shift, interp_order);
				gsl_matrix_set(detector_real_streams,time_id,det_id,creal(data));
				gsl_matrix_set(detector_imag_streams,time_id,det_id,cimag(data));
			}


			// Use 2x Loop unrooling
			for(time_id=0;time_id<=ntime_interp/2;time_id+=2){
				double complex data0 = interpolate_time_series(snr_list[det_id], start_time + time_id*dt + time_shift, interp_order);
				gsl_matrix_set(detector_real_streams,time_id,det_id,creal(data0));
				gsl_matrix_set(detector_imag_streams,time_id,det_id,cimag(data0));

				double complex data1 = interpolate_time_series(snr_list[det_id], start_time + (ntime_interp-time_id-1)*dt + time_shift, interp_order);
				gsl_matrix_set(detector_real_streams,ntime_interp-time_id-1,det_id,creal(data1));
				gsl_matrix_set(detector_imag_streams,ntime_interp-time_id-1,det_id,cimag(data1));
			}
		}*/

		t = clock() - t;
    	time_taken = ((double)t)/CLOCKS_PER_SEC;;
		if (grid_id==0){
			printf("Time cost of shifting data: %f\n", time_taken);
		}

		getGsigma_matrix(detectors,sigmas,Ndet,ra,dec,ref_gps_time,Gsigma);
		//Calculate M
		double temp_element;
		int ii,jj;
		for(ii=0;ii<Ndet;ii++){
			for(jj=0;jj<2;jj++){
				temp_element = Gsigma[2*ii+jj];
				gsl_matrix_set(G_sigma,ii,jj,temp_element);
				gsl_matrix_set(G_sigma_transpose,jj,ii,temp_element);
			}
		}
		gsl_matrix_mult(G_sigma_transpose,G_sigma,M_prime); //M_prime here is actually M

		//Calculate M'^{-1} and M0'^{-1}
        double M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22;
        double aa = gsl_matrix_get(M_prime,0,0) + gsl_matrix_get(M_prime,1,1) + xi;
        double bb = 2*gsl_matrix_get(M_prime,0,1);
        double cc = 2*gsl_matrix_get(M_prime,1,0);
        double dd = gsl_matrix_get(M_prime,1,1) + gsl_matrix_get(M_prime,0,0) + xi;
        double detMprime = aa*dd - bb*cc;
        M_inverse_11 = dd/(aa*dd-bb*cc);
        M_inverse_12 = -cc/(aa*dd-bb*cc);
        M_inverse_21 = -bb/(aa*dd-bb*cc);
        M_inverse_22 = aa/(aa*dd-bb*cc);

        double M0_inverse_11,M0_inverse_12,M0_inverse_21,M0_inverse_22;
        double aa0 = gsl_matrix_get(M_prime,0,0) + gsl_matrix_get(M_prime,1,1) + xi;
        double dd0 = gsl_matrix_get(M_prime,1,1) + gsl_matrix_get(M_prime,0,0)+ xi;
        double detM0prime = aa0*dd0;
        M0_inverse_11 = 1.0/aa0;
        M0_inverse_12 = 0.0;
        M0_inverse_21 = 0.0;
        M0_inverse_22 = 1.0/dd0;

		t = clock() - t;
    	time_taken = ((double)t)/CLOCKS_PER_SEC;;
		if (grid_id==0){
			printf("Time cost of setting M: %f\n", time_taken);
		}

		//transform mf data to j stream
		double time_shifts[Ndet];
		double time_shift;
		double complex data;
		for(det_id=0;det_id<Ndet;det_id++){
			time_shifts[det_id] = XLALTimeDelayFromEarthCenter((detectors[det_id]).location,ra,dec,&ligo_gps_time);
		}

		for(time_id=0;time_id<ntime_interp;time_id++){
			double temp0_real=0;
			double temp0_imag=0;
			double temp1_real=0;
			double temp1_imag=0;
			for(det_id=0;det_id<Ndet;det_id++){
				time_shift = time_shifts[det_id];
				data = interpolate_time_series(snr_list[det_id], start_time + time_id*dt + time_shift, interp_order);

				temp0_real += creal(data)*gsl_matrix_get(G_sigma,det_id,0);
				temp0_imag += cimag(data)*gsl_matrix_get(G_sigma,det_id,0);
				temp1_real += creal(data)*gsl_matrix_get(G_sigma,det_id,1);
				temp1_imag += cimag(data)*gsl_matrix_get(G_sigma,det_id,1);
			}
			gsl_matrix_set(J_real_streams,time_id,0,temp0_real);
			gsl_matrix_set(J_imag_streams,time_id,0,temp0_imag);
			gsl_matrix_set(J_real_streams,time_id,1,temp1_real);
			gsl_matrix_set(J_imag_streams,time_id,1,temp1_imag);
		}


		/*for (time_id = 0; time_id <= ntime_interp/2; time_id += 2) {
			// Use 2x loop unrolling
			double temp0_real_0 = 0, temp0_imag_0 = 0, temp1_real_0 = 0, temp1_imag_0 = 0;
			double temp0_real_1 = 0, temp0_imag_1 = 0, temp1_real_1 = 0, temp1_imag_1 = 0;

			for (det_id = 0; det_id < Ndet; det_id++) {
				temp0_real_0 += gsl_matrix_get(detector_real_streams, time_id, det_id) * gsl_matrix_get(G_sigma, det_id, 0);
				temp0_imag_0 += gsl_matrix_get(detector_imag_streams, time_id, det_id) * gsl_matrix_get(G_sigma, det_id, 0);
				temp1_real_0 += gsl_matrix_get(detector_real_streams, time_id, det_id) * gsl_matrix_get(G_sigma, det_id, 1);
				temp1_imag_0 += gsl_matrix_get(detector_imag_streams, time_id, det_id) * gsl_matrix_get(G_sigma, det_id, 1);

				temp0_real_1 += gsl_matrix_get(detector_real_streams, ntime_interp-time_id-1, det_id) * gsl_matrix_get(G_sigma, det_id, 0);
				temp0_imag_1 += gsl_matrix_get(detector_imag_streams, ntime_interp-time_id-1, det_id) * gsl_matrix_get(G_sigma, det_id, 0);
				temp1_real_1 += gsl_matrix_get(detector_real_streams, ntime_interp-time_id-1, det_id) * gsl_matrix_get(G_sigma, det_id, 1);
				temp1_imag_1 += gsl_matrix_get(detector_imag_streams, ntime_interp-time_id-1, det_id) * gsl_matrix_get(G_sigma, det_id, 1);
			}
			gsl_matrix_set(J_real_streams,time_id,0,temp0_real_0);
			gsl_matrix_set(J_imag_streams,time_id,0,temp0_imag_0);
			gsl_matrix_set(J_real_streams,time_id,1,temp1_real_0);
			gsl_matrix_set(J_imag_streams,time_id,1,temp1_imag_0);

			gsl_matrix_set(J_real_streams,ntime_interp-time_id-1,0,temp0_real_1);
			gsl_matrix_set(J_imag_streams,ntime_interp-time_id-1,0,temp0_imag_1);
			gsl_matrix_set(J_real_streams,ntime_interp-time_id-1,1,temp1_real_1);
			gsl_matrix_set(J_imag_streams,ntime_interp-time_id-1,1,temp1_imag_1);
		}*/


		t = clock() - t;
    	time_taken = ((double)t)/CLOCKS_PER_SEC;;
		if (grid_id==0){
			printf("Time cost of setting J: %f\n", time_taken);
		}

		//calculate skymap
		//double snr_temp;
		double log_exp_term1,log_exp_term2,log_exp_term,log_exp_term3,log_exp_term4;
		double j_r1,j_r2,j_i1,j_i2;
		double log_prob_margT_bicorr=-100;
		double prefactor = log(detMprime);
		double prefactor0 = log(detM0prime);

		coh_skymap_bicorr[grid_id]=0;

		// numerical time marginalization
		for(time_id=0;time_id<ntime_interp;time_id++){
			j_r1 = gsl_matrix_get(J_real_streams,time_id,0);
			j_r2 = gsl_matrix_get(J_real_streams,time_id,1);
			j_i1 = gsl_matrix_get(J_imag_streams,time_id,0);
			j_i2 = gsl_matrix_get(J_imag_streams,time_id,1);

			// avoid big number, use log
			log_exp_term1 = logsumexp4(
				quadratic_form(j_r1+j_i2+alpha,j_r2+j_i1+alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2,
				quadratic_form(j_r1+j_i2-alpha,j_r2+j_i1+alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2,
				quadratic_form(j_r1+j_i2+alpha,j_r2+j_i1-alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2,
				quadratic_form(j_r1+j_i2-alpha,j_r2+j_i1-alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2
			) - prefactor/2;


			log_exp_term2 = logsumexp4(
				quadratic_form(j_r1+j_i2+alpha,j_r2-j_i1+alpha,M0_inverse_11,M0_inverse_12,M0_inverse_21,M0_inverse_22)/2,
				quadratic_form(j_r1+j_i2-alpha,j_r2-j_i1+alpha,M0_inverse_11,M0_inverse_12,M0_inverse_21,M0_inverse_22)/2,
				quadratic_form(j_r1+j_i2+alpha,j_r2-j_i1-alpha,M0_inverse_11,M0_inverse_12,M0_inverse_21,M0_inverse_22)/2,
				quadratic_form(j_r1+j_i2-alpha,j_r2-j_i1-alpha,M0_inverse_11,M0_inverse_12,M0_inverse_21,M0_inverse_22)/2
			) - prefactor0/2;

			log_exp_term3 = logsumexp4(
				quadratic_form(j_r1-j_i2+alpha,j_r2+j_i1+alpha,M0_inverse_11,M0_inverse_12,M0_inverse_21,M0_inverse_22)/2,
				quadratic_form(j_r1-j_i2-alpha,j_r2+j_i1+alpha,M0_inverse_11,M0_inverse_12,M0_inverse_21,M0_inverse_22)/2,
				quadratic_form(j_r1-j_i2+alpha,j_r2+j_i1-alpha,M0_inverse_11,M0_inverse_12,M0_inverse_21,M0_inverse_22)/2,
				quadratic_form(j_r1-j_i2-alpha,j_r2+j_i1-alpha,M0_inverse_11,M0_inverse_12,M0_inverse_21,M0_inverse_22)/2
			) - prefactor0/2;

			log_exp_term4 = logsumexp4(
				quadratic_form(j_r1-j_i2+alpha,j_r2-j_i1+alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2,
				quadratic_form(j_r1-j_i2-alpha,j_r2-j_i1+alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2,
				quadratic_form(j_r1-j_i2+alpha,j_r2-j_i1-alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2,
				quadratic_form(j_r1-j_i2-alpha,j_r2-j_i1-alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2
			) - prefactor/2;

			log_exp_term = logsumexp4(log_exp_term1,log_exp_term2,log_exp_term3,log_exp_term4);
			log_prob_margT_bicorr = logsumexp(log_prob_margT_bicorr,log_exp_term);
		}

		t = clock() - t;
    	time_taken = ((double)t)/CLOCKS_PER_SEC;;
		if (grid_id==0){
			printf("Time cost of time marginalization: %f\n", time_taken);
		}

		coh_skymap_bicorr[grid_id] = log_prob_margT_bicorr;

		// clean
		//gsl_matrix_free(detector_real_streams);
		//gsl_matrix_free(detector_imag_streams);
		gsl_matrix_free(M_prime);
		gsl_matrix_free(G_sigma);
		gsl_matrix_free(G_sigma_transpose);
		gsl_matrix_free(J_real_streams);
		gsl_matrix_free(J_imag_streams);


	} // end of for(grid_id)
	} // end of omp

	DestroyCOMPLEX16TimeSeriesList(snr_list,Ndet);

}
