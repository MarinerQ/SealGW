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
#include <lal/DetResponse.h>

#include <chealpix.h>
#include <exponential_integral_Ei.h>

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
    return m11 * a1 * a1 + (m12 + m21) * a1 * a2 + m22 * a2 * a2;
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

	if (interp_order==0)
	{ // step interpolation
		int_data = lal_array->data->data[diff < 0.5 ? index : index + 1];
	}
	else if(interp_order == 1)
	{ // linear interpolation
		double real = creal(lal_array->data->data[index])*(1-diff) + creal(lal_array->data->data[index+1])*diff;
		double imag = cimag(lal_array->data->data[index])*(1-diff) + cimag(lal_array->data->data[index+1])*diff;
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

		double real = creal(y_1)*x*(x-1.0)/(2.0) + creal(y_2)*(x+1.0)*(x-1.0)/(-1.0) + creal(y_3)*(x+1)*x/(2.0);
		double imag = cimag(y_1)*x*(x-1.0)/(2.0) + cimag(y_2)*(x+1.0)*(x-1.0)/(-1.0) + cimag(y_3)*(x+1)*x/(2.0);
		int_data = real+I*imag;
	}
	else{
		printf("Wrong interp order!\n");
		exit(-1);
	}

	return int_data;
}


/*
Copied from https://lscsoft.docs.ligo.org/lalsuite/lal/_det_response_8c_source.html#l00044
for test use.
*/
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
	gps_time_ligo.gpsNanoSeconds = (int)(gps_time-(int)gps_time)*1000000000;
	gmst = XLALGreenwichMeanSiderealTime(&gps_time_ligo);
	//ComputeDetAMResponse(&fplus,&fcross,detector.response,ra,de,0.0,gmst);//psi = 0
	XLALComputeDetAMResponse(&fplus,&fcross,detector.response,ra,de,0.0,gmst);

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

	for(i=0;i<Ndet;i++){
		getGpc(detectors[i],ra,dec,gps_time,Gpc);
		Gsigma[i*2]   = Gpc[0]*sigma[i];
		Gsigma[i*2+1] = Gpc[1]*sigma[i];
	}

}

static void calcM(const double *Gsigma, int Ndet, double *M){
	/* M = Gsigma^T * Gsigma, where Gsigma is a Ndetx2 matrix, M is a 2x2 matrix. */
	int i,j,k;
	double sum;

	for(i=0; i<2; i++){
		for(j=0; j<2; j++)
		{
			sum=0;
			for(k=0; k<Ndet; k++){
				sum += Gsigma[k*2+i]*Gsigma[k*2+j];
			}
			M[2*i+j] = sum;
		}
	}
}

double calcExptermBiCorr(double j_r1, double j_r2,double j_i1, double j_i2,
				  double alpha, double prefactor, double prefactor0,
				  double M_inverse_11, double M_inverse_12, double M_inverse_21, double M_inverse_22,
				  double M0_inverse_11, double M0_inverse_12, double M0_inverse_21, double M0_inverse_22)
{
	double log_exp_term, log_exp_term1,log_exp_term2,log_exp_term3,log_exp_term4;

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

	return log_exp_term;
}

double calcExptermBi(double j_r1, double j_r2,double j_i1, double j_i2,
				  double alpha, double prefactor,
				  double M_inverse_11, double M_inverse_12, double M_inverse_21, double M_inverse_22)
{
	double log_exp_term, log_exp_term1,log_exp_term2;

	log_exp_term1 = logsumexp4(
		quadratic_form(j_r1+alpha,j_r2+alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2,
		quadratic_form(j_r1-alpha,j_r2+alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2,
		quadratic_form(j_r1+alpha,j_r2-alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2,
		quadratic_form(j_r1-alpha,j_r2-alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2
	);


	log_exp_term2 = logsumexp4(
		quadratic_form(j_i1+alpha,j_i2+alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2,
		quadratic_form(j_i1-alpha,j_i2+alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2,
		quadratic_form(j_i1+alpha,j_i2-alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2,
		quadratic_form(j_i1-alpha,j_i2-alpha,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2
	);

	//log_exp_term = logsumexp(log_exp_term1,log_exp_term2);
	log_exp_term = log_exp_term1 + log_exp_term2 - prefactor;

	return log_exp_term;
}

double calcExptermFlat(double j_r1, double j_r2,double j_i1, double j_i2,
				  double prefactor,
				  double M_inverse_11, double M_inverse_12, double M_inverse_21, double M_inverse_22)
{
	double log_exp_term, log_exp_term1, log_exp_term2;

	log_exp_term1 = quadratic_form(j_r1,j_r2,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2;
	log_exp_term2 = quadratic_form(j_i1,j_i2,M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22)/2;
	log_exp_term = log_exp_term1 + log_exp_term2 - prefactor;

	return log_exp_term;
}


double lal_resp_func(double ra, double dec, double gpstime, double psi, int detcode, int mode){
	/*
	Bilby has different geometry for ET&CE from LAL. This function calculate response functions in LAL.
	*/
	LALDetector tempdet;
	double fplus,fcross,gmst;
	LIGOTimeGPS gps_time_ligo;
	gps_time_ligo.gpsSeconds = (int)gpstime;
	gps_time_ligo.gpsNanoSeconds = (int)(gpstime-(int)gpstime)*1000000000;
	gmst = XLALGreenwichMeanSiderealTime(&gps_time_ligo);

	tempdet = lalCachedDetectors[detcode];
	//ComputeDetAMResponse(&fplus,&fcross,tempdet.response,ra,dec,psi,gmst);
	XLALComputeDetAMResponse(&fplus,&fcross,tempdet.response,ra,dec,psi,gmst);

	if(mode==0)
	{
		return fplus;
	}
	else if (mode==1)
	{
		return fcross;
	}

}

double lal_dt_func(double ra, double dec, double gpstime, int detcode){
	/*
	Bilby has different geometry for ET&CE from LAL. This function calculate time delay from geocenter in LAL.
	*/
	LALDetector tempdet;
	tempdet = lalCachedDetectors[detcode];

	LIGOTimeGPS gps_time_ligo;
	gps_time_ligo.gpsSeconds = (int)gpstime;
	gps_time_ligo.gpsNanoSeconds = (int)(gpstime-(int)gpstime)*1000000000;

	double dt;
	dt = XLALTimeDelayFromEarthCenter(tempdet.location,ra,dec,&gps_time_ligo);
	return dt;
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
This function uses the max-snr detector's timestamp as the time parameter to be marginalized,
rather than geocent time tc. This is more robust in real detection in which true tc is unknown.
max_snr_det_id is used to label the detector - from 0,1,2, rather than LAL det code.
This function is for internal use. To call this in python, use coherent_skymap_bicorr_usetimediff.
*/
void _coherent_skymap_bicorr(
				double *coh_skymap_bicorr, // The probability skymap we want to return
				const double *time_arrays,
				COMPLEX16TimeSeries **snr_list,
				LALDetector *detectors,
				const double *sigmas,
				const int *ntimes,
				const int Ndet,
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
				const int max_snr_det_id,
				const int use_timediff,
				const double premerger_time)
{
	int grid_id,time_id,det_id;

	double dt = (end_time-start_time)/ntime_interp;
	double ref_gps_time = (start_time + end_time)/2.0 - premerger_time;

	LIGOTimeGPS ligo_gps_time;
	ligo_gps_time.gpsSeconds = (int)(ref_gps_time);
	ligo_gps_time.gpsNanoSeconds = (ref_gps_time-(int)(ref_gps_time)) * 1E9;


	double mu_multimodal = prior_mu;
	double sigma_multimodal = prior_sigma;
	double xi = 1/sigma_multimodal/sigma_multimodal;
	double alpha = mu_multimodal*xi;
	#pragma omp parallel num_threads(nthread) private(time_id,det_id)  shared(coh_skymap_bicorr,snr_list,detectors)
	{
	#pragma omp for
	for(grid_id=0;grid_id<ngrid;grid_id+=1){
		coh_skymap_bicorr[grid_id]=0;

		double Gsigma[2*Ndet];
		double M[4];

		double ra, dec;
		pix2ang_nest64(nside, argsort_pix_id[grid_id], &dec, &ra);
		dec = M_PI/2 - dec;

		getGsigma_matrix(detectors,sigmas,Ndet,ra,dec,ref_gps_time,Gsigma);
		//Calculate M
		calcM(Gsigma, Ndet, M);
		//Calculate M'^{-1} and M0'^{-1}
        double M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22;
		double aa = M[0] + M[3] + xi;
        double bb = 2*M[1];
        double cc = 2*M[2];
        double dd = M[3] + M[0] + xi;
        double detMprime = aa*dd - bb*cc;
        M_inverse_11 = dd/(aa*dd-bb*cc);
        M_inverse_12 = -cc/(aa*dd-bb*cc);
        M_inverse_21 = -bb/(aa*dd-bb*cc);
        M_inverse_22 = aa/(aa*dd-bb*cc);

        double M0_inverse_11,M0_inverse_12,M0_inverse_21,M0_inverse_22;
		double aa0 = M[0] + M[3] + xi;
        double dd0 = aa0;
        double detM0prime = aa0*dd0;
        M0_inverse_11 = 1.0/aa0;
        M0_inverse_12 = 0.0;
        M0_inverse_21 = 0.0;
        M0_inverse_22 = 1.0/dd0;

		double log_exp_term;
		double j_r1,j_r2,j_i1,j_i2;
		double log_prob_margT_bicorr=-1000000000;
		double prefactor = log(detMprime);
		double prefactor0 = log(detM0prime);

		//transform matched filtering snr to j stream
		double time_shifts[Ndet];
		double time_shift;
		double complex data;
		double max_snr_det_dt = XLALTimeDelayFromEarthCenter((detectors[max_snr_det_id]).location,ra,dec,&ligo_gps_time);
		double dt_ref=0.0;
		if (use_timediff){dt_ref = max_snr_det_dt;}

		for(det_id=0;det_id<Ndet;det_id++){
			if(det_id==max_snr_det_id){
				time_shifts[det_id] = max_snr_det_dt-dt_ref;
			}
			else{
				time_shifts[det_id] = XLALTimeDelayFromEarthCenter((detectors[det_id]).location,ra,dec,&ligo_gps_time)-dt_ref;
			}
		}

		// without loop unrolling
		for(time_id=0;time_id<ntime_interp;time_id++){
			j_r1 = 0;
			j_r2 = 0;
			j_i1 = 0;
			j_i2 = 0;

			for(det_id=0;det_id<Ndet;det_id++){
				time_shift = time_shifts[det_id];
				data = interpolate_time_series(snr_list[det_id], start_time + time_id*dt + time_shift, interp_order);
				//data = creal(data)*cos(cimag(data)) + creal(data)*sin(cimag(data)) * I ; //
				j_r1 += creal(data)*Gsigma[2*det_id];
				j_i1 += cimag(data)*Gsigma[2*det_id];
				j_r2 += creal(data)*Gsigma[2*det_id+1];
				j_i2 += cimag(data)*Gsigma[2*det_id+1];

			}

			log_exp_term = calcExptermBiCorr(j_r1, j_r2,j_i1, j_i2,
				alpha, prefactor, prefactor0,
				M_inverse_11, M_inverse_12, M_inverse_21, M_inverse_22,
				M0_inverse_11, M0_inverse_12, M0_inverse_21, M0_inverse_22);

			log_prob_margT_bicorr = logsumexp(log_prob_margT_bicorr,log_exp_term);
		}

		coh_skymap_bicorr[grid_id] = log_prob_margT_bicorr;
	} // end of for(grid_id)
	} // end of omp
}


/*
Coherent localization skymap with bimodal noncorrelated-digonal prior.
*/
void _coherent_skymap_bi(
				double *coh_skymap_bi, // The probability skymap we want to return
				const double *time_arrays,
				COMPLEX16TimeSeries **snr_list,
				LALDetector *detectors,
				const double *sigmas,
				const int *ntimes,
				const int Ndet,
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
				const int max_snr_det_id,
				const int use_timediff,
				const double premerger_time)
{
	int grid_id,time_id,det_id;

	double dt = (end_time-start_time)/ntime_interp;
	double ref_gps_time = (start_time + end_time)/2.0 - premerger_time;

	LIGOTimeGPS ligo_gps_time;
	ligo_gps_time.gpsSeconds = (int)(ref_gps_time);
	ligo_gps_time.gpsNanoSeconds = (ref_gps_time-(int)(ref_gps_time)) * 1E9;


	double mu_multimodal = prior_mu;
	double sigma_multimodal = prior_sigma;
	double xi = 1/sigma_multimodal/sigma_multimodal;
	double alpha = mu_multimodal*xi;
	#pragma omp parallel num_threads(nthread) private(time_id,det_id)  shared(coh_skymap_bi,snr_list,detectors)
	{
	#pragma omp for
	for(grid_id=0;grid_id<ngrid;grid_id+=1){
		coh_skymap_bi[grid_id]=0;

		double Gsigma[2*Ndet];
		double M[4];

		double ra, dec;
		pix2ang_nest64(nside, argsort_pix_id[grid_id], &dec, &ra);
		dec = M_PI/2 - dec;

		getGsigma_matrix(detectors,sigmas,Ndet,ra,dec,ref_gps_time,Gsigma);
		//Calculate M
		calcM(Gsigma, Ndet, M);
		//Calculate M'^{-1} and M0'^{-1}
        double M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22;
		double aa = M[0] + xi;
        double bb = M[1];
        double cc = M[2];
        double dd = M[3] + xi;
        double detMprime = aa*dd - bb*cc;
        M_inverse_11 = dd/(aa*dd-bb*cc);
        M_inverse_12 = -cc/(aa*dd-bb*cc);
        M_inverse_21 = -bb/(aa*dd-bb*cc);
        M_inverse_22 = aa/(aa*dd-bb*cc);

		double log_exp_term;
		double j_r1,j_r2,j_i1,j_i2;
		double log_prob_margT_bi=-1000000000;
		double prefactor = log(detMprime);

		//transform matched filtering snr to j stream
		double time_shifts[Ndet];
		double time_shift;
		double complex data;
		double max_snr_det_dt = XLALTimeDelayFromEarthCenter((detectors[max_snr_det_id]).location,ra,dec,&ligo_gps_time);
		double dt_ref=0.0;
		if (use_timediff){dt_ref = max_snr_det_dt;}

		for(det_id=0;det_id<Ndet;det_id++){
			if(det_id==max_snr_det_id){
				time_shifts[det_id] = max_snr_det_dt-dt_ref;
			}
			else{
				time_shifts[det_id] = XLALTimeDelayFromEarthCenter((detectors[det_id]).location,ra,dec,&ligo_gps_time)-dt_ref;
			}
		}

		for(time_id=0;time_id<ntime_interp;time_id++){
			j_r1 = 0;
			j_r2 = 0;
			j_i1 = 0;
			j_i2 = 0;

			for(det_id=0;det_id<Ndet;det_id++){
				time_shift = time_shifts[det_id];
				data = interpolate_time_series(snr_list[det_id], start_time + time_id*dt + time_shift, interp_order);
				//data = creal(data)*cos(cimag(data)) + creal(data)*sin(cimag(data)) * I ; //
				j_r1 += creal(data)*Gsigma[2*det_id];
				j_i1 += cimag(data)*Gsigma[2*det_id];
				j_r2 += creal(data)*Gsigma[2*det_id+1];
				j_i2 += cimag(data)*Gsigma[2*det_id+1];

			}

			log_exp_term = calcExptermBi(j_r1, j_r2,j_i1, j_i2,
				alpha, prefactor,
				M_inverse_11, M_inverse_12, M_inverse_21, M_inverse_22);

			log_prob_margT_bi = logsumexp(log_prob_margT_bi,log_exp_term);
		}

		coh_skymap_bi[grid_id] = log_prob_margT_bi;
	} // end of for(grid_id)
	} // end of omp
}

/*
Coherent localization skymap with flat prior.
*/
void _coherent_skymap_flat(
				double *coh_skymap_bi, // The probability skymap we want to return
				const double *time_arrays,
				COMPLEX16TimeSeries **snr_list,
				LALDetector *detectors,
				const double *sigmas,
				const int *ntimes,
				const int Ndet,
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
				const int max_snr_det_id,
				const int use_timediff,
				const double premerger_time)
{
	int grid_id,time_id,det_id;

	double dt = (end_time-start_time)/ntime_interp;
	double ref_gps_time = (start_time + end_time)/2.0 - premerger_time;

	LIGOTimeGPS ligo_gps_time;
	ligo_gps_time.gpsSeconds = (int)(ref_gps_time);
	ligo_gps_time.gpsNanoSeconds = (ref_gps_time-(int)(ref_gps_time)) * 1E9;


	//double mu_multimodal = prior_mu;
	//double sigma_multimodal = prior_sigma;
	//double xi = 1/sigma_multimodal/sigma_multimodal;
	//double alpha = mu_multimodal*xi;
	#pragma omp parallel num_threads(nthread) private(time_id,det_id)  shared(coh_skymap_bi,snr_list,detectors)
	{
	#pragma omp for
	for(grid_id=0;grid_id<ngrid;grid_id+=1){
		coh_skymap_bi[grid_id]=0;

		double Gsigma[2*Ndet];
		double M[4];

		double ra, dec;
		pix2ang_nest64(nside, argsort_pix_id[grid_id], &dec, &ra);
		dec = M_PI/2 - dec;

		getGsigma_matrix(detectors,sigmas,Ndet,ra,dec,ref_gps_time,Gsigma);
		//Calculate M
		calcM(Gsigma, Ndet, M);
		//Calculate M'^{-1} and M0'^{-1}
        double M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22;
		double aa = M[0];
        double bb = M[1];
        double cc = M[2];
        double dd = M[3];
        double detMprime = aa*dd - bb*cc;
        M_inverse_11 = dd/(aa*dd-bb*cc);
        M_inverse_12 = -cc/(aa*dd-bb*cc);
        M_inverse_21 = -bb/(aa*dd-bb*cc);
        M_inverse_22 = aa/(aa*dd-bb*cc);

		double log_exp_term;
		double j_r1,j_r2,j_i1,j_i2;
		double log_prob_margT_bi=-1000000000;
		double prefactor = log(detMprime);

		//transform matched filtering snr to j stream
		double time_shifts[Ndet];
		double time_shift;
		double complex data;
		double max_snr_det_dt = XLALTimeDelayFromEarthCenter((detectors[max_snr_det_id]).location,ra,dec,&ligo_gps_time);
		double dt_ref=0.0;
		if (use_timediff){dt_ref = max_snr_det_dt;}

		for(det_id=0;det_id<Ndet;det_id++){
			if(det_id==max_snr_det_id){
				time_shifts[det_id] = max_snr_det_dt-dt_ref;
			}
			else{
				time_shifts[det_id] = XLALTimeDelayFromEarthCenter((detectors[det_id]).location,ra,dec,&ligo_gps_time)-dt_ref;
			}
		}

		for(time_id=0;time_id<ntime_interp;time_id++){
			j_r1 = 0;
			j_r2 = 0;
			j_i1 = 0;
			j_i2 = 0;

			for(det_id=0;det_id<Ndet;det_id++){
				time_shift = time_shifts[det_id];
				data = interpolate_time_series(snr_list[det_id], start_time + time_id*dt + time_shift, interp_order);

				j_r1 += creal(data)*Gsigma[2*det_id];
				j_i1 += cimag(data)*Gsigma[2*det_id];
				j_r2 += creal(data)*Gsigma[2*det_id+1];
				j_i2 += cimag(data)*Gsigma[2*det_id+1];

			}

			log_exp_term = calcExptermFlat(j_r1, j_r2,j_i1, j_i2,
				prefactor,
				M_inverse_11, M_inverse_12, M_inverse_21, M_inverse_22);

			log_prob_margT_bi = logsumexp(log_prob_margT_bi,log_exp_term);
		}

		coh_skymap_bi[grid_id] = log_prob_margT_bi;
	} // end of for(grid_id)
	} // end of omp
}


/*
Coherent localization skymap with Gaussian prior.
*/
void _coherent_skymap_gaussian(
				double *coh_skymap_bi, // The probability skymap we want to return
				const double *time_arrays,
				COMPLEX16TimeSeries **snr_list,
				LALDetector *detectors,
				const double *sigmas,
				const int *ntimes,
				const int Ndet,
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
				const int max_snr_det_id,
				const int use_timediff,
				const double premerger_time)
{
	int grid_id,time_id,det_id;

	double dt = (end_time-start_time)/ntime_interp;
	double ref_gps_time = (start_time + end_time)/2.0 - premerger_time;

	LIGOTimeGPS ligo_gps_time;
	ligo_gps_time.gpsSeconds = (int)(ref_gps_time);
	ligo_gps_time.gpsNanoSeconds = (ref_gps_time-(int)(ref_gps_time)) * 1E9;


	//double mu_multimodal = prior_mu;
	//double sigma_multimodal = prior_sigma;
	//double xi = 1/sigma_multimodal/sigma_multimodal;
	//double alpha = mu_multimodal*xi;
	#pragma omp parallel num_threads(nthread) private(time_id,det_id)  shared(coh_skymap_bi,snr_list,detectors)
	{
	#pragma omp for
	for(grid_id=0;grid_id<ngrid;grid_id+=1){
		coh_skymap_bi[grid_id]=0;

		double Gsigma[2*Ndet];
		double M[4];

		double ra, dec;
		pix2ang_nest64(nside, argsort_pix_id[grid_id], &dec, &ra);
		dec = M_PI/2 - dec;

		getGsigma_matrix(detectors,sigmas,Ndet,ra,dec,ref_gps_time,Gsigma);
		//Calculate M
		calcM(Gsigma, Ndet, M);
		//Calculate M'^{-1} and M0'^{-1}
        double M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22;
		double R = 140;
		double xi = 5.0/3.0*R*R;
		double aa = M[0] + xi;
        double bb = M[1];
        double cc = M[2];
        double dd = M[3] + xi;
        double detMprime = aa*dd - bb*cc;
        M_inverse_11 = dd/(aa*dd-bb*cc);
        M_inverse_12 = -cc/(aa*dd-bb*cc);
        M_inverse_21 = -bb/(aa*dd-bb*cc);
        M_inverse_22 = aa/(aa*dd-bb*cc);

		double log_exp_term;
		double j_r1,j_r2,j_i1,j_i2;
		double log_prob_margT_bi=-1000000000;
		double prefactor = log(detMprime);

		//transform matched filtering snr to j stream
		double time_shifts[Ndet];
		double time_shift;
		double complex data;
		double max_snr_det_dt = XLALTimeDelayFromEarthCenter((detectors[max_snr_det_id]).location,ra,dec,&ligo_gps_time);
		double dt_ref=0.0;
		if (use_timediff){dt_ref = max_snr_det_dt;}

		for(det_id=0;det_id<Ndet;det_id++){
			if(det_id==max_snr_det_id){
				time_shifts[det_id] = max_snr_det_dt-dt_ref;
			}
			else{
				time_shifts[det_id] = XLALTimeDelayFromEarthCenter((detectors[det_id]).location,ra,dec,&ligo_gps_time)-dt_ref;
			}
		}

		for(time_id=0;time_id<ntime_interp;time_id++){
			j_r1 = 0;
			j_r2 = 0;
			j_i1 = 0;
			j_i2 = 0;

			for(det_id=0;det_id<Ndet;det_id++){
				time_shift = time_shifts[det_id];
				data = interpolate_time_series(snr_list[det_id], start_time + time_id*dt + time_shift, interp_order);

				j_r1 += creal(data)*Gsigma[2*det_id];
				j_i1 += cimag(data)*Gsigma[2*det_id];
				j_r2 += creal(data)*Gsigma[2*det_id+1];
				j_i2 += cimag(data)*Gsigma[2*det_id+1];

			}

			log_exp_term = calcExptermFlat(j_r1, j_r2,j_i1, j_i2,
				prefactor,
				M_inverse_11, M_inverse_12, M_inverse_21, M_inverse_22);

			log_prob_margT_bi = logsumexp(log_prob_margT_bi,log_exp_term);
		}

		coh_skymap_bi[grid_id] = log_prob_margT_bi;
	} // end of for(grid_id)
	} // end of omp
}


/*not complete yet. do not use.*/
void _coherent_snr(
				double *coh_skymap_bi, // The probability skymap we want to return
				const double *time_arrays,
				COMPLEX16TimeSeries **snr_list,
				LALDetector *detectors,
				const double *sigmas,
				const int *ntimes,
				const int Ndet,
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
				const int max_snr_det_id,
				const int use_timediff,
				const double premerger_time)
{
	int grid_id,time_id,det_id;

	double dt = (end_time-start_time)/ntime_interp;
	double ref_gps_time = (start_time + end_time)/2.0 - premerger_time;

	LIGOTimeGPS ligo_gps_time;
	ligo_gps_time.gpsSeconds = (int)(ref_gps_time);
	ligo_gps_time.gpsNanoSeconds = (ref_gps_time-(int)(ref_gps_time)) * 1E9;

	double coh_snr,coh_snr_temp;

	//double mu_multimodal = prior_mu;
	//double sigma_multimodal = prior_sigma;
	//double xi = 1/sigma_multimodal/sigma_multimodal;
	//double alpha = mu_multimodal*xi;
	#pragma omp parallel num_threads(nthread) private(time_id,det_id)  shared(coh_skymap_bi,snr_list,detectors)
	{
	#pragma omp for
	for(grid_id=0;grid_id<ngrid;grid_id+=1){
		coh_skymap_bi[grid_id]=0;

		double Gsigma[2*Ndet];
		double M[4];

		double ra, dec;
		pix2ang_nest64(nside, argsort_pix_id[grid_id], &dec, &ra);
		dec = M_PI/2 - dec;

		getGsigma_matrix(detectors,sigmas,Ndet,ra,dec,ref_gps_time,Gsigma);
		//Calculate M
		calcM(Gsigma, Ndet, M);
		//Calculate M'^{-1} and M0'^{-1}
        double M_inverse_11,M_inverse_12,M_inverse_21,M_inverse_22;
		double aa = M[0];
        double bb = M[1];
        double cc = M[2];
        double dd = M[3];
        double detMprime = aa*dd - bb*cc;
        M_inverse_11 = dd/(aa*dd-bb*cc);
        M_inverse_12 = -cc/(aa*dd-bb*cc);
        M_inverse_21 = -bb/(aa*dd-bb*cc);
        M_inverse_22 = aa/(aa*dd-bb*cc);

		double log_exp_term;
		double j_r1,j_r2,j_i1,j_i2;
		double log_prob_margT_bi=-1000000000;
		double prefactor = log(detMprime);

		//transform matched filtering snr to j stream
		double time_shifts[Ndet];
		double time_shift;
		double complex data;
		double max_snr_det_dt = XLALTimeDelayFromEarthCenter((detectors[max_snr_det_id]).location,ra,dec,&ligo_gps_time);
		double dt_ref=0.0;
		if (use_timediff){dt_ref = max_snr_det_dt;}

		for(det_id=0;det_id<Ndet;det_id++){
			if(det_id==max_snr_det_id){
				time_shifts[det_id] = max_snr_det_dt-dt_ref;
			}
			else{
				time_shifts[det_id] = XLALTimeDelayFromEarthCenter((detectors[det_id]).location,ra,dec,&ligo_gps_time)-dt_ref;
			}
		}

		for(time_id=0;time_id<ntime_interp;time_id++){
			j_r1 = 0;
			j_r2 = 0;
			j_i1 = 0;
			j_i2 = 0;

			for(det_id=0;det_id<Ndet;det_id++){
				time_shift = time_shifts[det_id];
				data = interpolate_time_series(snr_list[det_id], start_time + time_id*dt + time_shift, interp_order);

				j_r1 += creal(data)*Gsigma[2*det_id];
				j_i1 += cimag(data)*Gsigma[2*det_id];
				j_r2 += creal(data)*Gsigma[2*det_id+1];
				j_i2 += cimag(data)*Gsigma[2*det_id+1];

			}

			coh_snr_temp = calcExptermFlat(j_r1, j_r2,j_i1, j_i2,
				0.0,
				M_inverse_11, M_inverse_12, M_inverse_21, M_inverse_22);

			log_prob_margT_bi = logsumexp(log_prob_margT_bi,log_exp_term);
		}

		coh_skymap_bi[grid_id] = log_prob_margT_bi;
	} // end of for(grid_id)
	} // end of omp
}

// Comparison function for qsort in descending order
int compare_descending(const void *a, const void *b) {
    double val_a = *(const double *)a;
    double val_b = *(const double *)b;

    if (val_a > val_b) {
        return -1;
    } else if (val_a < val_b) {
        return 1;
    } else {
        return 0;
    }
}
int *rank_OneOfFour(const double *array, int length) {
    if (length % 4 != 0) {
        printf("Array size doesn't fit\n");
        exit(-1);
    }

    int i, j = 0, sum = 0;
    int quarter_length = length / 4;
    int *argsort = (int *)malloc(sizeof(int) * length);
    double *sorted_array = (double *)malloc(sizeof(double) * length);

    // Copy the input array to a new array for sorting
    memcpy(sorted_array, array, sizeof(double) * length);

    // Sort the new array in descending order
    qsort(sorted_array, length, sizeof(double), compare_descending);

    // Find the threshold value at the quarter_length position
    double threshold = sorted_array[quarter_length - 1];

    // Fill the argsort array with indices of elements greater than or equal to the threshold
    for (i = 0; i < length; i++) {
        if (array[i] >= threshold) {
            argsort[sum] = i;
            sum++;
        }
    }

    int *argsort_final = (int *)malloc(sizeof(int) * quarter_length);
    for (i = 0; i < quarter_length; i++) {
        argsort_final[i] = argsort[i];
    }
    free(argsort);
    free(sorted_array);

    return argsort_final;
}



void create_healpix_skygrids(int nside, double *ra_grids, double *dec_grids){
	int i;
	int npix = 12*nside*nside;
	for(i=0;i<npix;i++){
		pix2ang_nest64(nside, i, &dec_grids[i], &ra_grids[i]);
		dec_grids[i] = M_PI/2 - dec_grids[i];
	}
}

void normalize_log_probs(int npix, double *log_probs){
	double maxlogprob = -1000000000;
	double sum = 0;
	int i;

	// Find the maximum log probability
	#pragma omp parallel for reduction(max:maxlogprob)
	for(i = 0; i < npix; i++){
		if(log_probs[i] > maxlogprob){
			maxlogprob = log_probs[i];
		}
	}

	// Exponentiate and sum the probabilities
	#pragma omp parallel for reduction(+:sum)
	for(i = 0; i < npix; i++){
		log_probs[i] = exp(log_probs[i] - maxlogprob);
		sum += log_probs[i];
	}

	// Normalize the probabilities
	double inv_sum = 1.0 / sum;
	#pragma omp parallel for
	for(i = 0; i < npix; i++){
		log_probs[i] *= inv_sum;
	}
}

void coherent_skymap_multires(
	double *coh_skymap_multires, // The probability skymap we want to return
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
	const int use_timediff,
	const int prior_type,
	const double premerger_time)
{
	int i,j,i_level;
	int nside_base = 16;
	int npix_base  = 12*nside_base*nside_base;

	int nside_final = nside_base*pow(2,nlevel);
	int npix_final = 12*nside_final*nside_final;

	double *coh_skymap = (double*)malloc(sizeof(double)*npix_base);
	//initialize the argsort
	int *argsort        = (int*)malloc(sizeof(int)*npix_base/4);
	int *argsort_temp   = (int*)malloc(sizeof(int)*npix_base/4);
	int *argsort_pix_id = (int*)malloc(sizeof(int)*npix_base);
	for(i=0;i<npix_base/4;i++){
		argsort[i] = i;
		for(j=0;j<4;j++){
			argsort_pix_id[i*4+j] = i*4+j;
		}
	}

	// pre-process data
	COMPLEX16TimeSeries ** snr_list = CreateCOMPLEX16TimeSeriesList(time_arrays, snr_arrays, Ndet, ntimes);
	LALDetector tempdet, detectors[Ndet];
	int det_id;
	for(det_id=0; det_id<Ndet; det_id++){
		tempdet = lalCachedDetectors[detector_codes[det_id]];
		detectors[det_id] = tempdet;
	}

	for(i_level=0;i_level<nlevel+1;i_level++){
		int i_nside = nside_base*pow(2,i_level);
		int i_npix  = 12*i_nside*i_nside;
		//calculate skymap

		switch (prior_type)
		{
		case 0:
			_coherent_skymap_bicorr(
				coh_skymap, // The probability skymap we want to return
				time_arrays,
				snr_list,
				detectors,
				sigmas,
				ntimes,
				Ndet,
				argsort_pix_id,
				i_nside,
				npix_base,
				start_time,
				end_time,
				ntime_interp,
				prior_mu,
				prior_sigma,
				nthread,
				interp_order,
				max_snr_det_id,
				use_timediff,
				premerger_time);
			break;

		case 1:
			_coherent_skymap_bi(
				coh_skymap,
				time_arrays,
				snr_list,
				detectors,
				sigmas,
				ntimes,
				Ndet,
				argsort_pix_id,
				i_nside,
				npix_base,
				start_time,
				end_time,
				ntime_interp,
				prior_mu,
				prior_sigma,
				nthread,
				interp_order,
				max_snr_det_id,
				use_timediff,
				premerger_time);
			break;

		case 2:
			_coherent_skymap_flat(
				coh_skymap,
				time_arrays,
				snr_list,
				detectors,
				sigmas,
				ntimes,
				Ndet,
				argsort_pix_id,
				i_nside,
				npix_base,
				start_time,
				end_time,
				ntime_interp,
				prior_mu,
				prior_sigma,
				nthread,
				interp_order,
				max_snr_det_id,
				use_timediff,
				premerger_time);
			break;

		case 3:
			_coherent_skymap_gaussian(
				coh_skymap,
				time_arrays,
				snr_list,
				detectors,
				sigmas,
				ntimes,
				Ndet,
				argsort_pix_id,
				i_nside,
				npix_base,
				start_time,
				end_time,
				ntime_interp,
				prior_mu,
				prior_sigma,
				nthread,
				interp_order,
				max_snr_det_id,
				use_timediff,
				premerger_time);
			break;

		default:
			printf("Wrong prior type!\n");
			exit(-1);
		}



		//update skymap
		int nfactor = (int)pow(4,nlevel-i_level);
		int index;
		for(i=0;i<npix_base;i++){
			index = argsort_pix_id[i];
			for(j=0;j<nfactor;j++){
				coh_skymap_multires[index*nfactor+j] = coh_skymap[i];
			}
		}

		if(i_level<nlevel+1){
			argsort = rank_OneOfFour(coh_skymap,npix_base);
			for(i=0;i<npix_base/4;i++){
				argsort_temp[i] = argsort_pix_id[argsort[i]];
			}

			// new pixel ids for the next loop
			for(i=0;i<npix_base/4;i++){
				for(j=0;j<4;j++){
					argsort_pix_id[i*4+j] = argsort_temp[i]*4+j;
				}
			}
		}

	} // end of multires loop

	normalize_log_probs(npix_final, coh_skymap_multires);

	free(argsort);
	free(argsort_temp);
	free(argsort_pix_id);
	free(coh_skymap);
	DestroyCOMPLEX16TimeSeriesList(snr_list,Ndet);
}
