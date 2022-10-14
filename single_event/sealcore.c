#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <string.h>
#include <time.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>

#include <lal/LALDetectors.h>
#include <lal/LALSimulation.h>
#include <lal/TimeDelay.h>
#include <lal/LALDatatypes.h>

#include <coherent.h>



static double max_in_4(double loga, double logb, double logc, double logd){
	double temp = loga;
	if(temp<logb) temp = logb;
	if(temp<logc) temp = logc;
	if(temp<logd) temp = logd;
	return temp;
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
	int i,j,k;
	double data=0;
	
	if(A->size2 != B->size1){
		printf("A and B doesn't fit! \n");
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

static double complex step_interpolate_time_series(time_series* timeseries, double time)
{
	double start_time = timeseries->start_time;
	//printf("timeseries start time: %f", timeseries->start_time);
	//printf("timeseries npoint: %f", timeseries->npoint);
	//printf("timeseries delta t: %f", timeseries->delta_t);
	double end_time   = timeseries->start_time + timeseries->npoint*timeseries->delta_t;
	double delta_t    = timeseries->delta_t;

	if(time < start_time){
		printf("interpolate time can not smaller than the start time of time series!\n");
		exit(-1);
	}
	else if(time > end_time){
		//printf("%f > %f\n", time, end_time);
		printf("interpolate time can not larger than the end time of time series!\n");
		exit(-1);
	}

	int index = (int)((time-start_time)/delta_t);
	double diff = (time-start_time)/delta_t-(double)index;
	double complex int_data;
	if(diff<0.5){
		double real = creal(timeseries->data[index]);
		double imag = cimag(timeseries->data[index]);
		int_data = real+I*imag;
	}
	else{
		double real = creal(timeseries->data[index+1]);
		double imag = cimag(timeseries->data[index+1]);
		int_data = real+I*imag;
	}

	return int_data;
}

static double complex linear_interpolate_time_series(time_series* timeseries, double time)
{
	double start_time = timeseries->start_time;
	double end_time   = timeseries->start_time + timeseries->npoint*timeseries->delta_t;
	double delta_t    = timeseries->delta_t;

	if(time < start_time){
		printf("interpolate time can not smaller than the start time of time series!\n");
		exit(-1);
	}
	else if(time > end_time){
		printf("interpolate time can not larger than the end time of time series!\n");
		exit(-1);
	}

	int index = (int)((time-start_time)/delta_t);
	double diff = (time-start_time)/delta_t-(double)index;
	double real = creal(timeseries->data[index])*(1-diff) + creal(timeseries->data[index+1])*diff;
	double imag = cimag(timeseries->data[index])*(1-diff) + cimag(timeseries->data[index+1])*diff;
	double complex int_data = real+I*imag;

	return int_data;
}

static double complex quadratic_interpolate_time_series(time_series* timeseries, double time)
{
	double start_time = timeseries->start_time;
	double end_time   = timeseries->start_time + timeseries->npoint*timeseries->delta_t;
	double delta_t    = timeseries->delta_t;

	if(time < start_time){
		printf("interpolate time can not smaller than the start time of time series!\n");
		exit(-1);
	}
	else if(time > end_time){
		printf("interpolate time can not larger than the end time of time series!\n");
		exit(-1);
	}

	int index = (int)((time-start_time)/delta_t);
	double diff = (time-start_time)/delta_t-(double)index;
	double complex y_1,y_2,y_3;
	double x;
	
	if(diff<0.5){
		x = diff;
		y_1 = timeseries->data[index-1];
		y_2 = timeseries->data[index];
		y_3 = timeseries->data[index+1];
	}
	else{
		x = 1 - diff;
		y_1 = timeseries->data[index];
		y_2 = timeseries->data[index+1];
		y_3 = timeseries->data[index+2];
	}

	double real = creal(y_1)*x*(x-1.0)/(2.0) + creal(y_2)*(x+1.0)*(x-1.0)/(-1.0) + creal(y_3)*(x+1)*x/(2.0);
	double imag = cimag(y_1)*x*(x-1.0)/(2.0) + cimag(y_2)*(x+1.0)*(x-1.0)/(-1.0) + cimag(y_3)*(x+1)*x/(2.0);
	double int_data = real+I*imag;

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



/*
Coherent localization skymap with bimodal correlated-digonal prior. 
See arXiv:2110.01874.
*/
double *coherent_skymap_bicorr(
				const data_streams *strain_data, 
				const LALDetector *detectors, 
				const double *sigma, 
				const double *ra_grids, 
				const double *dec_grids, 
				int ngrid, 
				double start_time, 
				double end_time, 
				int ntime,
                double prior_mu,
                double prior_sigma)
{
	printf("start calculating the sky map by coherent method\n");

	int grid_id,time_id,det_id;
	int Ndet = strain_data->Nstream;
	double Gsigma[2*Ndet]  //,singular[2];
	double dt = (end_time-start_time)/ntime;
	double ref_gps_time = (start_time + end_time)/2.0;

	//get the U matrix skymap first
	//double *cohfactor = coherent_snr_combination_factors_skymap(detectors,sigma,Ndet,ra_grids,dec_grids,ngrid,ref_gps_time);
	//double *Gsigma_temp[Ndet*2];
	double *coh_skymap_bicorr = (double*)malloc(sizeof(double)*ngrid);
	printf("ngrid  = %d  \n",ngrid);
	printf("ntime  = %d  \n",ntime);
	printf("ndet   = %d  \n",Ndet);

	//gsl_matrix *Utrans = gsl_matrix_alloc(Ndet,2);
	gsl_matrix *detector_real_streams = gsl_matrix_calloc(ntime,Ndet);
	gsl_matrix *detector_imag_streams = gsl_matrix_calloc(ntime,Ndet);
	//gsl_matrix *signal_real_streams   = gsl_matrix_calloc(ntime,2);
	//gsl_matrix *signal_imag_streams   = gsl_matrix_calloc(ntime,2);

	gsl_matrix *M_prime = gsl_matrix_alloc(2,2);

	gsl_matrix *G_sigma = gsl_matrix_alloc(Ndet,2); //same as previouly defined
	gsl_matrix *G_sigma_transpose = gsl_matrix_alloc(2,Ndet);

	gsl_matrix *J_real_streams   = gsl_matrix_calloc(ntime,2);
	gsl_matrix *J_imag_streams   = gsl_matrix_calloc(ntime,2);

	//gsl_vector *null_stream = gsl_vector_calloc(ntime);
	//gsl_matrix *I_dagger    = gsl_matrix_calloc(Ndet,Ndet);
	//gsl_vector_set_zero(null_stream);

	LIGOTimeGPS ligo_gps_time;
	ligo_gps_time.gpsSeconds = (int)(ref_gps_time);
	ligo_gps_time.gpsNanoSeconds = 0;


	double mu_multimodal = prior_mu;
	double sigma_multimodal = prior_sigma;
	double xi = 1/sigma_multimodal/sigma_multimodal;
	double alpha = mu_multimodal*xi;

	for(grid_id=0;grid_id<ngrid;grid_id++){
		//set parameters
		double ra  = ra_grids[grid_id];
		double dec = dec_grids[grid_id];

		//get transform matrix
		//for(det_id=0;det_id<Ndet;det_id++){
		//	gsl_matrix_set(Utrans,det_id,0,cohfactor[grid_id*Ndet*2+det_id*2+0]);
		//	gsl_matrix_set(Utrans,det_id,1,cohfactor[grid_id*Ndet*2+det_id*2+1]);
		//}


		//time shift the data
		for(det_id=0;det_id<Ndet;det_id++){
			double time_shift = XLALTimeDelayFromEarthCenter((detectors[det_id]).location,ra,dec,&ligo_gps_time);
			for(time_id=0;time_id<ntime;time_id++){
				double complex data = step_interpolate_time_series(strain_data->streams[det_id], start_time + time_id*dt + time_shift); //&strain_data
				//double complex data = linear_interpolate_time_series(&strain_data->streams[det_id], start_time + time_id*dt + time_shift);
				//double complex data = quadratic_interpolate_time_series(&strain_data->streams[det_id], start_time + time_id*dt + time_shift);
				gsl_matrix_set(detector_real_streams,time_id,det_id,creal(data));
				gsl_matrix_set(detector_imag_streams,time_id,det_id,cimag(data));
			}
		}
		
		


		//transform from matched filter data to signal stream
        /*
		for(time_id=0;time_id<ntime;time_id++){
			double temp0_real=0;
			double temp0_imag=0;
			double temp1_real=0;
			double temp1_imag=0;
			for(det_id=0;det_id<Ndet;det_id++){
				temp0_real += gsl_matrix_get(detector_real_streams,time_id,det_id)*gsl_matrix_get(Utrans,det_id,0);
				temp0_imag += gsl_matrix_get(detector_imag_streams,time_id,det_id)*gsl_matrix_get(Utrans,det_id,0);
				temp1_real += gsl_matrix_get(detector_real_streams,time_id,det_id)*gsl_matrix_get(Utrans,det_id,1);
				temp1_imag += gsl_matrix_get(detector_imag_streams,time_id,det_id)*gsl_matrix_get(Utrans,det_id,1);
			}
			gsl_matrix_set(signal_real_streams,time_id,0,temp0_real);
			gsl_matrix_set(signal_imag_streams,time_id,0,temp0_imag);
			gsl_matrix_set(signal_real_streams,time_id,1,temp1_real);
			gsl_matrix_set(signal_imag_streams,time_id,1,temp1_imag);
		}*/

		

		//transform from strain data to null stream
		/*if(Ndet>2){
			calc_null_stream(detector_real_streams,detector_imag_streams,Utrans,I_dagger,null_stream);
		}*/
		
		//calculate sigular value
		getGsigma_matrix(detectors,sigma,Ndet,ra,dec,ref_gps_time,Gsigma);
		//svd_Gsigma_get_singular_value(Gsigma,Ndet,NULL,singular);

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
        //double bb0 = 0;
        //double cc0 = 0;
        double dd0 = gsl_matrix_get(M_prime,1,1) + gsl_matrix_get(M_prime,0,0)+ xi;
        double detM0prime = aa0*dd0;
        M0_inverse_11 = 1.0/aa0;
        M0_inverse_12 = 0.0;
        M0_inverse_21 = 0.0;
        M0_inverse_22 = 1.0/dd0;
		
		//transform mf data to j stream
		for(time_id=0;time_id<ntime;time_id++){
			double temp0_real=0;
			double temp0_imag=0;
			double temp1_real=0;
			double temp1_imag=0;
			for(det_id=0;det_id<Ndet;det_id++){
				temp0_real += gsl_matrix_get(detector_real_streams,time_id,det_id)*gsl_matrix_get(G_sigma,det_id,0);
				temp0_imag += gsl_matrix_get(detector_imag_streams,time_id,det_id)*gsl_matrix_get(G_sigma,det_id,0);
				temp1_real += gsl_matrix_get(detector_real_streams,time_id,det_id)*gsl_matrix_get(G_sigma,det_id,1);
				temp1_imag += gsl_matrix_get(detector_imag_streams,time_id,det_id)*gsl_matrix_get(G_sigma,det_id,1);
			}
			gsl_matrix_set(J_real_streams,time_id,0,temp0_real);
			gsl_matrix_set(J_imag_streams,time_id,0,temp0_imag);
			gsl_matrix_set(J_real_streams,time_id,1,temp1_real);
			gsl_matrix_set(J_imag_streams,time_id,1,temp1_imag);
		}


		//calculate skymap
		//double signal0_real,signal0_imag,signal1_real,signal1_imag;
		double snr_temp;
		double log_exp_term1,log_exp_term2,log_exp_term,log_exp_term3,log_exp_term4;
		double j_r1,j_r2,j_i1,j_i2;
		double snr_null_temp;
		double log_probability_bicorr;
		double log_prob_margT_bicorr=-100;
		double prefactor = log(detMprime);
		double prefactor0 = log(detM0prime);
		//coh_skymap_bicorr[grid_id+0*ngrid]=0;
		//coh_skymap_bicorr[grid_id+1*ngrid]=0;
		coh_skymap_bicorr[grid_id]=0;
		for(time_id=0;time_id<ntime;time_id++){
			//signal0_real = gsl_matrix_get(signal_real_streams,time_id,0);
			//signal0_imag = gsl_matrix_get(signal_imag_streams,time_id,0);
			//signal1_real = gsl_matrix_get(signal_real_streams,time_id,1);
			//signal1_imag = gsl_matrix_get(signal_imag_streams,time_id,1);

			//snr_temp = sqrt(signal0_real*signal0_real + signal0_imag*signal0_imag + 
			//				signal1_real*signal1_real + signal1_imag*signal1_imag);

			//snr_null_temp = snr_temp*snr_temp - gsl_vector_get(null_stream,time_id);
			//snr_null_temp = sqrt(snr_null_temp);
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
			


			//exp_term = (signal0_real*signal0_real + signal0_imag*signal0_imag)*factor0 + (signal1_real*signal1_real + signal1_imag*signal1_imag)*factor1;

			//log_probability_bicorr = prefactor + 1.0/2.0*exp_term - gsl_vector_get(null_stream,time_id)/2.0;
			log_probability_bicorr = log_exp_term;

			log_prob_margT_bicorr = logsumexp(log_prob_margT_bicorr,log_probability_bicorr);

			//if(snr_temp>coh_skymap_bicorr[grid_id+0*ngrid]){//find the maximum snr value accross the ntime
			//	coh_skymap_bicorr[grid_id+0*ngrid] = snr_temp;
			//}
			/*if(snr_null_temp>coh_skymap_bicorr[grid_id+1*ngrid]){
				coh_skymap_bicorr[grid_id+1*ngrid] = snr_null_temp;
			}*/
		}
		coh_skymap_bicorr[grid_id] = log_prob_margT_bicorr;
	}

	free(cohfactor);
	//gsl_matrix_free(Utrans);
	gsl_matrix_free(detector_real_streams);
	gsl_matrix_free(detector_imag_streams);
	//gsl_matrix_free(signal_real_streams);
	//gsl_matrix_free(signal_imag_streams);
	//gsl_matrix_free(I_dagger);

	gsl_matrix_free(M_prime);
	gsl_matrix_free(G_sigma);
	gsl_matrix_free(G_sigma_transpose);
	gsl_matrix_free(J_real_streams);
	gsl_matrix_free(J_imag_streams);


	printf("end of calculate coherent snr \n");
	printf("---------------------------------------\n");

	return coh_skymap_bicorr;
}