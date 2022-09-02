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

int LOAD_SKYGRIDS=0;
double *healpix_skygrids_nside_16;
double *healpix_skygrids_nside_32;
double *healpix_skygrids_nside_64;
double *healpix_skygrids_nside_128;
double *healpix_skygrids_nside_256;
double *healpix_skygrids_nside_512;
double *healpix_skygrids_nside_1024;

data_streams* create_data_streams(int Ndet)
{
	data_streams *datastreams = (data_streams*)malloc(sizeof(data_streams));
	time_series  *streams  = (time_series* )malloc(sizeof(time_series)*Ndet);

	datastreams->streams = streams;
	datastreams->Nstream = Ndet;
	return datastreams;
}

time_series* create_time_series(int npoint, double start_time, double delta_t, double complex *data)
{
	time_series *timeseries = (time_series*)malloc(sizeof(time_series));

	timeseries->npoint = npoint;
	timeseries->start_time = start_time;
	timeseries->delta_t = delta_t;
	timeseries->data = (double complex*)malloc(sizeof(double complex)*npoint);

	if(data==NULL){
		return timeseries;
	}
	else{
		int i;
		for(i=0;i<npoint;i++){
			timeseries->data[i] = data[i];
		}
	}

    return timeseries;
}

void free_time_series(time_series *timeseries)
{
	free(timeseries->data);
	free(timeseries);
}

void free_data_streams(data_streams *datastreams)
{
	int i;
	//for(i=0;i<datastreams->Nstream;i++){
	//	free_time_series(&(datastreams->streams[i]));
	//}
	free(datastreams->streams);
	free(datastreams);
}

static void my_memcpy_double(double *array_copy, const double *array, int npoint)
{
	int i;
	for(i=0;i<npoint;i++){
		array_copy[i] = array[i];
	}
}

time_series* readsnr2time_series(char *filename)
{
	printf("txt file from    : %s \n",filename);

	int i=0;
	int npoint;
	double gps_time,deltaT;

	//read data from txt
	FILE *ft_r = fopen(filename,"r");
	if(ft_r==NULL){
		printf("can not found the file\n");
		exit(-1);
	}
	
	npoint=0;
	double time,data_real,data_imag;
	while(fscanf(ft_r,"%lf %lf %lf",&time,&data_real,&data_imag)!=EOF){
		//printf("%e %e %e\n",time,data_real,data_imag);
		npoint++;
	}
	fclose(ft_r);
	
	FILE *fp_r = fopen(filename,"r");

	double complex *data = (double complex*)malloc(sizeof(double complex)*npoint);
	while(fscanf(fp_r,"%lf %lf %lf",&time,&data_real,&data_imag)!=EOF){
		data[i] = data_real + data_imag*I;
		//printf("%e \n",data[i]);
		if(i==0) gps_time=time; 
		if(i==1) deltaT=time-gps_time;
		i++;
	}
	fclose(fp_r);

	time_series *timeseries = create_time_series(npoint,gps_time,deltaT,NULL);

	for(i=0;i<npoint;i++){
		timeseries->data[i] = data[i];
	}

	printf("GPSTime = %lf \n",timeseries->start_time);
	printf("npoint  = %d  \n",timeseries->npoint);
	printf("deltaT  = %lf \n",timeseries->delta_t);
	printf("srate   = %lf \n",1.0/timeseries->delta_t);
	printf("end of read data \n");
	printf("---------------------------------------\n");

	free(data);

	return timeseries;
}

void read_skygrids(char *filename, double *ra, double *dec)
{
	printf("txt file from    : %s \n",filename);

	int npoint;

	//read data from txt
	FILE *ft_r = fopen(filename,"r");
	if(ft_r==NULL){
		printf("can not found the file\n");
		exit(-1);
	}
	
	npoint=0;
	double ra_temp,dec_temp;
	while(fscanf(ft_r,"%lf %lf",&ra_temp,&dec_temp)!=EOF){
		ra[npoint]  = ra_temp;
		dec[npoint] = dec_temp;
		npoint++;
	}
	fclose(ft_r);

	printf("npoint  = %d  \n",npoint);
	printf("---------------------------------------\n");
}

void read_sigma(char *filename, double *sigma)
{
	printf("txt file from    : %s \n",filename);

	int npoint;

	//read data from txt
	FILE *ft_r = fopen(filename,"r");
	if(ft_r==NULL){
		printf("can not found the file\n");
		exit(-1);
	}
	
	printf("the horizon distance is \n");
	npoint=0;
	double sigma_temp;
	while(fscanf(ft_r,"%lf",&sigma_temp)!=EOF){
		sigma[npoint]  = sigma_temp;
		printf("%lf ",sigma_temp);
		npoint++;
	}
	fclose(ft_r);

	printf("\nnpoint  = %d  \n",npoint);
	printf("---------------------------------------\n");
}

void create_healpix_skygrids_from_file(int nside, double *ra_grids, double *dec_grids)
{
	if(ra_grids==NULL || dec_grids==NULL){
		printf("You need to alloc the grid pointer first\n");
		exit(-1);
	}

	if(LOAD_SKYGRIDS==0){
		char skygrids_filename[128];
		// /home/guestgw/cong/statistic_c/gaussian_noise/
		if(nside==16) strcpy(skygrids_filename,"sky_grids/sky_grids_nside_16.txt");
		else if(nside==32) strcpy(skygrids_filename,"sky_grids/sky_grids_nside_32.txt");
		else if(nside==64) strcpy(skygrids_filename,"sky_grids/sky_grids_nside_64.txt");
		else if(nside==128) strcpy(skygrids_filename,"sky_grids/sky_grids_nside_128.txt");
		else if(nside==256) strcpy(skygrids_filename,"sky_grids/sky_grids_nside_256.txt");
		else if(nside==512) strcpy(skygrids_filename,"sky_grids/sky_grids_nside_512.txt");
		else if(nside==1024) strcpy(skygrids_filename,"sky_grids/sky_grids_nside_1024.txt");
		else{ 
			printf("There is no such file!\n");
			exit(-1);
		}
	
		read_skygrids(skygrids_filename,ra_grids,dec_grids);
	}
	else if(LOAD_SKYGRIDS==1){
		int npix = 12*nside*nside;
		if(nside==16){ 
			my_memcpy_double(ra_grids,healpix_skygrids_nside_16,npix);
			my_memcpy_double(dec_grids,&healpix_skygrids_nside_16[npix],npix);
		}
		else if(nside==32){ 
			my_memcpy_double(ra_grids,healpix_skygrids_nside_32,npix);
			my_memcpy_double(dec_grids,&healpix_skygrids_nside_32[npix],npix);
		}
		else if(nside==64){ 
			my_memcpy_double(ra_grids,healpix_skygrids_nside_64,npix);
			my_memcpy_double(dec_grids,&healpix_skygrids_nside_64[npix],npix);
		}
		else if(nside==128){ 
			my_memcpy_double(ra_grids,healpix_skygrids_nside_128,npix);
			my_memcpy_double(dec_grids,&healpix_skygrids_nside_128[npix],npix);
		}
		else if(nside==256){ 
			my_memcpy_double(ra_grids,healpix_skygrids_nside_256,npix);
			my_memcpy_double(dec_grids,&healpix_skygrids_nside_256[npix],npix);
		}
		else if(nside==512){ 
			my_memcpy_double(ra_grids,healpix_skygrids_nside_512,npix);
			my_memcpy_double(dec_grids,&healpix_skygrids_nside_512[npix],npix);
		}
		else if(nside==1024){ 
			my_memcpy_double(ra_grids,healpix_skygrids_nside_1024,npix);
			my_memcpy_double(dec_grids,&healpix_skygrids_nside_1024[npix],npix);
		}
		else{
			printf("nside can only be 16 32 64 128 256 512 1024\n");
			exit(-1);
		}
	}
}

void load_healpix_skygrids_from_file()
{
	if(LOAD_SKYGRIDS==1){
		printf("you already load the skygrids\n");
		return ;
	}

	int i,nside[7] = {16,32,64,128,256,512,1024};

	for(i=0;i<7;i++){
		int npix = 12*nside[i]*nside[i];
		printf("loading data from healpix skygrids \n");
		printf("nside   = %d\n",nside[i]);
		printf("npix    = %d\n",npix);
		double *ra_grids = (double*)malloc(sizeof(double)*npix);
		double *dec_grids = (double*)malloc(sizeof(double)*npix);
		
		create_healpix_skygrids_from_file(nside[i],ra_grids,dec_grids);
		if(nside[i]==16){
			healpix_skygrids_nside_16 = (double*)malloc(sizeof(double)*npix*2);
			my_memcpy_double(healpix_skygrids_nside_16,ra_grids,npix);
			my_memcpy_double(&healpix_skygrids_nside_16[npix],dec_grids,npix);
		}
		else if(nside[i]==32){
			healpix_skygrids_nside_32 = (double*)malloc(sizeof(double)*npix*2);
			my_memcpy_double(healpix_skygrids_nside_32,ra_grids,npix);
			my_memcpy_double(&healpix_skygrids_nside_32[npix],dec_grids,npix);
		}
		else if(nside[i]==64){
			healpix_skygrids_nside_64 = (double*)malloc(sizeof(double)*npix*2);
			my_memcpy_double(healpix_skygrids_nside_64,ra_grids,npix);
			my_memcpy_double(&healpix_skygrids_nside_64[npix],dec_grids,npix);
		}
		else if(nside[i]==128){
			healpix_skygrids_nside_128 = (double*)malloc(sizeof(double)*npix*2);
			my_memcpy_double(healpix_skygrids_nside_128,ra_grids,npix);
			my_memcpy_double(&healpix_skygrids_nside_128[npix],dec_grids,npix);
		}
		else if(nside[i]==256){
			healpix_skygrids_nside_256 = (double*)malloc(sizeof(double)*npix*2);
			my_memcpy_double(healpix_skygrids_nside_256,ra_grids,npix);
			my_memcpy_double(&healpix_skygrids_nside_256[npix],dec_grids,npix);
		}
		else if(nside[i]==512){
			healpix_skygrids_nside_512 = (double*)malloc(sizeof(double)*npix*2);
			my_memcpy_double(healpix_skygrids_nside_512,ra_grids,npix);
			my_memcpy_double(&healpix_skygrids_nside_512[npix],dec_grids,npix);
		}
		else if(nside[i]==1024){
			healpix_skygrids_nside_1024 = (double*)malloc(sizeof(double)*npix*2);
			my_memcpy_double(healpix_skygrids_nside_1024,ra_grids,npix);
			my_memcpy_double(&healpix_skygrids_nside_1024[npix],dec_grids,npix);
		}
		free(ra_grids);
		free(dec_grids);
	}
	LOAD_SKYGRIDS=1;
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

//This function is copy from lalsuite
/**
 ** An implementation of the detector response formulae in Anderson et al
 ** PRD 63 042003 (2001) \cite ABCF2001 .
 **
 ** Computes F+ and Fx for a source at a specified sky position,
 ** polarization angle, and sidereal time.  Also requires the detector's
 ** response matrix which is defined by Eq. (B6) of [ABCF] using either
 ** Table 1 of \cite ABCF2001 or Eqs. (B11)--(B17) to compute the arm
 ** direction unit vectors.
 **/
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

static void svd_Gsigma_get_singular_value(const double *Gsigma, int Ndet, gsl_vector *work, double *singular)
{
	gsl_vector *s = gsl_vector_alloc(2);
	gsl_matrix *a = gsl_matrix_alloc(Ndet,2);
	gsl_matrix *v = gsl_matrix_alloc(2,2);
	
	int i;
	for(i=0;i<Ndet;i++){
		gsl_matrix_set(a,i,0,Gsigma[i*2]);
		gsl_matrix_set(a,i,1,Gsigma[i*2+1]);
	}

	if(work==NULL){
		gsl_vector *work_temp = gsl_vector_alloc(2);
		gsl_linalg_SV_decomp(a,v,s,work_temp);
		gsl_vector_free(work_temp);
	}
	else{
		gsl_linalg_SV_decomp(a,v,s,work);
	}

	singular[0] = gsl_vector_get(s,0);
	singular[1] = gsl_vector_get(s,1);
	
	gsl_matrix_free(a);
	gsl_matrix_free(v);
	gsl_vector_free(s);

}

//get the matrix U 
static void svd_Gsigma(const double *Gsigma, int Ndet, gsl_vector *work, double *e)
{
	gsl_vector *s = gsl_vector_alloc(2);
	gsl_matrix *a = gsl_matrix_alloc(Ndet,2);
	gsl_matrix *v = gsl_matrix_alloc(2,2);
	
	int i;
	for(i=0;i<Ndet;i++){
		gsl_matrix_set(a,i,0,Gsigma[i*2]);
		gsl_matrix_set(a,i,1,Gsigma[i*2+1]);
	}

	if(work==NULL){
		gsl_vector *work_temp = gsl_vector_alloc(2);
		gsl_linalg_SV_decomp(a,v,s,work_temp);
		gsl_vector_free(work_temp);
	}
	else{
		gsl_linalg_SV_decomp(a,v,s,work);
	}

	for(i=0;i<Ndet;i++){
		e[i*2]   = gsl_matrix_get(a,i,0);
		e[i*2+1] = gsl_matrix_get(a,i,1);
	}
	
	gsl_matrix_free(a);
	gsl_matrix_free(v);
	gsl_vector_free(s);

}

static double *coherent_snr_combination_factors_skymap(
				const LALDetector *detectors, 
				const double *sigma, 
				int Ndet, 
				const double *ra_grids, 
				const double *dec_grids, 
				int Ngrid, 
				double gps_time)
{
	double *cohfactor = (double*)malloc(sizeof(double)*2*Ndet*Ngrid);
	int grid_id,det_id;
	double *e = (double*)malloc(sizeof(double)*2*Ndet);
	double *Gsigma = (double*)malloc(sizeof(double)*2*Ndet);

	for(grid_id=0;grid_id<Ngrid;grid_id++){
		getGsigma_matrix(detectors,sigma,Ndet,ra_grids[grid_id],dec_grids[grid_id],gps_time,Gsigma);
		svd_Gsigma(Gsigma,Ndet,NULL,e);
		for(det_id=0;det_id<Ndet;det_id++){
			cohfactor[grid_id*Ndet*2+det_id*2+0] = e[det_id*2+0];
			cohfactor[grid_id*Ndet*2+det_id*2+1] = e[det_id*2+1];
		}

	}

	free(Gsigma);
	free(e);

	return cohfactor;
}


static void calc_null_stream(
				const gsl_matrix *detector_real_streams, 
				const gsl_matrix *detector_imag_streams, 
				const gsl_matrix *Utrans, 
				gsl_matrix *I_dagger, 
				gsl_vector *null_stream)
{
	int ntime = detector_real_streams->size1;
	int Ndet  = detector_real_streams->size2;
	int i,j;
	double element;
	gsl_matrix *null_real_streams = gsl_matrix_calloc(ntime,Ndet);
	gsl_matrix *null_imag_streams = gsl_matrix_calloc(ntime,Ndet);
	gsl_matrix *UtransT = gsl_matrix_calloc(2,Ndet);
	
	//projector on signal streams
	for(i=0;i<Ndet;i++){
		for(j=0;j<2;j++){
			gsl_matrix_set(UtransT,j,i,gsl_matrix_get(Utrans,i,j));
		}
	}
	gsl_matrix_mult(Utrans,UtransT,I_dagger);

	//1-projector
	for(i=0;i<Ndet;i++){
		for(j=0;j<Ndet;j++){
			element = gsl_matrix_get(I_dagger,i,j);
			if(i==j) gsl_matrix_set(I_dagger,i,j,1.0-element);
			else gsl_matrix_set(I_dagger,i,j,-element);
		}
	}

	//d^T (1-projector) d
	gsl_matrix_mult(detector_real_streams,I_dagger,null_real_streams);
	gsl_matrix_mult(detector_imag_streams,I_dagger,null_imag_streams);
	for(i=0;i<ntime;i++){
		element=0;
		for(j=0;j<Ndet;j++){
			element += gsl_matrix_get(detector_real_streams,i,j)*gsl_matrix_get(null_real_streams,i,j);
			element += gsl_matrix_get(detector_imag_streams,i,j)*gsl_matrix_get(null_imag_streams,i,j);
		}
		if(element<0) element=0;
		gsl_vector_set(null_stream,i,element);
	}
	

	gsl_matrix_free(null_real_streams);
	gsl_matrix_free(null_imag_streams);
	gsl_matrix_free(UtransT);
}

double *coherent_skymap_alan(
				const data_streams *strain_data, 
				const LALDetector *detectors, 
				const double *sigma, 
				const double *ra_grids, 
				const double *dec_grids, 
				int ngrid, 
				double start_time, 
				double end_time, 
				int ntime)
{
	printf("start calculating the sky map by coherent method\n");

	int grid_id,time_id,det_id;
	int Ndet = strain_data->Nstream;
	double Gsigma[2*Ndet],singular[2];
	double dt = (end_time-start_time)/ntime;
	double ref_gps_time = (start_time + end_time)/2.0;

	//get the U matrix skymap first
	double *cohfactor = coherent_snr_combination_factors_skymap(detectors,sigma,Ndet,ra_grids,dec_grids,ngrid,ref_gps_time);
	//double *coh_skymap = (double*)malloc(sizeof(double)*ngrid*6);//6 skymaps
	double *coh_skymap = (double*)malloc(sizeof(double)*ngrid*3);//6 skymaps
	printf("ngrid  = %d  \n",ngrid);
	printf("ntime  = %d  \n",ntime);
	printf("ndet   = %d  \n",Ndet);

	gsl_matrix *Utrans = gsl_matrix_alloc(Ndet,2);
	gsl_matrix *detector_real_streams = gsl_matrix_calloc(ntime,Ndet);
	gsl_matrix *detector_imag_streams = gsl_matrix_calloc(ntime,Ndet);
	gsl_matrix *signal_real_streams   = gsl_matrix_calloc(ntime,2);
	gsl_matrix *signal_imag_streams   = gsl_matrix_calloc(ntime,2);

	gsl_vector *null_stream = gsl_vector_calloc(ntime);
	gsl_matrix *I_dagger    = gsl_matrix_calloc(Ndet,Ndet);
	gsl_vector_set_zero(null_stream);

	LIGOTimeGPS ligo_gps_time;
	ligo_gps_time.gpsSeconds = (int)(ref_gps_time);
	ligo_gps_time.gpsNanoSeconds = 0;

	for(grid_id=0;grid_id<ngrid;grid_id++){
		//set parameters
		double ra  = ra_grids[grid_id];
		double dec = dec_grids[grid_id];

		//get transform matrix
		for(det_id=0;det_id<Ndet;det_id++){
			gsl_matrix_set(Utrans,det_id,0,cohfactor[grid_id*Ndet*2+det_id*2+0]);
			gsl_matrix_set(Utrans,det_id,1,cohfactor[grid_id*Ndet*2+det_id*2+1]);
		}

		//time shift the data
		for(det_id=0;det_id<Ndet;det_id++){
			double time_shift = XLALTimeDelayFromEarthCenter((detectors[det_id]).location,ra,dec,&ligo_gps_time);
			for(time_id=0;time_id<ntime;time_id++){
				double complex data = step_interpolate_time_series(strain_data->streams[det_id], start_time + time_id*dt + time_shift);
				//double complex data = linear_interpolate_time_series(&strain_data->streams[det_id], start_time + time_id*dt + time_shift);
				//double complex data = quadratic_interpolate_time_series(&strain_data->streams[det_id], start_time + time_id*dt + time_shift);
				gsl_matrix_set(detector_real_streams,time_id,det_id,creal(data));
				gsl_matrix_set(detector_imag_streams,time_id,det_id,cimag(data));
			}
		}

		//transform from matched filter data to signal stream
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
		}

		//transform from strain data to null stream
		if(Ndet>2){
			calc_null_stream(detector_real_streams,detector_imag_streams,Utrans,I_dagger,null_stream);
		}
		
		//calculate sigular value
		getGsigma_matrix(detectors,sigma,Ndet,ra,dec,ref_gps_time,Gsigma);
		svd_Gsigma_get_singular_value(Gsigma,Ndet,NULL,singular);


		//calculate skymap
		double signal0_real,signal0_imag,signal1_real,signal1_imag;
		//double p1,p2,pp=0;
		double snr_temp;
		double snr_null_temp;
		double log_probability;
		double log_prob_margT=-100;
		double prefactor = log(4*M_PI*M_PI/(singular[0]*singular[0]*singular[1]*singular[1]));
		coh_skymap[grid_id+0*ngrid]=0;
		coh_skymap[grid_id+1*ngrid]=0;
		coh_skymap[grid_id+2*ngrid]=0;
		for(time_id=0;time_id<ntime;time_id++){
			signal0_real = gsl_matrix_get(signal_real_streams,time_id,0);
			signal0_imag = gsl_matrix_get(signal_imag_streams,time_id,0);
			signal1_real = gsl_matrix_get(signal_real_streams,time_id,1);
			signal1_imag = gsl_matrix_get(signal_imag_streams,time_id,1);

			snr_temp = sqrt(signal0_real*signal0_real + signal0_imag*signal0_imag + 
							signal1_real*signal1_real + signal1_imag*signal1_imag);

			snr_null_temp = snr_temp*snr_temp - gsl_vector_get(null_stream,time_id);
			snr_null_temp = sqrt(snr_null_temp);


			log_probability = snr_temp*snr_temp/8.0 + prefactor - gsl_vector_get(null_stream,time_id)/2.0;
			log_prob_margT = logsumexp(log_prob_margT,log_probability);

			if(snr_temp>coh_skymap[grid_id+0*ngrid]){//find the maximum snr value accross the ntime
				coh_skymap[grid_id+0*ngrid] = snr_temp;
			}
			if(snr_null_temp>coh_skymap[grid_id+1*ngrid]){
				coh_skymap[grid_id+1*ngrid] = snr_null_temp;
			}
		}
		coh_skymap[grid_id+2*ngrid] = log_prob_margT;
	}

	free(cohfactor);
	gsl_matrix_free(Utrans);
	gsl_matrix_free(detector_real_streams);
	gsl_matrix_free(detector_imag_streams);
	gsl_matrix_free(signal_real_streams);
	gsl_matrix_free(signal_imag_streams);
	gsl_matrix_free(I_dagger);
	gsl_vector_free(null_stream);

	printf("end of calculate coherent snr \n");
	printf("---------------------------------------\n");

	return coh_skymap;
}

double find_max_snr(
				const data_streams *strain_data, 
				const LALDetector *detectors, 
				const double *sigma, 
				const double *ra_grids, 
				const double *dec_grids, 
				int ngrid, 
				double start_time, 
				double end_time, 
				int ntime)
{
	printf("start calculating the sky map by coherent method\n");

	int grid_id,time_id,det_id;
	int Ndet = strain_data->Nstream;
	double max_snr=0, max_snr_onedet=0;
	double max_snr_temp_onedet;
	//find max snr
	for(det_id=0;det_id<Ndet;det_id++){
		max_snr_onedet = 0;

		for(time_id=0; time_id<strain_data->streams[0]->npoint; time_id++){

			max_snr_temp_onedet = creal(strain_data->streams[det_id]->data[time_id])*creal(strain_data->streams[det_id]->data[time_id]) + cimag(strain_data->streams[det_id]->data[time_id])*cimag(strain_data->streams[det_id]->data[time_id]);
			
			if (max_snr_temp_onedet>max_snr_onedet){
				max_snr_onedet = max_snr_temp_onedet;
			}
		}
		max_snr += max_snr_onedet;
	}
	max_snr = sqrt(max_snr);
	//printf("max mf snr: %f\n", max_snr);
	return max_snr;
}

double find_max_cohsnr(
				const data_streams *strain_data, 
				const LALDetector *detectors, 
				const double *sigma, 
				const double *ra_grids, 
				const double *dec_grids, 
				int ngrid, 
				double start_time, 
				double end_time, 
				int ntime)
{
	printf("start calculating the sky map by coherent method\n");

	int grid_id,time_id,det_id;
	int Ndet = strain_data->Nstream;
	double Gsigma[2*Ndet],singular[2];
	double dt = (end_time-start_time)/ntime;
	double ref_gps_time = (start_time + end_time)/2.0;

	//get the U matrix skymap first
	double *cohfactor = coherent_snr_combination_factors_skymap(detectors,sigma,Ndet,ra_grids,dec_grids,ngrid,ref_gps_time);
	//double *Gsigma_temp[Ndet*2];
	//double *coh_skymap = (double*)malloc(sizeof(double)*ngrid*6);//6 skymaps
	//double *coh_skymap_qian = (double*)malloc(sizeof(double)*ngrid*3);//6 skymaps
	printf("ngrid  = %d  \n",ngrid);
	printf("ntime  = %d  \n",ntime);
	printf("ndet   = %d  \n",Ndet);

	gsl_matrix *detector_real_streams = gsl_matrix_calloc(ntime,Ndet);
	gsl_matrix *detector_imag_streams = gsl_matrix_calloc(ntime,Ndet);

	LIGOTimeGPS ligo_gps_time;
	ligo_gps_time.gpsSeconds = (int)(ref_gps_time);
	ligo_gps_time.gpsNanoSeconds = 0;

	double max_snr=0;
	double max_snr_temp;

	double max_onepoint_snr=0;
	double max_onepoint_snr_temp;

	for(grid_id=0;grid_id<ngrid;grid_id++){
		//set parameters
		double ra  = ra_grids[grid_id];
		double dec = dec_grids[grid_id];

		//time shift the data
		for(det_id=0;det_id<Ndet;det_id++){
			double time_shift = XLALTimeDelayFromEarthCenter((detectors[det_id]).location,ra,dec,&ligo_gps_time);
			for(time_id=0;time_id<ntime;time_id++){
				double complex data = step_interpolate_time_series(strain_data->streams[det_id], start_time + time_id*dt + time_shift);
				//double complex data = linear_interpolate_time_series(&strain_data->streams[det_id], start_time + time_id*dt + time_shift);
				//double complex data = quadratic_interpolate_time_series(&strain_data->streams[det_id], start_time + time_id*dt + time_shift);
				gsl_matrix_set(detector_real_streams,time_id,det_id,creal(data));
				gsl_matrix_set(detector_imag_streams,time_id,det_id,cimag(data));
			}
		}
		//find max snr
		
		for(time_id=0;time_id<ntime;time_id++){
			max_onepoint_snr_temp = 0;
			for(det_id=0;det_id<Ndet;det_id++){
				max_onepoint_snr_temp += gsl_matrix_get(detector_real_streams,time_id,det_id)*gsl_matrix_get(detector_real_streams,time_id,det_id);
			}
			for(det_id=0;det_id<Ndet;det_id++){
				max_onepoint_snr_temp += gsl_matrix_get(detector_imag_streams,time_id,det_id)*gsl_matrix_get(detector_imag_streams,time_id,det_id);
			}
			max_onepoint_snr_temp = sqrt(max_onepoint_snr_temp);
			if(max_onepoint_snr_temp > max_onepoint_snr){
				max_onepoint_snr = max_onepoint_snr_temp;
			}
			max_snr_temp = max_onepoint_snr;
		}
		if (max_snr_temp>max_snr){
			max_snr = max_snr_temp;
		}
	}
	return max_snr;
}

double *coherent_skymap_qian(
				const data_streams *strain_data, 
				const LALDetector *detectors, 
				const double *sigma, 
				const double *ra_grids, 
				const double *dec_grids, 
				int ngrid, 
				double start_time, 
				double end_time, 
				int ntime)
{
	printf("start calculating the sky map by coherent method\n");

	int grid_id,time_id,det_id;
	int Ndet = strain_data->Nstream;
	double Gsigma[2*Ndet],singular[2];
	double dt = (end_time-start_time)/ntime;
	double ref_gps_time = (start_time + end_time)/2.0;

	//get the U matrix skymap first
	double *cohfactor = coherent_snr_combination_factors_skymap(detectors,sigma,Ndet,ra_grids,dec_grids,ngrid,ref_gps_time);
	//double *Gsigma_temp[Ndet*2];
	//double *coh_skymap = (double*)malloc(sizeof(double)*ngrid*6);//6 skymaps
	double *coh_skymap_qian = (double*)malloc(sizeof(double)*ngrid*2);//2 skymaps: cohSNR, prob
	printf("ngrid  = %d  \n",ngrid);
	printf("ntime  = %d  \n",ntime);
	printf("ndet   = %d  \n",Ndet);

	gsl_matrix *Utrans = gsl_matrix_alloc(Ndet,2);
	gsl_matrix *detector_real_streams = gsl_matrix_calloc(ntime,Ndet);
	gsl_matrix *detector_imag_streams = gsl_matrix_calloc(ntime,Ndet);
	gsl_matrix *signal_real_streams   = gsl_matrix_calloc(ntime,2);
	gsl_matrix *signal_imag_streams   = gsl_matrix_calloc(ntime,2);

	gsl_matrix *M_prime = gsl_matrix_alloc(2,2);

	gsl_matrix *G_sigma = gsl_matrix_alloc(Ndet,2); //same as previouly defined
	gsl_matrix *G_sigma_transpose = gsl_matrix_alloc(2,Ndet);

	gsl_matrix *J_real_streams   = gsl_matrix_calloc(ntime,2);
	gsl_matrix *J_imag_streams   = gsl_matrix_calloc(ntime,2);

	//gsl_vector *null_stream = gsl_vector_calloc(ntime);
	gsl_matrix *I_dagger    = gsl_matrix_calloc(Ndet,Ndet);
	//gsl_vector_set_zero(null_stream);

	LIGOTimeGPS ligo_gps_time;
	ligo_gps_time.gpsSeconds = (int)(ref_gps_time);
	ligo_gps_time.gpsNanoSeconds = 0;

	//printf("timeseries start time: %f\n", strain_data->streams[0]->start_time);
	double max_mf_snr=find_max_snr(strain_data, detectors, sigma,ra_grids,dec_grids,ngrid,start_time,end_time,ntime);

	/*
    // design noise, bimodal least fit
    double mu_multimodal = 0.00029915*max_mf_snr - 0.0001853;
    double sigma_multimodal = 0.0001759*max_mf_snr + 3.75904e-05;
	double xi = 1/sigma_multimodal/sigma_multimodal;
	double alpha = mu_multimodal*xi;
		
		*/
	
	// O2 noise, bimodal least fit
	//double mu_multimodal = 0.000466361*max_mf_snr - 0.00036214;
    //double sigma_multimodal = 0.00032248*max_mf_snr - 0.0005245;
	//double xi = 1/sigma_multimodal/sigma_multimodal;
	//double alpha = mu_multimodal*xi;
	


	//max_mf_snr = 32.4;
	//GW170817 4096hz data PSD from Chichi
	double mu_multimodal = 0.00045839*max_mf_snr - 0.000733838;
	double sigma_multimodal = 0.0002892833*max_mf_snr - 0.00040155;
	double xi = 1/sigma_multimodal/sigma_multimodal;
	double alpha = mu_multimodal*xi;

	for(grid_id=0;grid_id<ngrid;grid_id++){
		//set parameters
		double ra  = ra_grids[grid_id];
		double dec = dec_grids[grid_id];

		//get transform matrix
		for(det_id=0;det_id<Ndet;det_id++){
			gsl_matrix_set(Utrans,det_id,0,cohfactor[grid_id*Ndet*2+det_id*2+0]);
			gsl_matrix_set(Utrans,det_id,1,cohfactor[grid_id*Ndet*2+det_id*2+1]);
		}

		/* wrong place
		//get G_sigma
		getGsigma_matrix(detectors, sigma, Ndet,ra_grids,dec_grids,ref_gps_time,Gsigma_temp);
		for(det_id=0;det_id<Ndet;det_id++){
			gsl_matrix_set(G_sigma,det_id,0,Gsigma_temp[det_id*2+0]);
			gsl_matrix_set(G_sigma,det_id,1,Gsigma_temp[det_id*2+1]);
		}*/

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
		}

		

		//transform from strain data to null stream
		/*if(Ndet>2){
			calc_null_stream(detector_real_streams,detector_imag_streams,Utrans,I_dagger,null_stream);
		}*/
		
		//calculate sigular value
		getGsigma_matrix(detectors,sigma,Ndet,ra,dec,ref_gps_time,Gsigma);
		svd_Gsigma_get_singular_value(Gsigma,Ndet,NULL,singular);

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
		double signal0_real,signal0_imag,signal1_real,signal1_imag;
		double snr_temp;
		double log_exp_term1,log_exp_term2,log_exp_term,log_exp_term3,log_exp_term4;
		double j_r1,j_r2,j_i1,j_i2;
		double snr_null_temp;
		double log_probability_qian;
		double log_prob_margT_qian=-100;
		double prefactor = log(detMprime);
		double prefactor0 = log(detM0prime);
		coh_skymap_qian[grid_id+0*ngrid]=0;
		coh_skymap_qian[grid_id+1*ngrid]=0;
		//coh_skymap_qian[grid_id+0*ngrid]=0;
		//coh_skymap[grid_id+6*ngrid]=0;
		for(time_id=0;time_id<ntime;time_id++){
			signal0_real = gsl_matrix_get(signal_real_streams,time_id,0);
			signal0_imag = gsl_matrix_get(signal_imag_streams,time_id,0);
			signal1_real = gsl_matrix_get(signal_real_streams,time_id,1);
			signal1_imag = gsl_matrix_get(signal_imag_streams,time_id,1);

			snr_temp = sqrt(signal0_real*signal0_real + signal0_imag*signal0_imag + 
							signal1_real*signal1_real + signal1_imag*signal1_imag);

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

			//log_probability_qian = prefactor + 1.0/2.0*exp_term - gsl_vector_get(null_stream,time_id)/2.0;
			log_probability_qian = log_exp_term;

			log_prob_margT_qian = logsumexp(log_prob_margT_qian,log_probability_qian);

			if(snr_temp>coh_skymap_qian[grid_id+0*ngrid]){//find the maximum snr value accross the ntime
				coh_skymap_qian[grid_id+0*ngrid] = snr_temp;
			}
			/*if(snr_null_temp>coh_skymap_qian[grid_id+1*ngrid]){
				coh_skymap_qian[grid_id+1*ngrid] = snr_null_temp;
			}*/
		}
		//coh_skymap_qian[grid_id+2*ngrid] = log_prob_margT_qian;
		coh_skymap_qian[grid_id+1*ngrid] = log_prob_margT_qian;
	}

	free(cohfactor);
	gsl_matrix_free(Utrans);
	gsl_matrix_free(detector_real_streams);
	gsl_matrix_free(detector_imag_streams);
	gsl_matrix_free(signal_real_streams);
	gsl_matrix_free(signal_imag_streams);
	gsl_matrix_free(I_dagger);

	gsl_matrix_free(M_prime);
	gsl_matrix_free(G_sigma);
	gsl_matrix_free(G_sigma_transpose);
	gsl_matrix_free(J_real_streams);
	gsl_matrix_free(J_imag_streams);


	printf("end of calculate coherent snr \n");
	printf("---------------------------------------\n");

	return coh_skymap_qian;
}

static int *rank_OneOfFour(const double *array, int npoint)
{
	if(npoint%4!=0){
		printf("array size doesn't fit\n");
		exit(-1);
	}

	int i,j=0,sum=0;
	int *argsort = (int*)malloc(sizeof(int)*npoint);
	while(sum!=npoint/4){
		sum=0;
		for(i=0;i<npoint;i++){
			if(array[i]>=array[j]){ 
				argsort[sum] = i;
				sum++;
			}
		}
		printf("%d \n",j);
		j++;
	}
	int *argsort_final = (int*)malloc(sizeof(int)*npoint/4);

	for(i=0;i<npoint/4;i++){
		argsort_final[i] = argsort[i];
	}
	free(argsort);

	return argsort_final;
}

static int *rank_OneOfFour2(const double *array, int npoint)
{
	if(npoint%4!=0){
		printf("array size doesn't fit\n");
		exit(-1);
	}

	double a_max;
	int i,sum=0;
	int *argsort = (int*)malloc(sizeof(int)*npoint/4);
	gsl_vector *vec = gsl_vector_calloc(npoint);
	for(i=0;i<npoint;i++){
		gsl_vector_set(vec,i,array[i]);
	}
	double a_min = gsl_vector_min(vec);
	while(sum<npoint/4){
		argsort[sum] = gsl_vector_max_index(vec);
		gsl_vector_set(vec,argsort[sum],a_min);
		sum++;
	}

	gsl_vector_free(vec);

	return argsort;
}

double *coh_skymap_multiRes_alan(
				const data_streams *strain_data, 
				const LALDetector *detectors, 
				const double *sigma, 
				double start_time, 
				double end_time, 
				int ntime,
				int nlevel)
{
	printf("===============================================================\n");
	printf("start calculate coherent skymap by multi-reselution method  ===\n");
	int i,j,i_level;
	int nside_base = 16;
	//the number fo point for calculate skymap always br npix_base
	int npix_base  = 12*nside_base*nside_base;
	int nside=nside_base*pow(2,nlevel);
	int npix =12*nside*nside;

	printf("base nside = %d \n",nside_base);
	printf("base npix  = %d \n",npix_base);
	printf("nlevel = %d \n",nlevel);
	printf("nside  = %d \n",nside);
	printf("npix   = %d \n",npix);
	printf("---------------------------------------\n");
	
	double *ra_grids  = (double*)malloc(sizeof(double)*npix);
	double *dec_grids = (double*)malloc(sizeof(double)*npix);
	double *ra_grids_calc  = (double*)malloc(sizeof(double)*npix_base);
	double *dec_grids_calc = (double*)malloc(sizeof(double)*npix_base);
	double *coh_skymap_multires = (double*)malloc(sizeof(double)*npix*3);
	create_healpix_skygrids_from_file(nside,ra_grids,dec_grids);
	double *ra_grids_temp,*dec_grids_temp,*coh_skymap;


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

	for(i_level=0;i_level<nlevel+1;i_level++){
		int i_nside = nside_base*pow(2,i_level);
		int i_npix  = 12*i_nside*i_nside;
		ra_grids_temp  = (double*)malloc(sizeof(double)*i_npix);
		dec_grids_temp = (double*)malloc(sizeof(double)*i_npix);
		create_healpix_skygrids_from_file(i_nside,ra_grids_temp,dec_grids_temp);

		for(i=0;i<npix_base;i++){
			ra_grids_calc[i]  = ra_grids_temp[argsort_pix_id[i]];
			dec_grids_calc[i] = dec_grids_temp[argsort_pix_id[i]];
		}

		//calculate skymap
		coh_skymap = coherent_skymap_alan(strain_data,detectors,sigma,ra_grids_calc,dec_grids_calc,npix_base,start_time,end_time,ntime);

		//update skymap
		int nfactor = (int)pow(4,nlevel-i_level);
		int index;
		for(i=0;i<npix_base;i++){
			index = argsort_pix_id[i];
			for(j=0;j<nfactor;j++){
				coh_skymap_multires[index*nfactor+j+0*npix] = coh_skymap[i+0*npix_base];//snr
				coh_skymap_multires[index*nfactor+j+1*npix] = coh_skymap[i+1*npix_base];//snr-null
				coh_skymap_multires[index*nfactor+j+2*npix] = coh_skymap[i+2*npix_base];//log_prob_margT
			}
		}
		printf("finish update skymap \n");

		//rank the pixal with highter probability and copy the index of
		free(argsort);
		argsort = rank_OneOfFour2(&coh_skymap[2*npix_base],npix_base);
		for(i=0;i<npix_base/4;i++){
			argsort_temp[i] = argsort_pix_id[argsort[i]];
		}
		for(i=0;i<npix_base/4;i++){
			for(j=0;j<4;j++){
				argsort_pix_id[i*4+j] = argsort_temp[i]*4+j;
			}
		}
		
		free(coh_skymap);
		free(ra_grids_temp);
		free(dec_grids_temp);
	}


	free(ra_grids);
	free(ra_grids_calc);
	free(dec_grids);
	free(dec_grids_calc);
	free(argsort);
	free(argsort_temp);
	free(argsort_pix_id);
	printf("end of calculate coherent skymap by multi-reselution method ===\n");
	printf("===============================================================\n");

	return coh_skymap_multires;
}


double *coh_skymap_multiRes_qian(
				const data_streams *strain_data, 
				const LALDetector *detectors, 
				const double *sigma, 
				double start_time, 
				double end_time, 
				int ntime,
				int nlevel)
{
	printf("===============================================================\n");
	printf("start calculate coherent skymap by multi-reselution method  ===\n");
	int i,j,i_level;
	int nside_base = 16;
	//the number fo point for calculate skymap always br npix_base
	int npix_base  = 12*nside_base*nside_base;
	int nside=nside_base*pow(2,nlevel);
	int npix =12*nside*nside;

	printf("base nside = %d \n",nside_base);
	printf("base npix  = %d \n",npix_base);
	printf("nlevel = %d \n",nlevel);
	printf("nside  = %d \n",nside);
	printf("npix   = %d \n",npix);
	printf("---------------------------------------\n");
	
	double *ra_grids  = (double*)malloc(sizeof(double)*npix);
	double *dec_grids = (double*)malloc(sizeof(double)*npix);
	double *ra_grids_calc  = (double*)malloc(sizeof(double)*npix_base);
	double *dec_grids_calc = (double*)malloc(sizeof(double)*npix_base);
	double *coh_skymap_multires_qian = (double*)malloc(sizeof(double)*npix*3);
	create_healpix_skygrids_from_file(nside,ra_grids,dec_grids);
	double *ra_grids_temp,*dec_grids_temp,*coh_skymap;


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

	for(i_level=0;i_level<nlevel+1;i_level++){
		int i_nside = nside_base*pow(2,i_level);
		int i_npix  = 12*i_nside*i_nside;
		ra_grids_temp  = (double*)malloc(sizeof(double)*i_npix);
		dec_grids_temp = (double*)malloc(sizeof(double)*i_npix);
		create_healpix_skygrids_from_file(i_nside,ra_grids_temp,dec_grids_temp);

		for(i=0;i<npix_base;i++){
			ra_grids_calc[i]  = ra_grids_temp[argsort_pix_id[i]];
			dec_grids_calc[i] = dec_grids_temp[argsort_pix_id[i]];
		}

		//calculate skymap
		coh_skymap = coherent_skymap_qian(strain_data,detectors,sigma,ra_grids_calc,dec_grids_calc,npix_base,start_time,end_time,ntime);

		//update skymap
		int nfactor = (int)pow(4,nlevel-i_level);
		int index;
		for(i=0;i<npix_base;i++){
			index = argsort_pix_id[i];
			for(j=0;j<nfactor;j++){
				coh_skymap_multires_qian[index*nfactor+j+0*npix] = coh_skymap[i+0*npix_base];//snr
				//coh_skymap_multires_qian[index*nfactor+j+1*npix] = coh_skymap[i+1*npix_base];//snr-null
				coh_skymap_multires_qian[index*nfactor+j+1*npix] = coh_skymap[i+1*npix_base];//log_prob_margT_qian
			}
		}
		printf("finish update skymap \n");

		//rank the pixal with highter probability and copy the index of
		free(argsort);
		argsort = rank_OneOfFour2(&coh_skymap[1*npix_base],npix_base);
		for(i=0;i<npix_base/4;i++){
			argsort_temp[i] = argsort_pix_id[argsort[i]];
		}
		for(i=0;i<npix_base/4;i++){
			for(j=0;j<4;j++){
				argsort_pix_id[i*4+j] = argsort_temp[i]*4+j;
			}
		}
		
		free(coh_skymap);
		free(ra_grids_temp);
		free(dec_grids_temp);
	}


	free(ra_grids);
	free(ra_grids_calc);
	free(dec_grids);
	free(dec_grids_calc);
	free(argsort);
	free(argsort_temp);
	free(argsort_pix_id);
	printf("end of calculate coherent skymap by multi-reselution method ===\n");
	printf("===============================================================\n");

	return coh_skymap_multires_qian;
}
