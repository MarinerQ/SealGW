#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_heapsort.h>
#include <gsl/gsl_permutation.h>

#include <lal/Date.h>

#include <coherent.h>
#include <filter.h>
#include <generate_signal.h>

time_series* lalTimeSeries2time_series(COMPLEX16TimeSeries *lal_timeseries)
{
	int npoint = lal_timeseries->data->length;
	double start_time = XLALGPSGetREAL8(&lal_timeseries->epoch);
	double delta_t    = lal_timeseries->deltaT;
	time_series *my_timeseries = create_time_series(npoint,start_time,delta_t,NULL);

	int i;
	for(i=0;i<npoint;i++){
		my_timeseries->data[i] = lal_timeseries->data->data[i];
	}

	return my_timeseries;
}

data_streams* convert_snr2data_streams(COMPLEX16TimeSeries **snr, int Ndet)
{
	int i_det;
	data_streams *datastreams = create_data_streams(Ndet);
	for(i_det=0;i_det<Ndet;i_det++){
		time_series *det_snr = lalTimeSeries2time_series(snr[i_det]);
		datastreams->streams[i_det] = det_snr;
		free(det_snr);
	}

	return datastreams;
}
//
//void data_streams_destroy(data_streams *datastreams)
//{
//	int i;
//	for(i=0;i<datastreams->Nstream;i++){
//		free(datastreams->streams[i].data);
//	}
//	free(datastreams->streams);
//	free(datastreams);	
//}

//read injection parameter
//injection_des usually has 17 components
void read_injection_des(char *filename, double *injection_des)
{
	printf("txt file from    : %s \n",filename);

	int npoint;

	//read data from txt
	FILE *ft_r = fopen(filename,"r");
	if(ft_r==NULL){
		printf("can not found the file\n");
		exit(-1);
	}
	
	printf("all the parameters is \n");
	npoint=0;
	double des_temp;
	while(fscanf(ft_r,"%lf",&des_temp)!=EOF){
		injection_des[npoint]  = des_temp;
		printf("%lf ",des_temp);
		npoint++;
	}
	fclose(ft_r);

	printf("\nnparameter  = %d  \n",npoint);
	printf("---------------------------------------\n");
}

//the distance between two point on the sky
double distance_2D_sphere(double phi1, double theta1, double phi2, double theta2)
{
	double x1=cos(theta1)*cos(phi1);
	double y1=cos(theta1)*sin(phi1);
	double z1=sin(theta1);
	double x2=cos(theta2)*cos(phi2);
	double y2=cos(theta2)*sin(phi2);
	double z2=sin(theta2);

	return acos(x1*x2+y1*y2+z1*z2);
}

//output the index of nearest point for given sky direction
void localization_point_healpix(const double *coh_skymap, int nside, int *index, double *ra, double *dec)
{
	int i,npix=12*nside*nside;
	double *ra_grids  = (double*)malloc(sizeof(double)*npix);
	double *dec_grids = (double*)malloc(sizeof(double)*npix);
	gsl_vector *pro_density = gsl_vector_alloc(npix);
	for(i=0;i<npix;i++) gsl_vector_set(pro_density,i,coh_skymap[i+1*npix]);
	
	*index = gsl_vector_max_index(pro_density);
	create_healpix_skygrids_from_file(nside,ra_grids,dec_grids);
	*ra  = ra_grids[*index];
	*dec = dec_grids[*index];

	free(ra_grids);
	free(dec_grids);
	gsl_vector_free(pro_density);
}

double healpix_nearby_interpolate(const double *healpix_skymap, int nside, double ra, double dec)
{
	int i,npix =12*nside*nside;
	double *ra_grids  = (double*)malloc(sizeof(double)*npix);
	double *dec_grids = (double*)malloc(sizeof(double)*npix);
	create_healpix_skygrids_from_file(nside,ra_grids,dec_grids);
	gsl_vector *dist  = gsl_vector_alloc(npix);
	for(i=0;i<npix;i++){
		double distance = distance_2D_sphere(ra,dec,ra_grids[i],dec_grids[i]);
		gsl_vector_set(dist,i,distance);
	}
	int index = gsl_vector_min_index(dist);

	free(ra_grids);
	free(dec_grids);
	gsl_vector_free(dist);

	return healpix_skymap[index];
}

//just finding the maximum point
double max_coherent_snr(const double *coh_skymap, int npix)
{
	double a_max=0;
	int i;
	for(i=0;i<npix;i++){
		if(a_max<coh_skymap[i+0*npix]) a_max=coh_skymap[i+0*npix];
	}

	return a_max;
}

int gsl_sort_vector_index(gsl_permutation *p, const gsl_vector *v);

// calculate the area of certain confident level
double exp_cumulative_skyarea(const double *coh_skymap, int nside, double ratio)
{
	int npix = 12*nside*nside;
	int i,check=1;
	double pro_max,area=0,cumsum=0,allsum=0;
	gsl_vector *work = gsl_vector_alloc(npix);
	gsl_permutation *p = gsl_permutation_alloc(npix);
	for(i=0;i<npix;i++){
		gsl_vector_set(work,i,coh_skymap[i+1*npix]);
	}
	int err=gsl_sort_vector_index(p,work);//elements in work from small to large stored in p through index
	pro_max = gsl_vector_max(work);

	for(i=0;i<npix;i++) allsum += exp(coh_skymap[gsl_permutation_get(p,i)+1*npix]-pro_max);
	for(i=0;i<npix && check==1;i++){
		cumsum += exp(coh_skymap[gsl_permutation_get(p,i)+1*npix]-pro_max);
		if(cumsum/allsum>(1-ratio)){ 
			check=0;
			area = (npix-i)/(double)npix;
		}
	}

	gsl_vector_free(work);
	gsl_permutation_free(p);

	return area;//unit is pixel
}

//calculate the confident level at certain point
double exp_cumulative_percentage(const double *coh_skymap, int nside, double ra, double dec)
{
	int i,npix=12*nside*nside;
	int check=1;
	double pro_max,cumsum=0,allsum=0;
	double value = healpix_nearby_interpolate(&coh_skymap[1*npix],nside,ra,dec);
	gsl_vector *work = gsl_vector_alloc(npix);
	gsl_permutation *p = gsl_permutation_alloc(npix);
	for(i=0;i<npix;i++){
		gsl_vector_set(work,i,coh_skymap[i+1*npix]);
	}
	gsl_sort_vector_index(p,work);
	pro_max = gsl_vector_max(work);

	for(i=0;i<npix;i++) allsum += exp(coh_skymap[gsl_permutation_get(p,i)+1*npix]-pro_max);
	for(i=0;i<npix && check==1;i++){
		double pro = coh_skymap[gsl_permutation_get(p,i)+1*npix];
		cumsum += exp(pro-pro_max);
		if(pro>value) check=0;
	}

	double cl = (1-cumsum/allsum)*100;

	gsl_vector_free(work);
	gsl_permutation_free(p);

	return cl;//unit: %
}

//calculate the search area
double exp_cumulative_search_area(const double *coh_skymap, int nside, double ra, double dec)
{
	int i,npix=12*nside*nside;
	int check=1;
	double pro_max,cumsum=0,allsum=0;
	double area;
	double value = healpix_nearby_interpolate(&coh_skymap[1*npix],nside,ra,dec);
	gsl_vector *work = gsl_vector_alloc(npix);
	gsl_permutation *p = gsl_permutation_alloc(npix);
	for(i=0;i<npix;i++){
		gsl_vector_set(work,i,coh_skymap[i+1*npix]);
	}
	gsl_sort_vector_index(p,work);
	pro_max = gsl_vector_max(work);

	for(i=0;i<npix;i++) allsum += exp(coh_skymap[gsl_permutation_get(p,i)+1*npix]-pro_max);
	for(i=0;i<npix && check==1;i++){
		double pro = coh_skymap[gsl_permutation_get(p,i)+1*npix];
		cumsum += exp(pro-pro_max);
		if(pro>value){ 
			check=0;
			area = (npix-i)/((double)npix);
		}
	}

	gsl_vector_free(work);
	gsl_permutation_free(p);

	return area;//unit : pixel
}


//********************generate skymap and statistic********************//
//In this part, we calculate the PDF skymap and do some statistic test.
//This part need to use the output of snr_generator.
//If you don't change the snr_generator, the event time is on 1234568146.
//ndata   :  can not be larger than the number of data you have.
//Ndet    : Don't change this!
//ntime   : The number of sample point between start_time and end_time.
//nlevel  : the maximum reselution of skymap.
//nthread : The number of thread you want.
//save_skymap    : the option for you to save the skymap.
//save_statistic : the option for you to save the statistic result.
//*******************************end**********************************//
void coherent_analysis()
{
	load_healpix_skygrids_from_file();

	double event_info[32]; 
	char event_info_filename[128];
	sprintf(event_info_filename,"data/event_info");
	read_event_info(event_info_filename, event_info);

	double trigger_time = event_info[0];
	double dt = 1/2048.0;
	//dt = 1/4096.0;
	double start_time = trigger_time - 0.01;
	//start_time = 1187008882.42;
	double end_time   = trigger_time + 0.01;
	//end_time = 1187008882.44;
	int ntime = (int) 1/dt*0.02*10; // say, 2048Hz * 20ms *10


	
	int Ndet= (int) event_info[1];
	int detnum, detcode;
	LALDetector detectors[Ndet];
	LALDetector tempdet;
	//char l1_snr_filename[128];
	//char h1_snr_filename[128];
	//char v1_snr_filename[128];
	char snr_filenames[Ndet][128];
	double sigma[Ndet];
	double maxsnr[Ndet];
	for(detnum=0; detnum<Ndet; detnum++){
		// load detectors
		detcode = (int)event_info[2+detnum];
		//printf("\n\ndetcode: %d\n\n", detcode);
		tempdet = lalCachedDetectors[detcode];
		detectors[detnum] = tempdet;

		// read SNR filenames
		//sprintf(snr_filenames[detnum],"data/snr_data/old_test_data/4096data/snr_det%d",detcode);
		sprintf(snr_filenames[detnum],"data/snr_data/snr_det%d",detcode);

		// read max SNRs
		maxsnr[detnum] = event_info[2+detnum+Ndet];
		
		// read sigmas
		sigma[detnum] = event_info[2+detnum+2*Ndet];
	}
	
	

	// generate data streams
	data_streams *strain_data = create_data_streams(Ndet);
	time_series *temp_snr;
	for(detnum=0; detnum<Ndet; detnum++){
		temp_snr = readsnr2time_series(snr_filenames[detnum]);
		strain_data->streams[detnum] = temp_snr;
	}
	//time_series *l1_snr = readsnr2time_series(l1_snr_filename);
	//time_series *h1_snr = readsnr2time_series(h1_snr_filename);
	//time_series *v1_snr = readsnr2time_series(v1_snr_filename);

	//COMPLEX8TimeSeries *l1_snr_lal = 
	//strain_data->streams[0] = l1_snr;
	//strain_data->streams[1] = h1_snr;
	//strain_data->streams[2] = v1_snr;



	
	int ndata=1;
	// What is the finest resolution you want
	int nlevel=6;
	// How many threads you want to use?
	int nthread=1;
	int nside = 16*pow(2,nlevel);
	int npix  = 12*nside*nside;
	
	
	
	
	//statistic
	double statis_cohsnr_qian[ndata];

	double statis_area90_qian[ndata];

	double statis_area50_qian[ndata];

	double statis_area_search_qian[ndata];

	double statis_location_qian[ndata*2];

	double statis_distance_qian[ndata];

	double statis_percentage_qian[ndata];

	//the option of saving data
	int save_skymap=1;
	int save_statistic=0;


	#pragma omp parallel num_threads(nthread)
	{
	int i_thread = omp_get_thread_num();
	int i, i_data = 0;
	char skymap_filename[128];
	sprintf(skymap_filename,"skymap/skymap_paper.txt");
	

	//calculate the skymap in adaptive way with multi-resolution
	//double *coh_skymap_multires_alan = coh_skymap_multiRes_alan(strain_data,detectors,sigma,start_time,end_time,ntime,nlevel);
	//printf("timeseries start time: %f", strain_data->streams[0]->start_time);
	double *coh_skymap_multires_qian = coh_skymap_multiRes_qian(strain_data,detectors,sigma,start_time,end_time,ntime,nlevel);
	
	//statistic analysis
	double ra_max_qian,dec_max_qian;
	double ra_inj  = 3.446;
	double dec_inj = -0.408;
	int index_max_qian;
	localization_point_healpix(coh_skymap_multires_qian,nside,&index_max_qian,&ra_max_qian,&dec_max_qian);
	printf("time prior: from %.2lf to %.2lf \n", start_time, end_time);
	printf("true (ra dec): %lf %lf deg\n",ra_inj*57.3,dec_inj*57.3);
	printf("max prob pixel (ra dec): %lf %lf deg\n",ra_max_qian*57.3,dec_max_qian*57.3);
	double max_cohsnr_qian  = max_coherent_snr(coh_skymap_multires_qian,npix);
	printf("max cohsnr %f\n", max_cohsnr_qian);

	double area_90_qian     = exp_cumulative_skyarea(coh_skymap_multires_qian,nside,0.9)*4/M_PI*180*180; //transform pixel to degree^2
	printf("90 confidence area = %lf deg^2\n", area_90_qian);

	double area_50_qian     = exp_cumulative_skyarea(coh_skymap_multires_qian,nside,0.5)*4/M_PI*180*180;
	printf("50 confidence area = %lf deg^2\n", area_50_qian);

	double area_search_qian = exp_cumulative_search_area(coh_skymap_multires_qian,nside,ra_inj,dec_inj)*4/M_PI*180*180;
	printf("search area = %lf deg^2\n", area_search_qian);

	double deviation_qian   = distance_2D_sphere(ra_inj,dec_inj,ra_max_qian,dec_max_qian);
	printf("deviation from max pixel to true location = %lf deg\n", deviation_qian*57.3);

	double percentage_qian  = exp_cumulative_percentage(coh_skymap_multires_qian,nside,ra_inj,dec_inj);

	statis_cohsnr_qian[i_data] = max_cohsnr_qian;

	statis_area90_qian[i_data] = area_90_qian;

	statis_area50_qian[i_data] = area_50_qian;

	statis_area_search_qian[i_data] = area_search_qian;

	statis_location_qian[2*i_data]   = ra_max_qian;
	statis_location_qian[2*i_data+1] = dec_max_qian;

	statis_percentage_qian[i_data]   = percentage_qian;

	statis_distance_qian[i_data] = deviation_qian;

	//save data
	
	if(save_skymap){
		printf("writing data to %s\n",skymap_filename);
		FILE *skymap_snr_multires = fopen(skymap_filename,"w");
		for(i=0;i<npix;i++){
			fprintf(skymap_snr_multires,"%e %e\n"
											,coh_skymap_multires_qian[i+0*npix]//snr
											//,coh_skymap_multires_qian[i+1*npix]//snr-null
											,coh_skymap_multires_qian[i+1*npix]//log_prob_margT
											);
		}
		fclose(skymap_snr_multires);
	}

	free_data_streams(strain_data);
	free(coh_skymap_multires_qian);
	
	}

	if(save_statistic){
		int i;
		FILE *statis_cohsnr_file_qian = fopen("statistic_cohsnr_qian_updated_factor.txt","w");

		FILE *statis_area90_file_qian = fopen("statistic_area90_qian_updated_factor.txt","w");

		FILE *statis_area50_file_qian = fopen("statistic_area50_qian_updated_factor.txt","w");

		FILE *statis_area_search_file_qian = fopen("statistic_area_search_qian_updated_factor.txt","w");

		FILE *statis_location_file_qian = fopen("statistic_location_qian_updated_factor.txt","w");

		FILE *statis_distance_file_qian = fopen("statistic_distance_qian_updated_factor.txt","w");

		FILE *statis_percenta_file_qian = fopen("statistic_percentage_qian_updated_factor.txt","w");

		for(i=0;i<ndata;i++){
			fprintf(statis_cohsnr_file_qian,"%.18e\n",statis_cohsnr_qian[i]);

			fprintf(statis_area90_file_qian,"%.18e\n",statis_area90_qian[i]);

			fprintf(statis_area50_file_qian,"%.18e\n",statis_area50_qian[i]);

			fprintf(statis_area_search_file_qian,"%.18e\n",statis_area_search_qian[i]);

			fprintf(statis_location_file_qian,"%.18e %.18e\n",statis_location_qian[i*2],statis_location_qian[i*2+1]);

			fprintf(statis_distance_file_qian,"%.18e\n",statis_distance_qian[i]);

			fprintf(statis_percenta_file_qian,"%.18e\n",statis_percentage_qian[i]);
		}

		fclose(statis_cohsnr_file_qian);
		fclose(statis_area90_file_qian);
		fclose(statis_area50_file_qian);
		fclose(statis_area_search_file_qian);
		fclose(statis_location_file_qian);
		fclose(statis_distance_file_qian);
		fclose(statis_percenta_file_qian);
	}
}

int main(int argc,char *argv[])
{
	int i;
	//for(i=0;i<argc;i++) printf("%d\n",atoi(argv[i]));
	//snr_generator(atoi(argv[1]));
	coherent_analysis();
}
