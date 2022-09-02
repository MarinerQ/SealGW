#ifndef _coherent_h
#define _coherent_h

#include <lal/LALDetectors.h>

typedef struct tagTimeSeries{
	int npoint;
	double start_time;
	double delta_t;
	double complex *data;
} time_series;

typedef struct tagDataStreams{
	int Nstream;
	time_series **streams;
} data_streams;

data_streams* create_data_streams(int Ndet);

time_series* create_time_series(
						int npoint, 
						double start_time, 
						double delta_t, 
						double complex *data);

void free_time_series(time_series *timeseries);

void free_data_streams(data_streams *datastreams);

time_series* readsnr2time_series(char *filename);

void read_skygrids(char *filename, double *ra, double *dec);

void read_sigma(char *filename, double *sigma);

void create_healpix_skygrids_from_file(
				int nside, 
				double *ra_grids, 
				double *dec_grids);

void load_healpix_skygrids_from_file();

double *coherent_skymap_alan(
				const data_streams *strain_data, 
				const LALDetector *detectors, 
				const double *sigma, 
				const double *ra_grids, 
				const double *dec_grids, 
				int ngrid, 
				double start_time, 
				double end_time, 
				int ntime);

double *coherent_skymap_qian(
				const data_streams *strain_data, 
				const LALDetector *detectors, 
				const double *sigma, 
				const double *ra_grids, 
				const double *dec_grids, 
				int ngrid, 
				double start_time, 
				double end_time, 
				int ntime);

double *coh_skymap_multiRes_alan(
				const data_streams *strain_data, 
				const LALDetector *detectors, 
				const double *sigma, 
				double start_time, 
				double end_time, 
				int ntime,
				int nlevel);

double *coh_skymap_multiRes_qian(
				const data_streams *strain_data, 
				const LALDetector *detectors, 
				const double *sigma, 
				double start_time, 
				double end_time, 
				int ntime,
				int nlevel);

#endif
