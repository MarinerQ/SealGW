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

#include <lal/LALDetectors.h>  // /fred/oz016/opt-pipe/include/
#include <lal/LALSimulation.h>
#include <lal/TimeDelay.h>
#include <lal/LALDatatypes.h>


typedef struct tagTimeSeries{
	int npoint;
	double start_time;
	double delta_t;
	double complex *data;
} time_series;

typedef struct tagTimeSeries2{
	int npoint;
	double start_time;
	double delta_t;
	double complex *data;
} time_series2;

typedef struct tagDataStreams{
	int Nstream;
	time_series **streams;
} data_streams;

typedef struct tagDataStreams2{
	int Nstream;
	COMPLEX8TimeSeries **streams;
} data_streams2;

static double max_in_4(double loga, double logb, double logc, double logd);

static double logsumexp(double loga, double logb);

static double logsumexp4(double loga, double logb, double logc, double logd);

static void gsl_matrix_mult(const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *C);

static double quadratic_form(double a1, double a2, double m11, double m12, double m21, double m22);

static double complex step_interpolate_time_series(time_series* timeseries, double time);


static double complex linear_interpolate_time_series(time_series* timeseries, double time);

static double complex quadratic_interpolate_time_series(time_series* timeseries, double time);

static void ComputeDetAMResponse(
		double *fplus,          /**< Returned value of F+ */
		double *fcross,         /**< Returned value of Fx */
		const REAL4 D[3][3],    /**< Detector response 3x3 matrix */
		const double ra,        /**< Right ascention of source (radians) */
		const double dec,       /**< Declination of source (radians) */
		const double psi,       /**< Polarization angle of source (radians) */
		const double gmst       /**< Greenwich mean sidereal time (radians) */
		);

static void getGpc(const LALDetector detector, double ra, double de, double gps_time, double *Gpc);

static void getGsigma_matrix(const LALDetector *detectors,const double *sigma, int Ndet, double ra, double dec, double gps_time,double *Gsigma);

double testfunc1(double a, double b);

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
                double prior_sigma);

