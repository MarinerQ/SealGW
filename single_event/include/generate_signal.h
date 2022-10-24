#ifndef _generate_signal_h
#define _generate_signal_h

#include <gsl/gsl_rng.h>

#include <lal/FrequencySeries.h>
#include <lal/TimeSeries.h>


gsl_rng *setup_gsl_rng(int injection_id);
double source_distance_distribution(double range, gsl_rng *rng);
double sin_distribution(gsl_rng* rng);

REAL8TimeSeries** generate_LHV_data(
				double flow, 
				double duration, 
				double srate, 
				double gps_time,
				int injection_id,
				gsl_rng *rng);

REAL8TimeSeries** generate_LHV_gw(
				double flow, 
				double srate, 
				double gps_time, 
				int injection_id, 
				gsl_rng *rng,
				double *injection_des);

COMPLEX16FrequencySeries* generate_template(
				double flow, 
				double fhigh, 
				double deltaF, 
				int injection_id, 
				double *injection_des);

void lal_injection(
				REAL8TimeSeries **data, 
				REAL8TimeSeries **strain, 
				int Ndet);

void my_injection(
				REAL8TimeSeries **data, 
				REAL8TimeSeries **strain, 
				int Ndet);

#endif
