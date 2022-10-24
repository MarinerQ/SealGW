#ifndef _filter_h
#define _filter_h

#include <lal/FrequencySeries.h>
#include <lal/TimeSeries.h>

void psd_interpolate(
				REAL8FrequencySeries **psd, 
				double srate, 
				double seg_time, 
				double duration, 
				int Ndet);

REAL8FrequencySeries** psd_estimator(
                REAL8TimeSeries **data, 
                double seg_time, 
                double stride_time,
                double srate,
                int Ndet);

COMPLEX16TimeSeries** matched_filter(
                REAL8TimeSeries **data_time, 
                COMPLEX16FrequencySeries *template, 
                REAL8FrequencySeries **psd, 
                double flow_cut, 
                double fhigh_cut,
                int Ndet,
                double *sigma);

#endif
