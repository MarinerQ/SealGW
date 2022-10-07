#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>

#include <lal/Units.h>
#include <lal/FrequencySeries.h>
#include <lal/TimeSeries.h>
#include <lal/LALConstants.h>
#include <lal/TimeFreqFFT.h>

#include <generate_signal.h>

void psd_interpolate(REAL8FrequencySeries **psd, double srate, double seg_time, double duration, int Ndet)
{
    double deltaF     = 1.0/seg_time;
    double deltaF_int = 1.0/duration;
    int length     = (int)(seg_time*srate);
    int length_int = (int)(duration*srate);

    REAL8FrequencySeries **psd_int = (REAL8FrequencySeries**)malloc(sizeof(REAL8FrequencySeries*)*Ndet);

    int i_det,i_point;
    for(i_det=0;i_det<Ndet;i_det++){
        int index=0;
        double ratio,f=0;
        psd_int[i_det] = XLALCreateREAL8FrequencySeries("PSD",&(psd[i_det]->epoch),0.0,1.0/duration,&lalSecondUnit,length_int/2+1);
        for(i_point=0;i_point<(length_int/2+1);i_point++){
            f = i_point*deltaF_int;
            index = (int)(f/deltaF);
            ratio = f/deltaF-index;
            if(index==(length/2+1)) psd_int[i_det]->data->data[i_point] = psd[i_det]->data->data[index];
            else psd_int[i_det]->data->data[i_point] = psd[i_det]->data->data[index]*(1-ratio)+psd[i_det]->data->data[index+1]*ratio;
        }

        XLALDestroyREAL8FrequencySeries(psd[i_det]);
        psd[i_det] = psd_int[i_det];
    }

    free(psd_int);
}

REAL8FrequencySeries** psd_estimator(
                REAL8TimeSeries **data, 
                double seg_time, 
                double stride_time,
                double srate,
                int Ndet)
{
    printf("=================================================\n");
    printf("psd_estimator is starting ...\n");
    printf("segment time = %lf\n",seg_time);
    printf("stride  time = %lf\n",stride_time);
    printf("number of det= %d\n",Ndet);
    int seglen = (int)(seg_time*srate);
    int stride = (int)(stride_time*srate);
    REAL8FrequencySeries **psd = (REAL8FrequencySeries**)malloc(sizeof(REAL8FrequencySeries*)*Ndet);
    REAL8Window *window = XLALCreateWelchREAL8Window(seglen);
    REAL8FFTPlan *fft_plan = XLALCreateREAL8FFTPlan(seglen,1,0);

    int i_det;
    for(i_det=0;i_det<Ndet;i_det++){
        psd[i_det] = XLALCreateREAL8FrequencySeries("PSD",&(data[i_det]->epoch),0.0,1.0/seg_time,&lalSecondUnit,seglen/2+1);
        int reclen = data[i_det]->data->length;
        int numseg = 1+(reclen-seglen)/stride;
        printf("------------------------------------------\n");
        printf("start estimate the power spetrum density\n");
        printf("reclen = %d\n",reclen);
        printf("seglen = %d\n",seglen);
        printf("stride = %d\n",stride);
        printf("numseg = %d\n",numseg);
        XLALREAL8AverageSpectrumMedian(psd[i_det],data[i_det],seglen,stride,window,fft_plan);
    }

    XLALDestroyREAL8Window(window);
    XLALDestroyREAL8FFTPlan(fft_plan);

    printf("end of psd_estimator\n");
    printf("=================================================\n");

    return psd;
}

COMPLEX16TimeSeries** matched_filter(
                REAL8TimeSeries **data_time, 
                COMPLEX16FrequencySeries *template, 
                REAL8FrequencySeries **psd, 
                double flow_cut, 
                double fhigh_cut,
                int Ndet,
                double *sigma)
{
    int i_det;
    int length = data_time[0]->data->length;
    double deltaT   = data_time[0]->deltaT;
    double duration = deltaT*length;
    double deltaF   = 1.0/duration; 
    LIGOTimeGPS epoch = data_time[0]->epoch;
    int flow_index = (int)(flow_cut/deltaF+1);
    int fhigh_index = (int)(fhigh_cut/deltaF);
    
    printf("=================================================\n");
    printf("matched_filter is starting ...\n");
    printf("data     length = %d\n",length/2+1);
    printf("psd      length = %d\n",psd[0]->data->length);
    printf("template length = %d\n",template->data->length);
    printf("flow index      = %d\n",flow_index);
    printf("fhigh index     = %d\n",fhigh_index);
    printf("------------------------------\n");

    //create working place
    COMPLEX16TimeSeries **snr_time  = (COMPLEX16TimeSeries**)malloc(sizeof(COMPLEX16TimeSeries*)*Ndet);
    COMPLEX16FrequencySeries **snr_freq  = (COMPLEX16FrequencySeries**)malloc(sizeof(COMPLEX16FrequencySeries*)*Ndet);
    COMPLEX16FrequencySeries **data_freq  = (COMPLEX16FrequencySeries**)malloc(sizeof(COMPLEX16FrequencySeries*)*Ndet);
    REAL8FFTPlan *fft_forward_plan = XLALCreateForwardREAL8FFTPlan(length,0);
    COMPLEX16FFTPlan *fft_reverse_plan = XLALCreateReverseCOMPLEX16FFTPlan(length/2+1,0);
    for(i_det=0;i_det<Ndet;i_det++){
        snr_time[i_det] = XLALCreateCOMPLEX16TimeSeries("SNR TS",&epoch,0.0,deltaT*2,&lalDimensionlessUnit,length/2+1);
        snr_freq[i_det] = XLALCreateCOMPLEX16FrequencySeries("SNR FS",&epoch,0.0,1.0/duration,&lalDimensionlessUnit,length/2+1);
        data_freq[i_det] = XLALCreateCOMPLEX16FrequencySeries("SNR FS",&epoch,0.0,1.0/duration,&lalDimensionlessUnit,length/2+1);
    }

    //from time series data to frequency series data
    for(i_det=0;i_det<Ndet;i_det++){
        XLALREAL8TimeFreqFFT(data_freq[i_det],data_time[i_det],fft_forward_plan);
    }

    //matched filter
    for(i_det=0;i_det<Ndet;i_det++){
        if(psd[i_det]->data->length!=data_freq[i_det]->data->length){
            printf("data and psd are not fit \n");
            exit(-1);
        }

        int i_point;
        int fmax_index = fhigh_index;
        if(fmax_index>template->data->length) fmax_index = template->data->length;
        
        double complex data,temp,psdv;
        sigma[i_det]=0;
        
        for(i_point=flow_index;i_point<fmax_index;i_point++){
            temp = template->data->data[i_point];
            psdv = psd[i_det]->data->data[i_point];
            sigma[i_det] += 4*creal(temp*conj(temp))/psdv*deltaF;
        }
        sigma[i_det] = sqrt(sigma[i_det]);
        printf("the horizon distance for %d detector is %lf \n",i_det,sigma[i_det]);
        
        for(i_point=flow_index;i_point<fmax_index;i_point++){
            data = data_freq[i_det]->data->data[i_point];
            temp = template->data->data[i_point];
            psdv = psd[i_det]->data->data[i_point];
            snr_freq[i_det]->data->data[i_point] = 4*data*conj(temp)/psdv/sigma[i_det]*deltaF;
        }
        for(i_point=0;i_point<flow_index;i_point++) snr_freq[i_det]->data->data[i_point]=0;
        for(i_point=fmax_index;i_point<(length/2+1);i_point++) snr_freq[i_det]->data->data[i_point]=0;
    }
    if(0){
        FILE *snr_freq_file=fopen("snr_freq.txt","w");
        int i_point;
        for(i_point=0;i_point<(length/2+1);i_point++){
            for(i_det=0;i_det<Ndet;i_det++){
                fprintf(snr_freq_file,"%e %e ",creal(snr_freq[i_det]->data->data[i_point]),cimag(snr_freq[i_det]->data->data[i_point]));
            }
            fprintf(snr_freq_file,"\n");
        }
    }

    //from frequency series snr to time series snr
    //for(i_det;i_det<Ndet;i_det++){
    //    XLALCOMPLEX16FreqTimeFFT(snr_time[i_det],snr_freq[i_det],fft_reverse_plan);
    //}
    
    if(1){
        int i_point;
        fftw_complex *in,*out;
        fftw_plan p;
        in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(length/2+1));
        out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*(length/2+1));
        p = fftw_plan_dft_1d(length/2+1,in,out,FFTW_BACKWARD,FFTW_ESTIMATE);
        for(i_det=0;i_det<Ndet;i_det++){
            for(i_point=0;i_point<(length/2+1);i_point++){
                in[i_point] = snr_freq[i_det]->data->data[i_point];
            }
            fftw_execute(p);
            
            for(i_point=0;i_point<(length/2+1);i_point++){
                snr_time[i_det]->data->data[i_point] = out[i_point];
            }
        }
        fftw_destroy_plan(p);
        fftw_free(in);
        fftw_free(out);
    }

    //free memory
    XLALDestroyREAL8FFTPlan(fft_forward_plan);
    XLALDestroyCOMPLEX16FFTPlan(fft_reverse_plan);
    for(i_det=0;i_det<Ndet;i_det++){
        XLALDestroyCOMPLEX16FrequencySeries(snr_freq[i_det]);
        XLALDestroyCOMPLEX16FrequencySeries(data_freq[i_det]);
    }
    free(snr_freq);
    free(data_freq);

    printf("end of matched_filater\n");
    printf("=================================================\n");

    return snr_time;
}

int test_memory_leackage()
//int main()
{
    int injection_id;
	int Ninjection=100;
    
	for(injection_id=0;injection_id<Ninjection;injection_id++){
	gsl_rng *rng = setup_gsl_rng(injection_id);
	//fake data
    double flow=10.0;
    double fhigh=1500.0;
    double duration=512.0;
    double srate=4096.0;
    double gps_time=1234567890;
    int Ndet=3;
    int i,npoint=(int)(duration*srate);

    //psd
    double seg_time = 32;
    double stride_time = 8;

    //print data
    int save_fakedata=0;
    int save_strain  =0;
    int save_psd     =0;
    int save_snr     =0;

    double injection_des[17];
    REAL8TimeSeries **fake_data;
    REAL8TimeSeries **strain_data;
    REAL8FrequencySeries  **psd;
    COMPLEX16FrequencySeries *template;
    COMPLEX16TimeSeries **snr;

    //generate fake data
    fake_data  = generate_LHV_data(flow,duration,srate,gps_time,injection_id,rng);

    //generate strain data for LHV detectors
    strain_data = generate_LHV_gw(flow,srate,gps_time+duration/2,injection_id,rng,injection_des);

    //injection
    my_injection(fake_data,strain_data,Ndet);

    //estimate psd from fake data
    psd = psd_estimator(fake_data,seg_time,stride_time,srate,Ndet);
    psd_interpolate(psd,srate,seg_time,duration,Ndet);
    
    //generate template
    template = generate_template(flow,0,1.0/duration,injection_id,injection_des);

    //matched filter
    double sigma[Ndet];
    snr = matched_filter(fake_data,template,psd,flow,fhigh,Ndet,sigma);

    //save data
    if(save_fakedata){
        FILE *fake_data_file = fopen("fake_data.txt","w");
        for(i=0;i<npoint;i++){
            double t = gps_time+i/srate;
            double data_l1 = fake_data[0]->data->data[i];
            double data_h1 = fake_data[1]->data->data[i];
            double data_v1 = fake_data[2]->data->data[i];
            
            fprintf(fake_data_file,"%.18e %e %e %e\n",t,data_l1,data_h1,data_v1);
        }
        fclose(fake_data_file);
    }
    if(save_strain){
        FILE *strain_file=fopen("strain.txt","w");
        int length = strain_data[0]->data->length;
        int t0 = strain_data[0]->epoch.gpsSeconds;
        printf("gps time %d\n",strain_data[0]->epoch.gpsSeconds);
        for(i=0;i<length;i++){
            double t = strain_data[0]->deltaT*i;
            double strain = strain_data[0]->data->data[i];
            fprintf(strain_file,"%.18e %e\n",t,strain);
        }
        fclose(strain_file);
    }
    if(save_psd){
        int length=npoint/2+1;
        FILE *psd_file=fopen("psd.txt","w");
        for(i=0;i<length;i++){
            double f = psd[0]->f0+i*psd[0]->deltaF;
            double psd_l1 = psd[0]->data->data[i];
            double psd_h1 = psd[1]->data->data[i];
            double psd_v1 = psd[2]->data->data[i];

            fprintf(psd_file,"%e %e %e %e\n",f,psd_l1,psd_h1,psd_v1);
        }
        fclose(psd_file);
    }
    if(save_snr){
        int length = snr[0]->data->length;
        FILE *snr_file=fopen("snr.txt","w");
        for(i=0;i<length;i++){
            double t = gps_time+i*2.0/srate;
            double snr_l1 = cabs(snr[0]->data->data[i]);
            double snr_h1 = cabs(snr[1]->data->data[i]);
            double snr_v1 = cabs(snr[2]->data->data[i]);

            fprintf(snr_file,"%.18e %e %e %e\n",t,snr_l1,snr_h1,snr_v1);
        }
        fclose(snr_file);

        int i_det;
        for(i_det=0;i_det<Ndet;i_det++){
            double mean=0;
            double var =0;
            for(i=length/4;i<length*3/4;i++){
                mean += creal(snr[i_det]->data->data[i])/(length/2);
                var  += (creal(snr[i_det]->data->data[i])*creal(snr[i_det]->data->data[i]))/(length/2);
            }
            var = var-mean*mean;
            printf("Detector ID %d mean = %.12lf\n",i_det,mean);
            printf("Detrctor ID %d var  = %.12lf\n",i_det,var);
        }
    }


    
    //free data
    for(i=0;i<Ndet;i++){
        XLALDestroyREAL8TimeSeries(fake_data[i]);
        XLALDestroyREAL8TimeSeries(strain_data[i]);
        XLALDestroyREAL8FrequencySeries(psd[i]);
		XLALDestroyCOMPLEX16TimeSeries(snr[i]);
    }
	XLALDestroyCOMPLEX16FrequencySeries(template);
    free(fake_data);
    free(strain_data);
    free(psd);
	free(snr);
	gsl_rng_free(rng);
	}

    return 0;
}
