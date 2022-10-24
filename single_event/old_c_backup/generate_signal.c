#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_rng.h>
#include <math.h>

#include <lal/LALStdlib.h>
#include <lal/FrequencySeries.h>
#include <lal/TimeSeries.h>
#include <lal/Units.h>
#include <lal/LALSimNoise.h>
#include <lal/Date.h>
#include <lal/LALSimulation.h>
#include <lal/LALSimInspiral.h>
#include <lal/LALConstants.h>
#include <lal/LALDict.h>

#include <generate_signal.h>
#include <filter.h>

gsl_rng *setup_gsl_rng(int injection_id)
{
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(rng,injection_id);
	
	return rng;
}

double prob_uni(double x,double range)
{
    double y;
    y = (x*range)*(x*range)/(range*range);
    return y;
}

double my_r_distribution(double range, gsl_rng *rng)
{
    int check=1;
    double r;
    while(check){
        double x = gsl_rng_uniform_pos(rng);
        double y = gsl_rng_uniform_pos(rng);
        if(y<=prob_uni(x,range)){
            check=0;
            r = x*range;
        }
    }
    return r;
}

double source_distance_distribution(double range, gsl_rng *rng)
{
	int check=1;
	double r;
	while(check){
		double x = gsl_rng_uniform_pos(rng)*range;
		double y = gsl_rng_uniform_pos(rng)*range;
		double z = gsl_rng_uniform_pos(rng)*range;
		if(x*x+y*y+z*z<range*range){
			check=0;
			r = sqrt(x*x+y*y+z*z);
		}
	}
	return r;
}

double sin_distribution(gsl_rng *rng)
{
	int check=1;
	double c,r;
	while(check){
		double x = gsl_rng_uniform_pos(rng)*2-1;
		double y = gsl_rng_uniform_pos(rng)*2-1;
		double z = gsl_rng_uniform_pos(rng)*2-1;
		if(x*x+y*y+z*z<1){
			check=0;
			r = sqrt(x*x+y*y+z*z);
			c = z/r;
		}
	}
	return acos(c);
}

static void MySimNoisePSDLIGOO2Sensitivity(REAL8FrequencySeries *psd_l1, REAL8FrequencySeries *psd_h1, REAL8FrequencySeries *psd_v1, double flow)
{
	int i,npoint=131073;
	FILE *psd_lh_file = fopen("LH_psd","r");
	FILE *psd_v_file  = fopen("V_psd","r");
	double *freq = (double*)malloc(sizeof(double)*npoint);
	double *psd_l1_temp = (double*)malloc(sizeof(double)*npoint);
	double *psd_h1_temp = (double*)malloc(sizeof(double)*npoint);
	double *psd_v1_temp = (double*)malloc(sizeof(double)*npoint);
	if(psd_l1->data->length!=psd_h1->data->length){
		printf("psd_l1 and psd_h1 must have the same length!\n");
		exit(-1);
	}
	else if(psd_l1->deltaF!=psd_h1->deltaF){
		printf("psd_l1 and psd_h1 must have the same deltaF\n");
		exit(-1);
	}

	double f,data;
	for(i=0;i<npoint*2;i++){
		fscanf(psd_lh_file,"%lf %lf",&f,&data);
		if(i<npoint){
			psd_h1_temp[i]=data;
			freq[i]=f;
		}
		else psd_l1_temp[i-npoint]=data;
	}

	for(i=0;i<npoint;i++){
		fscanf(psd_v_file,"%lf %lf",&f,&data);
		psd_v1_temp[i]=data;
	}

	int index;
	double delta_f=freq[1]-freq[0];
	for(i=0;i<psd_l1->data->length;i++){
		f = i*psd_l1->deltaF;
		index=(int)(f/delta_f);
		psd_l1->data->data[i]=psd_l1_temp[index];
		psd_h1->data->data[i]=psd_h1_temp[index];
		psd_v1->data->data[i]=psd_v1_temp[index];
		if(f<flow){
			psd_l1->data->data[i]=0;
			psd_h1->data->data[i]=0;
			psd_v1->data->data[i]=0;
		}
	}

	fclose(psd_lh_file);
	free(freq);
	free(psd_l1_temp);
	free(psd_h1_temp);
}

REAL8TimeSeries** generate_LHV_data(double flow, double duration, double srate, double gps_time, int injection_id, gsl_rng *rng)
{
    printf("=================================================\n");
    printf("generate_LHV_data is starting ...\n");
    int real_psd_h1=0;
    int save_th_psd=1;
    size_t length = (size_t)(duration*srate);
    LIGOTimeGPS epoch = {(int)gps_time,0};
    REAL8FrequencySeries *psd_l1,*psd_h1,*psd_v1;
    REAL8TimeSeries **seg = (REAL8TimeSeries**)malloc(sizeof(REAL8TimeSeries*)*3);

    seg[0] = XLALCreateREAL8TimeSeries("STRAIN",&epoch,0.0,1.0/srate,&lalStrainUnit,length);
    seg[1] = XLALCreateREAL8TimeSeries("STRAIN",&epoch,0.0,1.0/srate,&lalStrainUnit,length);
    seg[2] = XLALCreateREAL8TimeSeries("STRAIN",&epoch,0.0,1.0/srate,&lalStrainUnit,length);

    psd_l1 = XLALCreateREAL8FrequencySeries("L1 PSD",&epoch,0.0,1.0/duration,&lalSecondUnit,length/2+1);
    psd_h1 = XLALCreateREAL8FrequencySeries("H1 PSD",&epoch,0.0,1.0/duration,&lalSecondUnit,length/2+1);
    psd_v1 = XLALCreateREAL8FrequencySeries("V1 PSD",&epoch,0.0,1.0/duration,&lalSecondUnit,length/2+1);

    if(1){
		XLALSimNoisePSDaLIGOMidHighSensitivityP1200087(psd_l1,flow);
    	XLALSimNoisePSDaLIGOMidHighSensitivityP1200087(psd_h1,flow);
    	XLALSimNoisePSDAdVEarlyHighSensitivityP1200087(psd_v1,flow);
	}
	
	if(0) MySimNoisePSDLIGOO2Sensitivity(psd_l1,psd_h1,psd_v1,flow);
    
    if(real_psd_h1){
        int i;
        for(i=0;i<length/2+1;i++){
            psd_h1->data->data[i] *=4;
        }
    }
    if(save_th_psd){
        FILE *th_psd_file=fopen("th_psd.txt","w");
        int i;
        for(i=0;i<length/2+1;i++){
            double f = i/duration;
            double psd_data_l1 = psd_l1->data->data[i];
            double psd_data_h1 = psd_h1->data->data[i];
            double psd_data_v1 = psd_v1->data->data[i];

            fprintf(th_psd_file,"%e %e %e %e\n",f,psd_data_l1,psd_data_h1,psd_data_v1);
        }
        fclose(th_psd_file);
    }

    XLALSimNoise(seg[0],0,psd_l1,rng);
    XLALSimNoise(seg[1],0,psd_h1,rng);
    XLALSimNoise(seg[2],0,psd_v1,rng);

    //XLALSimNoise(seg[0],length/2,psd_l1,rng);
    //XLALSimNoise(seg[1],length/2,psd_h1,rng);
    //XLALSimNoise(seg[2],length/2,psd_v1,rng);

    XLALDestroyREAL8FrequencySeries(psd_l1);
    XLALDestroyREAL8FrequencySeries(psd_h1);
    XLALDestroyREAL8FrequencySeries(psd_v1);

    printf("end of geneate_LHV_data\n");
    printf("=================================================\n");

    return seg;
}

REAL8TimeSeries** generate_LHV_gw(double flow, double srate, double gps_time, int injection_id, gsl_rng *rng, double *injection_des)
{
    printf("=================================================\n");
    printf("generate_LHV_gw is starting ...\n");
    double m1=(1.3+gsl_rng_uniform(rng)*0.2)*LAL_MSUN_SI;
    double m2=(1.3+gsl_rng_uniform(rng)*0.2)*LAL_MSUN_SI;
    double S1x=0;
    double S1y=0;
    double S1z=0;
    double S2x=0;
    double S2y=0;
    double S2z=0;
    //double distance=source_distance_distribution(200,rng)*1e6*LAL_PC_SI;
    double distance=my_r_distribution(200,rng)*1e6*LAL_PC_SI;
    double i=sin_distribution(rng);
    double phiRef=2*M_PI*gsl_rng_uniform(rng);
    double longAscNodes=0.0;
    double eccentricity=0.0;
    double meanPerAno=0.0;
    double deltaT=1.0/srate;
    double f_min=flow;
    double f_ref=0.0;
    LALDict *LALpars=NULL;
    Approximant approximant=TaylorT4;
    REAL8TimeSeries *hplus=NULL;
    REAL8TimeSeries *hcros=NULL;
    
    XLALSimInspiralChooseTDWaveform(&hplus,&hcros,m1,m2,S1x,S1y,S1z,S2x,S2y,S2z,distance,i,phiRef,longAscNodes,eccentricity,meanPerAno,deltaT,f_min,f_ref,LALpars,approximant);

    int Second = (int)gps_time;
    int NanoSecond = (int)((gps_time-(int)gps_time)*1000000000);
    LIGOTimeGPS geocentric_arrival_time = {Second,NanoSecond};
    double right_ascension = 2*M_PI*gsl_rng_uniform(rng);
    double declination = sin_distribution(rng)-M_PI;
    double phi = 2*M_PI*gsl_rng_uniform(rng);;
    REAL8TimeSeries **strain = (REAL8TimeSeries**)malloc(sizeof(REAL8TimeSeries*)*3);
    XLALGPSAddGPS(&hplus->epoch,&geocentric_arrival_time);
    XLALGPSAddGPS(&hcros->epoch,&geocentric_arrival_time);
    LALDetector L1 = lalCachedDetectors[5];
    LALDetector H1 = lalCachedDetectors[4];
    LALDetector V1 = lalCachedDetectors[1];

    strain[0] = XLALSimDetectorStrainREAL8TimeSeries(hplus,hcros,right_ascension,declination,phi,&L1);
    strain[1] = XLALSimDetectorStrainREAL8TimeSeries(hplus,hcros,right_ascension,declination,phi,&H1);
    strain[2] = XLALSimDetectorStrainREAL8TimeSeries(hplus,hcros,right_ascension,declination,phi,&V1);

    XLALDestroyREAL8TimeSeries(hplus);
    XLALDestroyREAL8TimeSeries(hcros);

    double injection_des_copy[17]={m1,m2,S1x,S1y,S1z,S2x,S2y,S2z,distance,i,phiRef,longAscNodes,eccentricity,meanPerAno,phi,right_ascension,declination};
    
	int j;
    for(j=0;j<17;j++) injection_des[j]=injection_des_copy[j];

	printf("m1  = %lf\n",m1/LAL_MSUN_SI);
	printf("m2  = %lf\n",m2/LAL_MSUN_SI);
	printf("r   = %lf\n",distance/1e6/LAL_PC_SI);
	printf("ra  = %lf\n",right_ascension);
	printf("dec = %lf\n",declination);
    printf("end of generate_LHV_gw\n");
    printf("=================================================\n");
    return strain;
}

COMPLEX16FrequencySeries* generate_template(double flow, double fhigh, double deltaF, int injection_id, double *injection_des)
{
    double f_min=flow;
    double f_max=fhigh;
    double f_ref=0.0;
    LALDict *LALpars=NULL;
    Approximant approximant=SpinTaylorT4Fourier;
    COMPLEX16FrequencySeries *hplus=NULL;
    COMPLEX16FrequencySeries *hcros=NULL;

    XLALSimInspiralChooseFDWaveform(&hplus,&hcros,injection_des[0],injection_des[1],injection_des[2],injection_des[3],injection_des[4],injection_des[5],injection_des[6],injection_des[7],1e6*LAL_PC_SI,injection_des[9],injection_des[10],injection_des[11],injection_des[12],injection_des[13],deltaF,f_min,f_max,f_ref,LALpars,approximant);

    XLALDestroyCOMPLEX16FrequencySeries(hcros);

    return hplus;
}

void lal_injection(REAL8TimeSeries **data, REAL8TimeSeries **strain, int Ndet)
{
    int i,err;
    for(i=0;i<Ndet;i++){
        err = XLALSimAddInjectionREAL8TimeSeries(data[i],strain[i],NULL);
    }

}

void my_injection(REAL8TimeSeries **data, REAL8TimeSeries **strain, int Ndet)
{
    printf("=================================================\n");
    printf("my_injection is starting ...\n");
    int i_det,i_point,index;
    double t,ratio;
    for(i_det=0;i_det<Ndet;i_det++){
        double t_min_strain = strain[i_det]->epoch.gpsSeconds + strain[i_det]->epoch.gpsNanoSeconds*1.0e-9;
        double t_max_strain = t_min_strain + strain[i_det]->deltaT*(strain[i_det]->data->length-1);
        double t_min_data = data[i_det]->epoch.gpsSeconds + data[i_det]->epoch.gpsNanoSeconds*1.0e-9;

        //printf("%.18e %.18e %.18e \n",t_min_strain,t_max_strain,t_min_data);

        for(i_point=0;i_point<data[i_det]->data->length;i_point++){
            t = t_min_data + i_point*data[i_det]->deltaT;

            if(t>t_min_strain && t<t_max_strain){
                //printf("%.18e\n",t);
                index = (int)((t-t_min_strain)/strain[i_det]->deltaT);
                ratio = (t-t_min_strain)/strain[i_det]->deltaT-index;

                if(ratio==0) data[i_det]->data->data[i_point] += strain[i_det]->data->data[index];
                else data[i_det]->data->data[i_point] += strain[i_det]->data->data[index]*(1-ratio)+strain[i_det]->data->data[index+1]*ratio;
            }
        }
    }

    printf("end of my_injection\n");
    printf("=================================================\n");
}

static int test()
{
    double flow=15.0;
    double duration=512.0;
    double srate=4096.0;
    double gps_time=1234567890;
    double injection_des[17];
    int i,npoint=(int)(duration*srate);
    int injection_id=2;
	gsl_rng *rng = setup_gsl_rng(injection_id);

    REAL8TimeSeries **fake_data;
    REAL8TimeSeries **strain_data;
    fake_data  = generate_LHV_data(flow,duration,srate,gps_time,injection_id,rng);
    strain_data = generate_LHV_gw(flow,srate,gps_time+duration/2,injection_id,rng,injection_des);
    lal_injection(fake_data,strain_data,3);

    for(i=0;i<npoint;i++){
        double data_l1 = fake_data[0]->data->data[i];
        double data_h1 = fake_data[1]->data->data[i];
        double data_v1 = fake_data[2]->data->data[i];

        //printf("%e %e %e %e \n",i/srate,data_l1,data_h1,data_v1);
    }

    for(i=0;i<strain_data[0]->data->length;i++){
        printf("%e %e\n",i/srate,strain_data[0]->data->data[i]);
    }

	return 0;
}
