/*
*  cache some self-defined lal detectors in a stupid way.
*/



#include <lal/LALDatatypes.h>
#include <lal/LALDetectors.h>

/**
 * \name myET1 15km Interferometric Detector constants
 * The following constants describe the location and geometry of the
 * myET1 15km Interferometric Detector.
 */
/** @{ */
#define LAL_MYET1_DETECTOR_NAME               	"MYET1_Sardinia"	/**< myET1 detector name string */
#define LAL_MYET1_DETECTOR_PREFIX             	"M1"	/**< myET1 detector prefix string */
#define LAL_MYET1_DETECTOR_LONGITUDE_RAD      	0.16435765566	/**< myET1 vertex longitude (rad) */
#define LAL_MYET1_DETECTOR_LATITUDE_RAD       	0.70720741291	/**< myET1 vertex latitude (rad) */
#define LAL_MYET1_DETECTOR_ELEVATION_SI       	300.	/**< myET1 vertex elevation (m) */
#define LAL_MYET1_DETECTOR_ARM_X_AZIMUTH_RAD  	6.06942919522	/**< myET1 x arm azimuth (rad) */
#define LAL_MYET1_DETECTOR_ARM_Y_AZIMUTH_RAD  	4.49863286842	/**< myET1 y arm azimuth (rad) */
#define LAL_MYET1_DETECTOR_ARM_X_ALTITUDE_RAD 	0.00000000000	/**< myET1 x arm altitude (rad) */
#define LAL_MYET1_DETECTOR_ARM_Y_ALTITUDE_RAD 	0.00000000000	/**< myET1 y arm altitude (rad) */
#define LAL_MYET1_DETECTOR_ARM_X_MIDPOINT_SI  	7500.00000000000	/**< myET1 x arm midpoint (m) */
#define LAL_MYET1_DETECTOR_ARM_Y_MIDPOINT_SI  	7500.00000000000	/**< myET1 y arm midpoint (m) */
#define LAL_MYET1_VERTEX_LOCATION_X_SI        	4.79018449477e+06	/**< myET1 x-component of vertex location in Earth-centered frame (m) */
#define LAL_MYET1_VERTEX_LOCATION_Y_SI        	7.94470200972e+05	/**< myET1 y-component of vertex location in Earth-centered frame (m) */
#define LAL_MYET1_VERTEX_LOCATION_Z_SI        	4.12224332847e+06	/**< myET1 z-component of vertex location in Earth-centered frame (m) */
#define LAL_MYET1_ARM_X_DIRECTION_X           	-0.59166138	/**< myET1 x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYET1_ARM_X_DIRECTION_Y           	-0.31315911	/**< myET1 y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYET1_ARM_X_DIRECTION_Z           	0.74287831	/**< myET1 z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYET1_ARM_Y_DIRECTION_X           	0.29586253	/**< myET1 x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_MYET1_ARM_Y_DIRECTION_Y           	-0.94152064	/**< myET1 y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_MYET1_ARM_Y_DIRECTION_Z           	-0.16125835	/**< myET1 z-component of unit vector pointing along y arm in Earth-centered frame */
/** @} */

/**
 * \name myET2 15km Interferometric Detector constants
 * The following constants describe the location and geometry of the
 * myET2 15km Interferometric Detector.
 */
/** @{ */
#define LAL_MYET2_DETECTOR_NAME               	"MYET2_Netherlands"	/**< myET2 detector name string */
#define LAL_MYET2_DETECTOR_PREFIX             	"M2"	/**< myET2 detector prefix string */
#define LAL_MYET2_DETECTOR_LONGITUDE_RAD      	0.10334094501	/**< myET2 vertex longitude (rad) */
#define LAL_MYET2_DETECTOR_LATITUDE_RAD       	0.88523099661	/**< myET2 vertex latitude (rad) */
#define LAL_MYET2_DETECTOR_ELEVATION_SI       	0.	/**< myET2 vertex elevation (m) */
#define LAL_MYET2_DETECTOR_ARM_X_AZIMUTH_RAD  	2.88404694522	/**< myET2 x arm azimuth (rad) */
#define LAL_MYET2_DETECTOR_ARM_Y_AZIMUTH_RAD  	1.31325061843	/**< myET2 y arm azimuth (rad) */
#define LAL_MYET2_DETECTOR_ARM_X_ALTITUDE_RAD 	0.00000000000	/**< myET2 x arm altitude (rad) */
#define LAL_MYET2_DETECTOR_ARM_Y_ALTITUDE_RAD 	0.00000000000	/**< myET2 y arm altitude (rad) */
#define LAL_MYET2_DETECTOR_ARM_X_MIDPOINT_SI  	7500.00000000000	/**< myET2 x arm midpoint (m) */
#define LAL_MYET2_DETECTOR_ARM_Y_MIDPOINT_SI  	7500.00000000000	/**< myET2 y arm midpoint (m) */
#define LAL_MYET2_VERTEX_LOCATION_X_SI        	4.02460368661e+06	/**< myET2 x-component of vertex location in Earth-centered frame (m) */
#define LAL_MYET2_VERTEX_LOCATION_Y_SI        	4.17393236806e+05	/**< myET2 y-component of vertex location in Earth-centered frame (m) */
#define LAL_MYET2_VERTEX_LOCATION_Z_SI        	4.91388299551e+06	/**< myET2 z-component of vertex location in Earth-centered frame (m) */
#define LAL_MYET2_ARM_X_DIRECTION_X           	0.71826288	/**< myET2 x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYET2_ARM_X_DIRECTION_Y           	0.33056544	/**< myET2 y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYET2_ARM_X_DIRECTION_Z           	-0.61222947	/**< myET2 z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYET2_ARM_Y_DIRECTION_X           	-0.29586253	/**< myET2 x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_MYET2_ARM_Y_DIRECTION_Y           	0.94152064	/**< myET2 y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_MYET2_ARM_Y_DIRECTION_Z           	0.16125835	/**< myET2 z-component of unit vector pointing along y arm in Earth-centered frame */
/** @} */

/**
 * \name myET3 15km Interferometric Detector constants
 * The following constants describe the location and geometry of the
 * myET3 15km Interferometric Detector.
 */
/** @{ */
#define LAL_MYET3_DETECTOR_NAME               	"MYET3_Netherlands"	/**< myET3 detector name string */
#define LAL_MYET3_DETECTOR_PREFIX             	"M3"	/**< myET3 detector prefix string */
#define LAL_MYET3_DETECTOR_LONGITUDE_RAD      	0.10334094501	/**< myET3 vertex longitude (rad) */
#define LAL_MYET3_DETECTOR_LATITUDE_RAD       	0.88523099661	/**< myET3 vertex latitude (rad) */
#define LAL_MYET3_DETECTOR_ELEVATION_SI       	0.	/**< myET3 vertex elevation (m) */
#define LAL_MYET3_DETECTOR_ARM_X_AZIMUTH_RAD  	2.09864878183	/**< myET3 x arm azimuth (rad) */
#define LAL_MYET3_DETECTOR_ARM_Y_AZIMUTH_RAD  	0.52785245503	/**< myET3 y arm azimuth (rad) */
#define LAL_MYET3_DETECTOR_ARM_X_ALTITUDE_RAD 	0.00000000000	/**< myET3 x arm altitude (rad) */
#define LAL_MYET3_DETECTOR_ARM_Y_ALTITUDE_RAD 	0.00000000000	/**< myET3 y arm altitude (rad) */
#define LAL_MYET3_DETECTOR_ARM_X_MIDPOINT_SI  	7500.00000000000	/**< myET3 x arm midpoint (m) */
#define LAL_MYET3_DETECTOR_ARM_Y_MIDPOINT_SI  	7500.00000000000	/**< myET3 y arm midpoint (m) */
#define LAL_MYET3_VERTEX_LOCATION_X_SI        	4.02460368661e+06	/**< myET3 x-component of vertex location in Earth-centered frame (m) */
#define LAL_MYET3_VERTEX_LOCATION_Y_SI        	4.17393236806e+05	/**< myET3 y-component of vertex location in Earth-centered frame (m) */
#define LAL_MYET3_VERTEX_LOCATION_Z_SI        	4.91388299551e+06	/**< myET3 z-component of vertex location in Earth-centered frame (m) */
#define LAL_MYET3_ARM_X_DIRECTION_X           	0.29868216	/**< myET3 x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYET3_ARM_X_DIRECTION_Y           	0.89950069	/**< myET3 y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYET3_ARM_X_DIRECTION_Z           	-0.31888474	/**< myET3 z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYET3_ARM_Y_DIRECTION_X           	-0.71709495	/**< myET3 x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_MYET3_ARM_Y_DIRECTION_Y           	0.43201056	/**< myET3 y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_MYET3_ARM_Y_DIRECTION_Z           	0.54693848	/**< myET3 z-component of unit vector pointing along y arm in Earth-centered frame */
/** @} */

/**
 * \name myCE1 40km Interferometric Detector constants
 * The following constants describe the location and geometry of the
 * myCE1 15km Interferometric Detector.
 */
/** @{ */
#define LAL_MYCE1_DETECTOR_NAME               	"MYCE1_NewMexico"	/**< myCE1 detector name string */
#define LAL_MYCE1_DETECTOR_PREFIX             	"Z1"	/**< myCE1 detector prefix string */
#define LAL_MYCE1_DETECTOR_LONGITUDE_RAD      	-1.8587756534	/**< myCE1 vertex longitude (rad) */
#define LAL_MYCE1_DETECTOR_LATITUDE_RAD       	0.57875117996	/**< myCE1 vertex latitude (rad) */
#define LAL_MYCE1_DETECTOR_ELEVATION_SI       	1200.	/**< myCE1 vertex elevation (m) */
#define LAL_MYCE1_DETECTOR_ARM_X_AZIMUTH_RAD  	5.88375751328	/**< myCE1 x arm azimuth (rad) */
#define LAL_MYCE1_DETECTOR_ARM_Y_AZIMUTH_RAD  	4.31296118649	/**< myCE1 y arm azimuth (rad) */
#define LAL_MYCE1_DETECTOR_ARM_X_ALTITUDE_RAD 	0.00000000000	/**< myCE1 x arm altitude (rad) */
#define LAL_MYCE1_DETECTOR_ARM_Y_ALTITUDE_RAD 	0.00000000000	/**< myCE1 y arm altitude (rad) */
#define LAL_MYCE1_DETECTOR_ARM_X_MIDPOINT_SI  	20000.00000000000	/**< myCE1 x arm midpoint (m) */
#define LAL_MYCE1_DETECTOR_ARM_Y_MIDPOINT_SI  	20000.00000000000	/**< myCE1 y arm midpoint (m) */
#define LAL_MYCE1_VERTEX_LOCATION_X_SI        	-1.5182875279e+06	/**< myCE1 x-component of vertex location in Earth-centered frame (m) */
#define LAL_MYCE1_VERTEX_LOCATION_Y_SI        	-5.1256527935e+06	/**< myCE1 y-component of vertex location in Earth-centered frame (m) */
#define LAL_MYCE1_VERTEX_LOCATION_Z_SI        	3.4694836617e+06	/**< myCE1 z-component of vertex location in Earth-centered frame (m) */
#define LAL_MYCE1_ARM_X_DIRECTION_X           	-0.22975481	/**< myCE1 x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYCE1_ARM_X_DIRECTION_Y           	0.59362215	/**< myCE1 y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYCE1_ARM_X_DIRECTION_Z           	0.77124929	/**< myCE1 z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYCE1_ARM_Y_DIRECTION_X           	-0.94375938	/**< myCE1 x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_MYCE1_ARM_Y_DIRECTION_Y           	0.05770306	/**< myCE1 y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_MYCE1_ARM_Y_DIRECTION_Z           	-0.3255589	/**< myCE1 z-component of unit vector pointing along y arm in Earth-centered frame */
/** @} */

/**
 * \name myCE2 20km Interferometric Detector constants
 * The following constants describe the location and geometry of the
 * myCE2 20km Interferometric Detector.
 */
/** @{ */
#define LAL_MYCE2_DETECTOR_NAME               	"MYCE2_Idaho"	/**< myCE2 detector name string */
#define LAL_MYCE2_DETECTOR_PREFIX             	"Z2"	/**< myCE2 detector prefix string */
#define LAL_MYCE2_DETECTOR_LONGITUDE_RAD      	-1.96873139625	/**< myCE2 vertex longitude (rad) */
#define LAL_MYCE2_DETECTOR_LATITUDE_RAD       	0.764977811149	/**< myCE2 vertex latitude (rad) */
#define LAL_MYCE2_DETECTOR_ELEVATION_SI       	1460.	/**< myCE2 vertex elevation (m) */
#define LAL_MYCE2_DETECTOR_ARM_X_AZIMUTH_RAD  	2.67338332646	/**< myCE2 x arm azimuth (rad) */
#define LAL_MYCE2_DETECTOR_ARM_Y_AZIMUTH_RAD  	1.10258699966	/**< myCE2 y arm azimuth (rad) */
#define LAL_MYCE2_DETECTOR_ARM_X_ALTITUDE_RAD 	0.00000000000	/**< myCE2 x arm altitude (rad) */
#define LAL_MYCE2_DETECTOR_ARM_Y_ALTITUDE_RAD 	0.00000000000	/**< myCE2 y arm altitude (rad) */
#define LAL_MYCE2_DETECTOR_ARM_X_MIDPOINT_SI  	10000.00000000000	/**< myCE2 x arm midpoint (m) */
#define LAL_MYCE2_DETECTOR_ARM_Y_MIDPOINT_SI  	10000.00000000000	/**< myCE2 y arm midpoint (m) */
#define LAL_MYCE2_VERTEX_LOCATION_X_SI        	-1.7863037371e+06	/**< myCE2 x-component of vertex location in Earth-centered frame (m) */
#define LAL_MYCE2_VERTEX_LOCATION_Y_SI        	-4.2494486683e+06	/**< myCE2 y-component of vertex location in Earth-centered frame (m) */
#define LAL_MYCE2_VERTEX_LOCATION_Z_SI        	4.39549578216e+06	/**< myCE2 z-component of vertex location in Earth-centered frame (m) */
#define LAL_MYCE2_ARM_X_DIRECTION_X           	0.17654584	/**< myCE2 x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYCE2_ARM_X_DIRECTION_Y           	-0.7445841	/**< myCE2 y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYCE2_ARM_X_DIRECTION_Z           	-0.64375934	/**< myCE2 z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYCE2_ARM_Y_DIRECTION_X           	0.94375938	/**< myCE2 x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_MYCE2_ARM_Y_DIRECTION_Y           	-0.05770306	/**< myCE2 y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_MYCE2_ARM_Y_DIRECTION_Z           	0.3255589	/**< myCE2 z-component of unit vector pointing along y arm in Earth-centered frame */
/** @} */

/**
 * \name myCE3 20km Interferometric Detector constants
 * The following constants describe the location and geometry of the
 * myCE3 20km Interferometric Detector.
 */
/** @{ */
#define LAL_MYCE3_DETECTOR_NAME               	"MYCE3_Idaho"	/**< myCE3 detector name string */
#define LAL_MYCE3_DETECTOR_PREFIX             	"Z3"	/**< myCE3 detector prefix string */
#define LAL_MYCE3_DETECTOR_LONGITUDE_RAD      	-1.96873139625	/**< myCE3 vertex longitude (rad) */
#define LAL_MYCE3_DETECTOR_LATITUDE_RAD       	0.764977811149	/**< myCE3 vertex latitude (rad) */
#define LAL_MYCE3_DETECTOR_ELEVATION_SI       	1460.	/**< myCE3 vertex elevation (m) */
#define LAL_MYCE3_DETECTOR_ARM_X_AZIMUTH_RAD  	1.88798516306	/**< myCE3 x arm azimuth (rad) */
#define LAL_MYCE3_DETECTOR_ARM_Y_AZIMUTH_RAD  	0.31718883627	/**< myCE3 y arm azimuth (rad) */
#define LAL_MYCE3_DETECTOR_ARM_X_ALTITUDE_RAD 	0.00000000000	/**< myCE3 x arm altitude (rad) */
#define LAL_MYCE3_DETECTOR_ARM_Y_ALTITUDE_RAD 	0.00000000000	/**< myCE3 y arm altitude (rad) */
#define LAL_MYCE3_DETECTOR_ARM_X_MIDPOINT_SI  	10000.00000000000	/**< myCE3 x arm midpoint (m) */
#define LAL_MYCE3_DETECTOR_ARM_Y_MIDPOINT_SI  	10000.00000000000	/**< myCE3 y arm midpoint (m) */
#define LAL_MYCE3_VERTEX_LOCATION_X_SI        	-1.7863037371e+06	/**< myCE3 x-component of vertex location in Earth-centered frame (m) */
#define LAL_MYCE3_VERTEX_LOCATION_Y_SI        	-4.2494486683e+06	/**< myCE3 y-component of vertex location in Earth-centered frame (m) */
#define LAL_MYCE3_VERTEX_LOCATION_Z_SI        	4.39549578216e+06	/**< myCE3 z-component of vertex location in Earth-centered frame (m) */
#define LAL_MYCE3_ARM_X_DIRECTION_X           	0.79217542	/**< myCE3 x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYCE3_ARM_X_DIRECTION_Y           	-0.56730269	/**< myCE3 y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYCE3_ARM_X_DIRECTION_Z           	-0.22500169	/**< myCE3 z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_MYCE3_ARM_Y_DIRECTION_X           	0.54250189	/**< myCE3 x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_MYCE3_ARM_Y_DIRECTION_Y           	0.48569824	/**< myCE3 y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_MYCE3_ARM_Y_DIRECTION_Z           	0.68541149	/**< myCE3 z-component of unit vector pointing along y arm in Earth-centered frame */
/** @} */

LALFrDetector generate_fr(CHAR name[LALNameLength], CHAR prefix[3],
	REAL8 vertexLongitudeRadians, REAL8 vertexLatitudeRadians, REAL4 vertexElevation,
	REAL4 xArmAltitudeRadians, REAL4 xArmAzimuthRadians,
	REAL4 yArmAltitudeRadians, REAL4 yArmAzimuthRadians,
	REAL4 xArmMidpoint, REAL4 yArmMidpoint)
{
	LALFrDetector fr;
	fr.name[0] = name[0];
	fr.prefix[0] = prefix[0];

	fr.vertexLongitudeRadians = vertexLongitudeRadians;
	fr.vertexLatitudeRadians = vertexLatitudeRadians;
	fr.vertexElevation = vertexElevation;

	fr.xArmAltitudeRadians = xArmAltitudeRadians;
	fr.xArmAzimuthRadians = xArmAzimuthRadians;
	fr.yArmAltitudeRadians = yArmAltitudeRadians;
	fr.yArmAzimuthRadians = yArmAzimuthRadians;

	fr.xArmMidpoint = xArmMidpoint;
	fr.yArmMidpoint = yArmMidpoint;
	return fr;
}

LALDetector get_cached_sealdetector(int detcode){
	LALDetectorType type=LALDETECTORTYPE_IFODIFF;
	LALDetector detector;
	LALFrDetector fr;
	switch (detcode)
	{
	case 100:
		fr = generate_fr(LAL_MYET1_DETECTOR_NAME, LAL_MYET1_DETECTOR_PREFIX,
			LAL_MYET1_DETECTOR_LONGITUDE_RAD, LAL_MYET1_DETECTOR_LATITUDE_RAD, LAL_MYET1_DETECTOR_ELEVATION_SI,
			LAL_MYET1_DETECTOR_ARM_X_ALTITUDE_RAD, LAL_MYET1_DETECTOR_ARM_X_AZIMUTH_RAD,
			LAL_MYET1_DETECTOR_ARM_Y_ALTITUDE_RAD, LAL_MYET1_DETECTOR_ARM_Y_AZIMUTH_RAD,
			LAL_MYET1_DETECTOR_ARM_X_MIDPOINT_SI, LAL_MYET1_DETECTOR_ARM_Y_MIDPOINT_SI);
		break;

	case 101:
		fr = generate_fr(LAL_MYET2_DETECTOR_NAME, LAL_MYET2_DETECTOR_PREFIX,
			LAL_MYET2_DETECTOR_LONGITUDE_RAD, LAL_MYET2_DETECTOR_LATITUDE_RAD, LAL_MYET2_DETECTOR_ELEVATION_SI,
			LAL_MYET2_DETECTOR_ARM_X_ALTITUDE_RAD, LAL_MYET2_DETECTOR_ARM_X_AZIMUTH_RAD,
			LAL_MYET2_DETECTOR_ARM_Y_ALTITUDE_RAD, LAL_MYET2_DETECTOR_ARM_Y_AZIMUTH_RAD,
			LAL_MYET2_DETECTOR_ARM_X_MIDPOINT_SI, LAL_MYET2_DETECTOR_ARM_Y_MIDPOINT_SI);
		break;

	case 102:
		fr = generate_fr(LAL_MYET3_DETECTOR_NAME, LAL_MYET3_DETECTOR_PREFIX,
			LAL_MYET3_DETECTOR_LONGITUDE_RAD, LAL_MYET3_DETECTOR_LATITUDE_RAD, LAL_MYET3_DETECTOR_ELEVATION_SI,
			LAL_MYET3_DETECTOR_ARM_X_ALTITUDE_RAD, LAL_MYET3_DETECTOR_ARM_X_AZIMUTH_RAD,
			LAL_MYET3_DETECTOR_ARM_Y_ALTITUDE_RAD, LAL_MYET3_DETECTOR_ARM_Y_AZIMUTH_RAD,
			LAL_MYET3_DETECTOR_ARM_X_MIDPOINT_SI, LAL_MYET3_DETECTOR_ARM_Y_MIDPOINT_SI);
		break;

	case 103:
		fr = generate_fr(LAL_MYCE1_DETECTOR_NAME, LAL_MYCE1_DETECTOR_PREFIX,
			LAL_MYCE1_DETECTOR_LONGITUDE_RAD, LAL_MYCE1_DETECTOR_LATITUDE_RAD, LAL_MYCE1_DETECTOR_ELEVATION_SI,
			LAL_MYCE1_DETECTOR_ARM_X_ALTITUDE_RAD, LAL_MYCE1_DETECTOR_ARM_X_AZIMUTH_RAD,
			LAL_MYCE1_DETECTOR_ARM_Y_ALTITUDE_RAD, LAL_MYCE1_DETECTOR_ARM_Y_AZIMUTH_RAD,
			LAL_MYCE1_DETECTOR_ARM_X_MIDPOINT_SI, LAL_MYCE1_DETECTOR_ARM_Y_MIDPOINT_SI);
		break;

	case 104:
		fr = generate_fr(LAL_MYCE2_DETECTOR_NAME, LAL_MYCE2_DETECTOR_PREFIX,
			LAL_MYCE2_DETECTOR_LONGITUDE_RAD, LAL_MYCE2_DETECTOR_LATITUDE_RAD, LAL_MYCE2_DETECTOR_ELEVATION_SI,
			LAL_MYCE2_DETECTOR_ARM_X_ALTITUDE_RAD, LAL_MYCE2_DETECTOR_ARM_X_AZIMUTH_RAD,
			LAL_MYCE2_DETECTOR_ARM_Y_ALTITUDE_RAD, LAL_MYCE2_DETECTOR_ARM_Y_AZIMUTH_RAD,
			LAL_MYCE2_DETECTOR_ARM_X_MIDPOINT_SI, LAL_MYCE2_DETECTOR_ARM_Y_MIDPOINT_SI);
		break;

	case 105:
		fr = generate_fr(LAL_MYCE3_DETECTOR_NAME, LAL_MYCE3_DETECTOR_PREFIX,
			LAL_MYCE3_DETECTOR_LONGITUDE_RAD, LAL_MYCE3_DETECTOR_LATITUDE_RAD, LAL_MYCE3_DETECTOR_ELEVATION_SI,
			LAL_MYCE3_DETECTOR_ARM_X_ALTITUDE_RAD, LAL_MYCE3_DETECTOR_ARM_X_AZIMUTH_RAD,
			LAL_MYCE3_DETECTOR_ARM_Y_ALTITUDE_RAD, LAL_MYCE3_DETECTOR_ARM_Y_AZIMUTH_RAD,
			LAL_MYCE3_DETECTOR_ARM_X_MIDPOINT_SI, LAL_MYCE3_DETECTOR_ARM_Y_MIDPOINT_SI);
		break;

	default:
		printf("Wrong detector code!");
		exit(-1);
	} // end of switch

	return *XLALCreateDetector(&detector, &fr, type );
}
