////////////////////////////////////////////////////////////////////////////////
// File: exponential_integral_Ei.c                                            //
// Routine(s):                                                                //
//    Exponential_Integral_Ei                                                 //
//    xExponential_Integral_Ei                                                //
////////////////////////////////////////////////////////////////////////////////

#include <math.h>           // required for fabsl(), expl() and logl()
#include <float.h>          // required for LDBL_EPSILON, DBL_MAX

//                         Internally Defined Routines                        //
double      Exponential_Integral_Ei( double x );
long double xExponential_Integral_Ei( long double x );
double      Exponential_Integral_Ei_approx( double x );


static long double Continued_Fraction_Ei( long double x );
static long double Power_Series_Ei( long double x );
static long double Argument_Addition_Series_Ei( long double x);


//                         Internally Defined Constants                       //
static const long double epsilon = 10.0 * LDBL_EPSILON;
