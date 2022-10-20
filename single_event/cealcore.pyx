cdef extern from "lal/LALDatatypes.h":
    struct COMPLEX8TimeSeries:
        pass

cdef extern from "sealcore.h":
    double testfunc1(double a, double b);

def pytest1( a, b ):
    return testfunc1( a, b )

cdef extern from "sealcore.h":
    void testLALseries(COMPLEX8TimeSeries **snr_series, int ndet);


def PytestLALseries(snr_series, ndet):
    return testLALseries(snr_series, ndet)