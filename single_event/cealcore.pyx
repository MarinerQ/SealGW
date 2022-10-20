
cdef extern from "lal/LALDatatypes.h":
    struct COMPLEX8TimeSeries:
        pass


cdef extern from "Python.h":
    struct PyObject:
        pass

cdef extern from "sealcore.h":
    double testfunc1(double a, double b);

def pytest1( a, b ):
    return testfunc1( a, b )

cdef extern from "sealcore.h":
    #void testLALseries(double **snr_series, int ndet);
    void testdoubleseries(double *data_array, int ndet, int ntime);
    #void testLALseries(PyObject snr_series, int ndet);
    


def PytestLALseries(double[:] data_array, ndet, ntime):
    #return testLALseries(&snr_series[0], ndet)
    return testdoubleseries(&data_array[0], ndet, ntime)