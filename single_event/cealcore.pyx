cdef extern from "sealcore.h":
    double testfunc1(double a, double b);

def pytest1( a, b ):
    return testfunc1( a, b )
