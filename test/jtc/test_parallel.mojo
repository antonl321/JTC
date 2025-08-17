from testing import assert_almost_equal
from jtc1 import Field
from sys import simdwidthof
from math import sqrt


def test_scalar[dtype:DType]():
    a = Field[dtype](14,9,12)
    var niter = 5
    var nblks = 5
    x = a.iterate(niter,nblks=nblks,test_it=True)
    #print("norm2: ", a.norm2(), x)
    if dtype == DType.float32:
        atol=1e-6
    else:
        atol=1.e-14
    #print("atol=",atol)
    assert_almost_equal(x, SIMD[dtype,1](0.0), atol=atol, rtol=0.0)


def main():
    test_scalar[DType.float32]()
    test_scalar[DType.float64]()