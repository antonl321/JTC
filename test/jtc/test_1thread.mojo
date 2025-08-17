from testing import assert_almost_equal
from jtc import Field
from sys import simdwidthof

def test_scalar[dtype:DType]():
    a = Field[dtype](14,9,12)
    x = a.iterate(5,test_it=True)
    #atol = 0.0
    #print("norm2: ", a.norm2(), x)
    if dtype == DType.float32:
        atol=1e-6
    else:
        atol=1.e-14
    #print("atol=",atol)
    assert_almost_equal(x, SIMD[dtype,1](0.0), atol=atol, rtol=0.0)

def test_vectorized[dtype:DType,VW:Int]():
    a = Field[dtype](14,9,12)
    x = a.iterate_simd[VW](5,test_it=True)
    #atol = 0.0
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
    #alias VW = simdwidthof[DType.float32]()
    test_vectorized[DType.float32,simdwidthof[DType.float32]()]()
    test_vectorized[DType.float64,simdwidthof[DType.float64]()]()
