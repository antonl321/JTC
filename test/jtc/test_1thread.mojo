from testing import assert_almost_equal
from jtc import Field
from sys import simdwidthof

def test_scalar_float32():
    alias dtype = DType.float32
    atol = 1e-6
    a = Field[dtype](14,9,12)
    x = a.iterate(5,test_it=True)
    assert_almost_equal(x, SIMD[dtype,1](0.0), atol=atol, rtol=0.0)

def test_scalar_float64():
    alias dtype = DType.float64
    atol = 1e-14
    a = Field[dtype](14,9,12)
    x = a.iterate(5,test_it=True)
    assert_almost_equal(x, SIMD[dtype,1](0.0), atol=atol, rtol=0.0)


def test_vectorized_float32():
    alias dtype = DType.float32
    alias VW = simdwidthof[dtype]()
    atol = 1e-6
    a = Field[dtype](14,9,12)
    x = a.iterate_simd[VW](5,test_it=True)
    assert_almost_equal(x, SIMD[dtype,1](0.0), atol=atol, rtol=0.0)

def test_vectorized_and_parallelized_float32():
    alias dtype = DType.float32
    alias VW = simdwidthof[dtype]()
    atol = 1e-6
    a = Field[dtype](24,27,12)
    x = a.iterate_simd[VW](5,nblks=3,test_it=True)
    assert_almost_equal(x, SIMD[dtype,1](0.0), atol=atol, rtol=0.0)


def main():
    test_scalar_float32()
    test_scalar_float64()
    test_vectorized_float32()
    test_vectorized_and_parallelized_float32()
    #print("All tests passed.")