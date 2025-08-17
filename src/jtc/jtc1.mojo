from memory import UnsafePointer
from memory import memset_zero
from math import sin, atan, cos, sqrt
from algorithm import parallelize
#from layout import Layout
#from simd import SIMD, DType, always_inline

#alias sixth =1.0/6.0

struct Field[DT: DType]:
    var data: UnsafePointer[SIMD[DT,1]]
    var gx: Int
    var gy: Int
    var gz: Int
    var dsize: Int
    var eigenval: SIMD[DT,1]
    var bcurr: UnsafePointer[SIMD[DT,1]]
    var bnext: UnsafePointer[SIMD[DT,1]]


    
    fn __init__(out self, gx: Int, gy: Int, gz: Int, initialize: Bool = True):
        self.dsize = gx * gy * gz
        self.data = UnsafePointer[SIMD[DT,1]].alloc(2 * self.dsize)
        #memset_zero(self.data, self.dsize)
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.eigenval = 0.0
        self.bcurr = self.data
        self.bnext = self.data + self.dsize
        if initialize:
            self.initialize()

    fn __copyinit__[VW:Int = 4](out self, other: Self):
        self.gx = other.gx
        self.gy = other.gy
        self.gz = other.gz
        self.eigenval = other.eigenval
        self.data = UnsafePointer[SIMD[DT,1]].alloc(2*other.dsize)
        self.dsize = other.dsize
        self.bcurr = other.bcurr
        self.bnext = other.bnext
        for i in range(0,self.dsize - self.dsize % VW,VW):
            self.bcurr.store(i, other.bcurr.load[width=VW](i))
        for i in range(self.dsize - self.dsize % VW, self.dsize):
            self.bcurr[i] = other.bcurr[i]

    fn __add__[VW:Int](self, rhs: Field[DT]) raises -> Field[DT]:
        # test if the fields have the same sizes
        if self.gx != rhs.gx or self.gy != rhs.gy or self.gz != rhs.gz :
            raise ("cannot add fields with different grid sizes")
        var result = Field[DT](self.gx, self.gy, self.gz)
        var p : Int
        var igx = self.gx-2  # inner grid size in x dir
        var vwgx = igx - igx%VW # length made of an integer number of vector widths
        for k in range(1,self.gz-1):
            for j in range(1,self.gy-1):
                for i in range (1,vwgx+1,VW):
                    p = k * (self.gx) * (self.gy) + j * (self.gx) + i
                    result.bcurr.store[width=VW](p, self.bcurr.load[width=VW](p) + rhs.bcurr.load[width=VW](p))
                for i in range (vwgx+1, self.gx-1):
                    p = k * (self.gx) * (self.gy) + j * (self.gx) + i
                    result.bcurr.store(p, self.bcurr[p] + rhs.bcurr[p])
        return result

    fn __del__(owned self):
        self.data.free()
        #print("done del")
        
    fn zero(self):
        memset_zero(self.bcurr, self.dsize)

    #@always_inline
    #fn __getitem__(self, x: Int, y: Int, z: Int) -> SIMD[DT,1]:
    #    var p = z * (self.gx) * (self.gy) + y * (self.gx) + x 
    #    return self.data[p]

    #@always_inline
    #fn load[nelts:Int=VW](self, x: Int, y: Int, z: Int) -> SIMD[DT,nelts]:
    #    return self.data.load[nelts](z * (self.gx+2) * (self.gy+2) + y * (self.gx+2) + x)

    #@always_inline
    #fn __setitem__(self, x: Int, y: Int, z: Int, val: SIMD[DT,1]):
    #    self.data[z * (self.gx) * (self.gy) + y * (self.gx) + x] = val

    #@always_inline
    #fn store[nelts:Int=VW](self, x: Int, y: Int, z: Int, val: SIMD[DT, nelts]):
    #    self.data.simd_store(z * (self.gx+2) * (self.gy+2) + y * (self.gx+2) + x, val)

    # boundaries: all 0 for a start
    fn set_bc(self, iis: Int, ie:Int, js:Int, je:Int, ks:Int, ke:Int):
        for k in range(ks,ke):
            for j in range(js,je):
                for i in range(iis,ie):
                    var p = k * self.gy * self.gx + j * self.gx + i 
                    self.bcurr[p] = 0.0 
                    self.bnext[p] = 0.0

    fn initialize(mut self):
        # inner points
        var pi = SIMD[DT,1](4)*atan(SIMD[DT,1](1))
        for k in range(1,self.gz-1):
            for j in range(1,self.gy-1):
                for i in range(1,self.gx-1):
                    #self[i,j,k] = sin[1, DType.f32](1.0*(i+j+k))
                    self.bcurr[self.bidx(i,j,k)] = sin(pi*i/(self.gx-1)) \
                    * sin(pi*j/(self.gy-1)) \
                    * sin(pi*k/(self.gz-1))
        # bc north
        self.set_bc(0, 1, 1, self.gy-1, 1, self.gz-1)
        # bc south
        self.set_bc(self.gx-1, self.gx, 1, self.gy-1, 1, self.gz-1)
        # bc west
        self.set_bc(1, self.gx-1, 0, 1, 1, self.gz-1)
        # bc east
        self.set_bc(1, self.gx-1, self.gy-1, self.gy, 1, self.gz-1)
        # bc botttom
        self.set_bc(1, self.gx-1, 1, self.gy-1, 0, 1)
        # bc top
        self.set_bc(1, self.gx-1, 1, self.gy-1, self.gz-1, self.gz)

        alias kx = 1.0
        alias ky = 1.0
        alias kz = 1.0
        self.eigenval = (cos(pi*kx/(self.gx-1)) + cos(pi*ky/(self.gy-1)) + cos(pi*kz/(self.gz-1)))/3.0

    fn norm2 (self) -> SIMD[DT,1]:
        var s : SIMD[DT, 1] = 0.0
        for k in range(1,self.gz-1):
            for j in range(1,self.gy-1):
                for i in range(1,self.gx-1):
                   # self[i,j,k] = sin[1, DType.f32](1.0*pi)
                   s = s + self.bcurr[self.bidx(i,j,k)] ** 2
        return s
    
    fn bidx(self, i:Int, j:Int, k:Int) -> SIMD[DType.int64,1]:
        return k*self.gy*self.gx + j*self.gx +i

    fn iterate(mut self, niter: Int, nblks: Int = 1, test_it: Bool = False) -> SIMD[DT,1]:
        alias sixth = 1.0/6.0 #SIMD[DT,1](1.0)/SIMD[DT,1](6.0)
        
        var norm_start = SIMD[DT,1](0.0)
        if test_it:
            norm_start = self.norm2()
        # parallelize over blocks in y direction
        @parameter
        fn iterate_block(blkidx: Int):
            var blk_size: Int
            var blk_start: Int
            var blk_end: Int
            blk_size = (self.gy-2)//nblks
            blk_start = blkidx * blk_size + 1
            blk_end = blk_start + blk_size
            if blkidx == nblks - 1:
                blk_end = self.gy - 1
            
            for k in range(1,self.gz-1):
                for j in range(blk_start,blk_end):
                    for i in range(1,self.gx-1):
                        self.bnext[self.bidx(i,j,k)]= sixth * ( 
                            self.bcurr[self.bidx(i-1,j  ,k  )] + self.bcurr[self.bidx(i+1,j  ,k  )] +
                            self.bcurr[self.bidx(i  ,j-1,k  )] + self.bcurr[self.bidx(i  ,j+1,k  )] +
                            self.bcurr[self.bidx(i  ,j  ,k-1)] + self.bcurr[self.bidx(i  ,j  ,k+1)])

        for _ in range(niter):       
            parallelize[iterate_block](nblks, nblks)
            #iterate_block(0)  # run single thread for now    
            var aux = self.bcurr
            self.bcurr = self.bnext
            self.bnext = aux
        
        if test_it :
            return sqrt(self.norm2()/norm_start) - self.eigenval**niter
        else:
            return SIMD[DT,1](-1.0)  # return -1 if not testing

            # this shoulb be used with a debug flag
            #print("iter test: ",  sqrt(self.norm2()/norm_start) - self.eigenval**niter,norm_start, self.norm2(), sixth, self.eigenval)
            #print(self.bcurr[self.bidx(1,1,1)],self.bnext[self.bidx(1,1,1)])


    fn iterate_simd[VW:Int](mut self, niter: Int, test_it: Bool = False) -> SIMD[DT,1]:
        alias sixth = 1.0/6.0
        var norm_start = SIMD[DT,1](0.0)
        if test_it:
            norm_start = self.norm2()

        var igx = self.gx-2
        var vwgx = igx - igx%VW 
        for iter in range(niter):
            for k in range(1,self.gz-1):
                for j in range(1,self.gy-1):
                    for i in range(1,1+vwgx,VW):
                        var val_to_store = sixth * ( 
                            self.bcurr.load[width=VW](self.bidx(i-1,j  ,k  )) + self.bcurr.load[width=VW](self.bidx(i+1,j  ,k  )) +
                            self.bcurr.load[width=VW](self.bidx(i  ,j-1,k  )) + self.bcurr.load[width=VW](self.bidx(i  ,j+1,k  )) +
                            self.bcurr.load[width=VW](self.bidx(i  ,j  ,k-1)) + self.bcurr.load[width=VW](self.bidx(i  ,j  ,k+1)))
                        self.bnext.store(self.bidx(i,j,k), val_to_store)
                    for i in range (vwgx+1, self.gx-1):
                        self.bnext[self.bidx(i,j,k)]= sixth * ( 
                            self.bcurr[self.bidx(i-1,j  ,k  )] + self.bcurr[self.bidx(i+1,j  ,k  )] +
                            self.bcurr[self.bidx(i  ,j-1,k  )] + self.bcurr[self.bidx(i  ,j+1,k  )] +
                            self.bcurr[self.bidx(i  ,j  ,k-1)] + self.bcurr[self.bidx(i  ,j  ,k+1)])

            var aux = self.bcurr
            self.bcurr = self.bnext
            self.bnext = aux
        
        if test_it :
            return sqrt(self.norm2()/norm_start) - self.eigenval**niter
        else:
            return SIMD[DT,1](-1.0)  # return -1 if
            #print ("iter test: ",  sqrt(self.norm2()/norm_start) - self.eigenval**niter)




#def main():
#    var f = Field[DType.float64](16,16,16)
#    f.initialize()
#    f.iterate_simd[4](10, True)
#    print("norm2: ", f.norm2())
#    print("eigenval: ", f.eigenval)
    #f.iterate(1000, True)
    #print(f.norm2())
    #print(f.eigenval)





