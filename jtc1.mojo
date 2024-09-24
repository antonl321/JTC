from memory import DTypePointer
from memory import memset_zero
from buffer import Buffer
from math import sin, atan, cos, sqrt

#alias sixth =1.0/6.0


struct Field[DT: DType]:
    var data: DTypePointer[DT]
    var gx: Int
    var gy: Int
    var gz: Int
    var dsize: Int
    var eigenval: SIMD[DT,1]
    var bcurr: Buffer[DT]
    var bnext: Buffer[DT]

    
    fn __init__(inout self, gx: Int, gy: Int, gz: Int):
        self.dsize = gx * gy * gz
        self.data = DTypePointer[DT].alloc(2 * self.dsize)
        #memset_zero(self.data, self.dsize)
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.bcurr = Buffer[DT](self.data, self.dsize)
        self.bnext = Buffer[DT](self.data +self.dsize,self.dsize )
        self.eigenval = 0.0

    fn __copyinit__[VW:Int = 4](inout self, other: Self):
        self.gx = other.gx
        self.gy = other.gy
        self.gz = other.gz
        self.eigenval = other.eigenval
        self.data = DTypePointer[DT].alloc(2*other.dsize)
        self.dsize = other.dsize
        self.bcurr = Buffer[DT](self.data, self.dsize)
        self.bnext = Buffer[DT](self.data +self.dsize,self.dsize )
        #var sd = self.bcurr #Buffer[0,DT](self.data, self.dsize)
        #var od = other.bcurr #Buffer[0,DT](other.data, other.dsize)
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
                    result.data.store(p, self.bcurr.load[width=VW](p) + rhs.bcurr.load[width=VW](p))
                for i in range (vwgx+1, self.gx-1):
                    p = k * (self.gx) * (self.gy) + j * (self.gx) + i
                    result.data.store(p, self.bcurr.load[width=1](p) + rhs.bcurr.load[width=1](p))
        return result

    fn __del__(owned self):
        self.data.free()
        print("done del")
        
    fn zero(inout self):
        memset_zero[DT](self.data, self.dsize)

    @always_inline
    fn __getitem__(self, x: Int, y: Int, z: Int) -> SIMD[DT,1]:
        var p = z * (self.gx) * (self.gy) + y * (self.gx) + x 
        return self.data.load(p)

    #@always_inline
    #fn load[nelts:Int=VW](self, x: Int, y: Int, z: Int) -> SIMD[DT,nelts]:
    #    return self.data.simd_load[nelts](z * (self.gx+2) * (self.gy+2) + y * (self.gx+2) + x)

    @always_inline
    fn __setitem__(self, x: Int, y: Int, z: Int, val: SIMD[DT,1]):
        return self.data.store(z * (self.gx) * (self.gy) + y * (self.gx) + x, val)

    #@always_inline
    #fn store[nelts:Int=VW](self, x: Int, y: Int, z: Int, val: SIMD[DT, nelts]):
    #    self.data.simd_store(z * (self.gx+2) * (self.gy+2) + y * (self.gx+2) + x, val)

    # boundaries: all 0 for a start
    fn set_bc(inout self, iis: Int, ie:Int, js:Int, je:Int, ks:Int, ke:Int):
        for k in range(ks,ke):
            for j in range(js,je):
                for i in range(iis,ie):
                    var p = k * self.gy * self.gx + j * self.gx + i 
                    self.bcurr[p] = 0.0 
                    self.bnext[p] = 0.0

    fn initialize(inout self):
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

    fn norm2 (self) -> SIMD[DT,1] :
        var s : SIMD[DT, 1] = 0.0
        for k in range(1,self.gz-1):
            for j in range(1,self.gy-1):
                for i in range(1,self.gx-1):
                   # self[i,j,k] = sin[1, DType.f32](1.0*pi)
                   s = s + self.bcurr[self.bidx(i,j,k)] ** 2
        return s
    
    fn bidx(self, i:Int, j:Int, k:Int) -> Int:
        return k*self.gy*self.gx + j*self.gx +i

    fn iterate(inout self, niter: Int, test_it: Bool = False):
        alias sixth = 1.0/6.0 #SIMD[DT,1](1.0)/SIMD[DT,1](6.0)
        var norm_start = SIMD[DT,1](0.0)
        if test_it:
            norm_start = self.norm2()
        for iter in range(niter):
            for k in range(1,self.gz-1):
                for j in range(1,self.gy-1):
                    for i in range(1,self.gx-1):
                        self.bnext[self.bidx(i,j,k)]= sixth * ( 
                            self.bcurr[self.bidx(i-1,j  ,k  )] + self.bcurr[self.bidx(i+1,j  ,k  )] +
                            self.bcurr[self.bidx(i  ,j-1,k  )] + self.bcurr[self.bidx(i  ,j+1,k  )] +
                            self.bcurr[self.bidx(i  ,j  ,k-1)] + self.bcurr[self.bidx(i  ,j  ,k+1)])
            var aux = self.bcurr
            self.bcurr = self.bnext
            self.bnext = aux
        
        if test_it :
            print("iter test: ",  sqrt(self.norm2()/norm_start) - self.eigenval**niter,norm_start, self.norm2(), sixth, self.eigenval)
            print(self.bcurr[self.bidx(1,1,1)],self.bnext[self.bidx(1,1,1)])


    fn iterate_simd[VW:Int](inout self, niter: Int, test_it: Bool = False):
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
                        self.bnext.store[width=VW](self.bidx(i,j,k), sixth * (
                            self.bcurr.load[width=VW](self.bidx(i-1,j  ,k  )) + self.bcurr.load[width=VW](self.bidx(i+1,j  ,k  )) +
                            self.bcurr.load[width=VW](self.bidx(i  ,j-1,k  )) + self.bcurr.load[width=VW](self.bidx(i  ,j+1,k  )) +
                            self.bcurr.load[width=VW](self.bidx(i  ,j  ,k-1)) + self.bcurr.load[width=VW](self.bidx(i  ,j  ,k+1))))
                    for i in range (vwgx+1, self.gx-1):
                        self.bnext[self.bidx(i,j,k)]= sixth * ( 
                            self.bcurr[self.bidx(i-1,j  ,k  )] + self.bcurr[self.bidx(i+1,j  ,k  )] +
                            self.bcurr[self.bidx(i  ,j-1,k  )] + self.bcurr[self.bidx(i  ,j+1,k  )] +
                            self.bcurr[self.bidx(i  ,j  ,k-1)] + self.bcurr[self.bidx(i  ,j  ,k+1)])

            var aux = self.bcurr
            self.bcurr = self.bnext
            self.bnext = aux
        
        if test_it :
            print ("iter test: ",  sqrt(self.norm2()/norm_start) - self.eigenval**niter)





from tensor import Tensor, TensorSpec, TensorShape
from utils.index import Index
from memory import memset_zero
from math import sin, atan, cos, sqrt

#alias sixth =1.0/6.0

struct TField[DT: DType]:
    var gx: Int
    var gy: Int
    var gz: Int
    var fld: Tensor[DT]
    var gsize: Int
    var eigenval: SIMD[DT,1]
    var act: Int # current active component 

    #alias pi = 4.0*atan[DT,1](1.0)
    
    fn __init__(inout self, gx: Int, gy: Int, gz: Int):
        self.gsize = gx * gy * gz
        self.fld = Tensor[DT](2,gx,gy,gz)
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.eigenval = 0.0
        self.act = 0

    fn __copyinit__(inout self, other: Self):
        self.gx = other.gx
        self.gy = other.gy
        self.gz = other.gz
        self.eigenval = other.eigenval
        self.fld = other.fld
        self.gsize = other.gsize
        self.act = other.act
        
        
    fn __del__(owned self):
        print("del")
    #fn set_zero(inout self):
    #   memset_zero[DT,0](self.fld.data, self.fld.num_elements)


    # boundaries: all 0 for a start
    fn set_bc(inout self, iis: Int, ie:Int, js:Int, je:Int, ks:Int, ke:Int):
        for k in range(ks,ke):
            for j in range(js,je):
                for i in range(iis,ie): 
                    self.fld[Index(0,i,j,k)] = 0.0 
                    self.fld[Index(1,i,j,k)] = 0.0
    
    fn initialize(inout self):
        # inner points
        var pi = SIMD[DT,1](4)*atan(SIMD[DT,1](1))  # overkill ?
        for i in range(1,self.gx-1):
            for j in range(1,self.gy-1):
                for k in range(1,self.gz-1):
                    #self[i,j,k] = sin[1, DType.f32](1.0*(i+j+k))
                    self.fld[Index(self.act,i,j,k)] = sin(pi*i/(self.gx-1)) \
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

        var kx = 1.0
        var ky = 1.0
        var kz = 1.0
        self.eigenval = (cos(pi*kx/(self.gx-1)) + cos(pi*ky/(self.gy-1)) + cos(pi*kz/(self.gz-1)))/3.0

    fn norm2 (self) -> SIMD[DT,1] :
        var s : SIMD[DT, 1] = 0.0
        for k in range(1,self.gz-1):
            for j in range(1,self.gy-1):
                for i in range(1,self.gx-1):
                   # self[i,j,k] = sin[1, DType.f32](1.0*pi)
                   s = s + self.fld[self.act,i,j,k] ** 2
        return s
    

    fn iterate(inout self, niter: Int, test_it: Bool = False):
        alias sixth = 1.0/6.0
        var norm_start = SIMD[DT,1](0.0)
        if test_it:
            norm_start = self.norm2()
        for iter in range(niter):
            for i in range(1,self.gx-1):
                for j in range(1,self.gy-1):
                    for k in range(1,self.gz-1):
                        self.fld[Index(1-self.act,i,j,k)]= sixth * ( 
                            self.fld[self.act,i-1,j  ,k  ] + self.fld[self.act,i+1,j  ,k  ] +
                            self.fld[self.act,i  ,j-1,k  ] + self.fld[self.act,i  ,j+1,k  ] +
                            self.fld[self.act,i  ,j  ,k-1] + self.fld[self.act,i  ,j  ,k+1])
            self.act=1-self.act
        
        if test_it :
            print ("iter test: ",  niter, sqrt(self.norm2()/norm_start) - self.eigenval**niter)


    fn iterate_simd[VW:Int](inout self, niter: Int, test_it: Bool = False):
        alias sixth = 1.0/6.0
        var norm_start = SIMD[DT,1](0.0)
        if test_it:
            norm_start = self.norm2()

        var igz = self.gz-2
        var vwgz = igz - igz%VW 

        for iter in range(niter):
            var a = self.act
            for i in range(1,self.gx-1):
                for j in range(1,self.gy-1):
                    for k in range(1,1+vwgz,VW):
                        self.fld.store[width=VW](Index(1-a,i,j,k), sixth * (
                            self.fld.load[width=VW](a,i-1,j  ,k  ) + self.fld.load[width=VW](a,i+1,j  ,k  ) +
                            self.fld.load[width=VW](a,i  ,j-1,k  ) + self.fld.load[width=VW](a,i  ,j+1,k  ) +
                            self.fld.load[width=VW](a,i  ,j  ,k-1) + self.fld.load[width=VW](a,i  ,j  ,k+1)))
                    for k in range (vwgz+1, self.gz-1):
                        self.fld[Index(1-a,i,j,k)]= sixth * ( 
                            self.fld[a,i-1,j  ,k  ] + self.fld[a,i+1,j  ,k  ] +
                            self.fld[a,i  ,j-1,k  ] + self.fld[a,i  ,j+1,k  ] +
                            self.fld[a,i  ,j  ,k-1] + self.fld[a,i  ,j  ,k+1])

            self.act = 1 - self.act
            
        
        if test_it :
            print ("iter test: ",  niter, sqrt(self.norm2()/norm_start) - self.eigenval**niter)
    



import benchmark
from python import Python

alias VW = 4 # simd vector width (sys function not working)

fn run(gmin:Int, gmax:Int, gstep:Int, nswaps:Int=3) raises:
    var np = Python.import_module("numpy")
    var npoints = (gmax-gmin)//gstep +1
    var pdatx = np.zeros(npoints)
    var pdatyp = np.zeros(npoints)
    var pdatyt = np.zeros(npoints)
    print("npoints ", npoints)
    print("N   time  Swaps  LUPS")
    var idx = 0
    for n in range(gmin,gmax+1,gstep):
        print("min max n", gmin, gmax, n, idx)
        _ = pdatx.itemset(idx, n) # x axis
        var phi=Field[DType.float32](n,n,n)
        phi.initialize()
        var thi=TField[DType.float32](n,n,n)
        @parameter
        fn test_phi():
            phi.iterate_simd[4](nswaps)

        @parameter
        fn test_thi():
            thi.iterate_simd[4](nswaps)

        var rphi = benchmark.run[test_phi](max_runtime_secs=2, min_runtime_secs=0.1)
        #_ = (phi,)
        var tphi = rphi.mean()
        var rthi = benchmark.run[test_thi](max_runtime_secs=2, min_runtime_secs=0.1)
        #_ = (thi,)
        var tthi = rthi.mean()
        _ = pdatyp.itemset(idx, (n-2)**3/(tphi/nswaps)/1e9 )
        _ = pdatyt.itemset(idx, (n-2)**3/(tthi/nswaps)/1e9 )
        print(n, pdatx[idx], tphi, tthi, rphi.iters() * nswaps, idx, pdatx[idx], pdatyp[idx],pdatyt[idx])
        idx += 1
    #report.print()

    var pyplt = Python.import_module("matplotlib.pyplot")
    _ = pyplt.plot(pdatx,pdatyp)
    _ = pyplt.plot(pdatx,pdatyt)
    _ = pyplt.show()
    




