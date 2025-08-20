import benchmark
from python import Python
from jtc import Field
from sys import simdwidthof

  
alias dtype = DType.float32  # default data type
alias VW = simdwidthof[dtype]()

fn run(gmin:Int, gmax:Int, gstep:Int, nswaps:Int=3, nthreads:List[Int]=[1]) raises -> List[List[Float64]]:
    #var np = Python.import_module("numpy")
    var pltdata =List[List[Float64]]()
    var pdatx = List[Float64]()
    
    var npoints = (gmax-gmin)//gstep +1
    print("npoints", npoints)
    
    for n in range(gmin,gmax+1,gstep):
        pdatx.append(n) # x axis points
    #print(pdatx)
    pltdata.append(pdatx)
    
    print("N   time  Swaps  LUPS")
    for ith in nthreads:
        print("nthreads", ith)
        var idx = 0
        var pdaty = List[Float64]()
        for n in range(gmin,gmax+1,gstep):
            print("min max n", gmin, gmax, n, idx)
            var phi=Field[dtype](n,n,n)
            @parameter
            fn test_phi_fn():
                _ = phi.iterate[VW](nswaps,nblks=ith)

            var rphi = benchmark.run[test_phi_fn](max_runtime_secs=2, min_runtime_secs=0.1)
            var tphi = rphi.mean()
            pdaty.append(Float64((n-2)**3)/(tphi/Float64(nswaps))/1e9) 
            print(ith, n, pdatx[idx], tphi, rphi.iters() * nswaps, idx, pdatx[idx], pdaty[idx])
            idx += 1
        pltdata.append(pdaty)
    return pltdata
    
    
def main():
    var threads = [1,2,4,8]  # number of threads to test
    pltdata = run(4, 200, 4, 10, threads)  # run with grid sizes from 4 to 64 with step 4 and 30 swaps
    #print("pltdata", len(pltdata[1]))
    
    var pyplt = Python.import_module("matplotlib.pyplot")
    var np = Python.import_module("numpy")
    var pylist = Python.list()
    for x in pltdata[0]:  # x-axis data
        pylist.append(x)
    var x_data = np.array(pylist)
    for i in range(1, len(pltdata)):
        var pydata = Python.list()
        for y in pltdata[i]:
            pydata.append(y)
        var y_data = np.array(pydata)
        _ = pyplt.plot(x_data, y_data, label="nthreads: " + String(threads[i-1]))

    _ = pyplt.xlabel("Grid size")
    _ = pyplt.ylabel("LUPS (billion)")
    _ = pyplt.title("Performance Benchmark")
    _ = pyplt.legend()
    _ = pyplt.grid(True)
    _ = pyplt.show()