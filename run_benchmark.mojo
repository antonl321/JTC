import benchmark
from python import Python
from jtc import Field
import sys

  
alias dtype = DType.float32  # default data type
alias VW = sys.simd_width_of[dtype]()

fn run(gmin:Int, gmax:Int, gstep:Int, nswaps:Int=3, niters:Int=10, nthreads:List[Int]=[1]) raises -> List[List[Float64]]:
    #var np = Python.import_module("numpy")
    var pltdata =List[List[Float64]]()
    var pdatx = List[Float64]()
    
    var npoints = (gmax-gmin)//gstep +1
    print("npoints", npoints)
    
    for n in range(gmin,gmax+1,gstep):
        pdatx.append(n) # x axis points
    #print(pdatx)
    pltdata.append(pdatx)
    
    with open("benchmark_mojo_output.txt", "w") as fout:
        #benchmark.set_output(fout)
        fout.write("# threads gx gy gz iters min mean max \n")
        for ith in nthreads:
            #print("nthreads", ith)
            var idx = 0
            var pdaty = List[Float64]()
            for n in range(gmin,gmax+1,gstep):
                #print("min max n", gmin, gmax, n, idx)
                var phi=Field[dtype](n,n,n)
                @parameter
                fn test_phi_fn():
                    _ = phi.iterate[VW](nswaps,nblks=ith)
                # run params set to match the C version
                var rphi = benchmark.run[test_phi_fn](max_iters=niters, min_runtime_secs=0.2) #(max_runtime_secs=100000000, min_runtime_secs=0, max_iters=niters)
                var tphi = rphi.mean()
                rphi.print_full()
                pdaty.append(Float64((n-2)**3)/(tphi/Float64(nswaps))/1e9)
                fout.write(String(ith, n, n, n, rphi.iters(), 
                    rphi.min(), rphi.mean(), rphi.max(), "\n", sep=" "))
                #print(ith, n, n, rphi.iters(), rphi.min()/Float64(nswaps), rphi.mean()/Float64(nswaps), rphi.max()/Float64(nswaps),'\n')
                #print(pdatx[idx], tphi, rphi.iters() * nswaps, idx, pdatx[idx], pdaty[idx])
                idx += 1
            pltdata.append(pdaty)
    return pltdata
    
    
def main():
    plotperf = True
    var threads = [1,2,4,8]  # number of threads to test
    pltdata = run(180, 200, 4, 3, 10, threads)  # run with grid sizes from 4 to 64 with step 4, 3 swaps 10 iterations
    #print("pltdata", len(pltdata[1]))
    
    if not plotperf:
        return
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