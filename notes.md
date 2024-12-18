# libGpuStencil
We are solving the following problem:
One has a Grid with nSites(s).
One has D(d) NxN(i,j)-matrix fields A:1, ..., A:D.
One has nRhs(iRhs) N(i)-vector fields X:1, ..., X:nRhs and the same for Y.
One has nPnts(p) point mappings for every site s -> {(s:0, A:d:0), ..., (s:nPnts, A:d:nPnts)}


## Interface
How should the interface look like?
Need to consider:
- Template vs runtime parameters
- How to handle row major vs column major
- Layout of incoming data + layout changer?
- How does the geometrical information enter?

### Template vs dynamic parameters
The tile size needs to be a template parameter.
For the others I am not sure.
Best to just implement both (very easy) and then compare performance.

### Row major vs column major
Ahhhh lets just assume row major for the beginning and add an option for column major later if needed.

### Layout of incoming data
This is a particularly tricky one.
We assume the following:
1. The matrices A are contiguous in memory and the matrices are not in Grid layout -> "array of matrices"
2. The vector fields are contiguous in memory and the vectors are not in Grid layout -> "array of vectors"
Grid uses "array of matrices" for the vectors as well.
But I think we do not do that here as there is no performance increase to be expected.

### Geometrical information
There are two main approaches to interface the geometrical information
1. Pointers:  
    Basically have a list of pointers to the respective matrices for **each** stencil point as input.
    We do not need to worry about geometrics and the interface is more flexible but also less convenient.
2. Indices:
    Same as pointers but as indices for each matrix.
    Reduces the likelihood of illegal memory accesses.
3. Geometrical object:
    Pass the full geometrical information of the Grid and a stencil code.
    This is very memory safe but much more work.
I think I will go for number one :)

## Understanding Grids accelerator_for(...)
```
#define accelerator_for2dNB( iter1, num1, iter2, num2, nsimd, ... )	\
  {									\
    int nt=acceleratorThreads();					\ // nt = 2
    typedef uint64_t Iterator;						\
    auto lambda = [=] accelerator					\
      (Iterator iter1,Iterator iter2,Iterator lane) mutable {		\
      __VA_ARGS__;							\
    };									\
    dim3 cu_threads(nsimd,acceleratorThreads(),1);			\ // blockDim.x = nsimd = 32?
                                                              // blockDim.y = acceleratorThreads() = 2
                                                              // 
    dim3 cu_blocks ((num1+nt-1)/nt,num2,1);				\
    LambdaApply<<<cu_blocks,cu_threads,0,computeStream>>>(num1,num2,nsimd,lambda);	\
  }
```

map : nt = 1, num1 = numSites, num2 = 1, nsimd = blockSize


