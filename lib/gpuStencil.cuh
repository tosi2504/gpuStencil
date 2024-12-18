#pragma once

#include "datatypes.cuh"
#include "error.h"
#include "errorcheck.h"
#include "kernels.cuh"
#include "Accelerator.h"

template <
    unsigned N, unsigned nRhs,
    unsigned numRowTiles,  // in units of tiles
    unsigned numRhsTiles,  // in units of tiles
    unsigned rowTileSize, // in units of index
    unsigned rhsTileSize
> ret_status_t interface1 (
    complexF * const * d_d_Y, // d_d_Y[iRhs][s*N + i]
    const complexF * const * const * d_d_d_A, // d_d_d_A[s][p][i*N + j]
    const complexF * const * d_d_X, // d_d_X[iRhs][s*N + i]
    unsigned nPnts, unsigned nSites
) {
    const unsigned sizeShmem = sizeof(complexF)*(2*nRhs*N + N*numRowTiles*rowTileSize);
    auto func = ker_stencil2DBlocktilingV2<
        complexF, N, nRhs,
        numRowTiles, numRhsTiles,
        rowTileSize, rhsTileSize
    >;
    
    cudaError_t err = cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeShmem);
    if (err != cudaSuccess) {
        return GENERIC_ERROR;
    }
    
    func <<< nSites, numRowTiles*numRhsTiles, sizeShmem >>> 
        (d_d_Y, d_d_d_A, d_d_X, nPnts);
    CLCE();
    if (cudaGetLastError() != cudaSuccess) {
        return GENERIC_ERROR;
    }
    
    CCE(cudaDeviceSynchronize());
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return GENERIC_ERROR;
    }
    
    return OK;
}

template <
    unsigned N, unsigned nRhs,
    unsigned numRowTiles, unsigned numRhsTiles,
    unsigned rowTileSize, // in units of index
    unsigned rhsTileSize
> ret_status_t interface2 (
    complexF * const * d_d_Y, // d_d_Y[iRhs][s*N + i]
    const complexF * const * const * d_d_d_A, // d_d_d_A[s][p][i*N + j]
    const complexF * const * d_d_X, // d_d_X[iRhs][s*N + i]
    unsigned nPnts, unsigned nSites
) {
    using T = complexF;
    acceleratorThreads(1); // nt = 1
    unsigned blocksize = numRowTiles*numRhsTiles;
    uint64_t iter1, iter2;
    
    // here we go ... oh boy
    accelerator_for2dNB(iter1, nSites, iter2, 1, blocksize, {
        const unsigned site = blockIdx.x;
        const unsigned tIdx = threadIdx.x;
        constexpr unsigned numThreads = numRowTiles * numRhsTiles;
        assert(numThreads == blockDim.x);
        
        const unsigned dRhs = (tIdx % numRhsTiles) * rhsTileSize;  // in units of index
        const unsigned dRow = (tIdx / numRhsTiles) * rowTileSize; // in units of index
        
        __shared__ T shmem[2*nRhs*N + N*numRowTiles*rowTileSize]; // size = sizeof(T)*(2*nRhs*N + N*numRowTiles*rowTileSize)
        // __shared__ T shmem[5000]; // size = sizeof(T)*(2*nRhs*N + N*numRowTiles*rowTileSize)
        T * shmemX = shmem; // row major (lol)
        T * shmemY = shmemX + N*nRhs; // row major (lol)
        T * shmemA = shmemY + N*nRhs;  // row major (no lol) 
        
        for (unsigned i = tIdx; i < N*nRhs; i+=numThreads) {
            const unsigned iRhs = i / N;
            const unsigned n    = i % N;
            // this could perform bad -> need to check in the profiler and on its own
            // gmem accesses are coalesced but
            // shmem writes might lead to membank conflicts
            shmemX[rowm(n, iRhs, nRhs, N)] = d_d_X[iRhs][N*site + n]; 
        }

        const T * const * d_d_A = d_d_d_A[site]; // wtf is happening here
        for (unsigned p = 0; p < nPnts; p++) {
            const T * const d_A = d_d_A[p];
            for (unsigned iiRow = 0; iiRow < N; iiRow += numRowTiles*rowTileSize) {
                for (unsigned  i = tIdx; i < N*numRowTiles*rowTileSize; i+=numThreads) {
                    const unsigned iCol = i % N;
                    const unsigned iRow = i / N;
                    shmemA[rowm(iRow,iCol,N,numRowTiles*rowTileSize)] = d_A[rowm(iiRow + iRow, iCol, N, N)];
                }
                __syncthreads();

                const unsigned iidRow = iiRow + dRow; // in units of index
                if (iidRow >= N) continue; // access guard

                for (unsigned iiRhs = 0; iiRhs < nRhs; iiRhs += numRhsTiles*rhsTileSize) {
                    const unsigned iidRhs = iiRhs + dRhs; // in units of index
                    if (iidRhs >= nRhs) continue; // access guard

                    T regRes[rowTileSize][rhsTileSize] = {0.0};
                    T regX[rhsTileSize]  = {0.0}; // values of shmemX
                    T regA[rowTileSize] = {0.0}; // values of shmemA
                    for (unsigned _k = 0; _k < N; _k++) {
                        const unsigned k = _k;
                        // const unsigned k = (tIdx + _k) % N;
                        for (unsigned iTileRow = 0; iTileRow < rowTileSize; iTileRow++) {
                            regA[iTileRow] = shmemA[rowm(dRow+iTileRow, k, N, numRowTiles*rowTileSize)];
                        }
                        for (unsigned iTileRhs = 0; iTileRhs < rhsTileSize; iTileRhs++) {
                            regX[iTileRhs] = shmemX[rowm(k, iidRhs+iTileRhs, nRhs, N)];
                        }

                        // perform the arithmetics :)
                        for (unsigned iTileRow = 0; iTileRow < rowTileSize; iTileRow++) {
                            for (unsigned iTileRhs = 0; iTileRhs < rhsTileSize; iTileRhs++) {
                                // regRes[iTileRow][iTileRhs] += regA[iTileRow] * regX[iTileRhs];
                                multiply_accumulate(regRes[iTileRow][iTileRhs], regA[iTileRow], regX[iTileRhs]);
                            }
                        }
                    }

                    if (p == 0) {
                        for (unsigned iTileRow = 0; iTileRow < rowTileSize; iTileRow++) {
                            for (unsigned iTileRhs = 0; iTileRhs < rhsTileSize; iTileRhs++) {
                                const unsigned iRow = iidRow + iTileRow;
                                const unsigned iRhs = iidRhs + iTileRhs;
                                shmemY[rowm(iRow, iRhs, nRhs, N)]  = regRes[iTileRow][iTileRhs];
                            }
                        }
                    } else { 
                        for (unsigned iTileRow = 0; iTileRow < rowTileSize; iTileRow++) {
                            for (unsigned iTileRhs = 0; iTileRhs < rhsTileSize; iTileRhs++) {
                                const unsigned iRow = iidRow + iTileRow;
                                const unsigned iRhs = iidRhs + iTileRhs;
                                shmemY[rowm(iRow, iRhs, nRhs, N)] += regRes[iTileRow][iTileRhs];
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }

        for (unsigned i = tIdx; i < N*nRhs; i+=numThreads) {
            const unsigned iRhs = i / N;
            const unsigned n    = i % N;
            d_d_Y[iRhs][N*site + n] = shmemY[rowm(n, iRhs, nRhs, N)];
        }
    })

    return OK;
}

